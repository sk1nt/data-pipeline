"""Background poller for the Unusual Whales REST API.

Polls the following endpoints during RTH and stores results in Redis:

  Endpoint                             Interval (RTH)   Purpose
  ─────────────────────────────────    ───────────────  ─────────────────────────────────
  flow-alerts?has_sweep=true           15 s             Sweep/aggregated alert feed
  flow-alerts (all)                    30 s             Full alert feed (floors, repeats)
  market/market-tide                   5 min            Net call/put premium in 5-min bars
  darkpool/recent                      60 s             Block print feed
  market/sector-etfs                   10 min           Sector-level call/put flow
  market/total-options-volume          once at open     Daily call/put baseline

Off-hours:  all intervals × 12 (passive maintenance).

Rate budget at RTH cadence:
  ~8 req/min peak, ~2,800 req/day  →  14 % of the 20,000/day limit.

Redis keys written:
  uw:sweep:latest             latest sweep alert (JSON)
  uw:sweep:history            list of last 500 sweep alerts
  uw:sweep:stream             pubsub channel (new sweep payload)
  uw:alert:latest             latest full flow alert
  uw:alert:history            list of last 500 flow alerts
  uw:alert:stream             pubsub channel (new alert payload)
  uw:market_tide:latest       latest market-tide snapshot (list of 5-min bars)
  uw:market_tide:stream       pubsub channel
  uw:darkpool:latest          latest dark-pool batch
  uw:darkpool:stream          pubsub channel (new print payload)
  uw:sector_etf:latest        latest sector-ETF snapshot
  uw:sector_etf:stream        pubsub channel
  uw:options_volume:latest    latest total-options-volume (daily baseline)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, time
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import aiohttp

from src.lib.redis_client import RedisClient

LOGGER = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# ── Redis key constants ──────────────────────────────────────────────────────
SWEEP_LATEST_KEY       = "uw:sweep:latest"
SWEEP_HISTORY_KEY      = "uw:sweep:history"
SWEEP_STREAM_CHANNEL   = "uw:sweep:stream"

ALERT_LATEST_KEY       = "uw:alert:latest"
ALERT_HISTORY_KEY      = "uw:alert:history"
ALERT_STREAM_CHANNEL   = "uw:alert:stream"

TIDE_LATEST_KEY        = "uw:market_tide:latest"
TIDE_STREAM_CHANNEL    = "uw:market_tide:stream"

DARKPOOL_LATEST_KEY    = "uw:darkpool:latest"
DARKPOOL_STREAM_CHANNEL = "uw:darkpool:stream"

SECTOR_LATEST_KEY      = "uw:sector_etf:latest"
SECTOR_STREAM_CHANNEL  = "uw:sector_etf:stream"

OPTIONS_VOL_LATEST_KEY = "uw:options_volume:latest"

CACHE_TTL = 86_400          # 24 h
HISTORY_LIMIT = 500
SWEEP_MIN_PREMIUM = 100_000  # filter noise below $100k


@dataclass
class UWRestPollerSettings:
    api_key: str
    # RTH poll intervals (seconds)
    sweep_interval_rth: float = 15.0
    alert_interval_rth: float = 30.0
    tide_interval_rth: float = 300.0
    darkpool_interval_rth: float = 60.0
    sector_interval_rth: float = 600.0
    # Off-hours multiplier applied to all RTH intervals
    off_hours_multiplier: float = 12.0
    # Sweep filter threshold
    min_sweep_premium: int = SWEEP_MIN_PREMIUM
    # Per-page limit for REST calls
    page_limit: int = 25


class UWRestPoller:
    """Poll UW REST API endpoints and push results into Redis."""

    BASE_URL = "https://api.unusualwhales.com/api"

    def __init__(
        self,
        settings: UWRestPollerSettings,
        *,
        redis_client: Optional[RedisClient] = None,
    ) -> None:
        self.settings = settings
        self.redis = redis_client
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        # Track seen IDs to avoid re-publishing the same alert twice
        self._seen_sweep_ids: Set[str] = set()
        self._seen_alert_ids: Set[str] = set()
        self._seen_darkpool_ids: Set[str] = set()
        # Track timestamps for interval scheduling
        self._last_sweep: float = 0.0
        self._last_alert: float = 0.0
        self._last_tide: float = 0.0
        self._last_darkpool: float = 0.0
        self._last_sector: float = 0.0
        self._options_vol_fetched_date: Optional[date] = None
        # Rate-limit awareness
        self._daily_req_count: int = 0
        self._minute_remaining: int = 120

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._task and not self._task.done():
            LOGGER.warning("UW REST poller already running")
            return
        if not self.settings.api_key:
            LOGGER.warning("UW_API_KEY missing; UW REST poller will not start")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="uw-rest-poller")
        LOGGER.info("UW REST poller started")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            self._task = None
        LOGGER.info("UW REST poller stopped")

    def status(self) -> Dict[str, Any]:
        return {
            "running": bool(self._task and not self._task.done()),
            "daily_req_count": self._daily_req_count,
            "minute_remaining": self._minute_remaining,
        }

    # ── Main loop ────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Accept": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=12)
        connector = aiohttp.TCPConnector(limit=4, force_close=True)

        async with aiohttp.ClientSession(
            headers=headers, connector=connector, timeout=timeout
        ) as session:
            while not self._stop_event.is_set():
                now = asyncio.get_event_loop().time()
                is_rth = self._is_rth_now()
                mult = 1.0 if is_rth else self.settings.off_hours_multiplier

                try:
                    today = date.today()

                    # Once per day: total options volume baseline
                    if self._options_vol_fetched_date != today:
                        await self._fetch_options_volume(session)

                    # Sweep alerts
                    if now - self._last_sweep >= self.settings.sweep_interval_rth * mult:
                        await self._fetch_sweeps(session)
                        self._last_sweep = now

                    # Full flow alerts
                    if now - self._last_alert >= self.settings.alert_interval_rth * mult:
                        await self._fetch_alerts(session)
                        self._last_alert = now

                    # Dark pool
                    if now - self._last_darkpool >= self.settings.darkpool_interval_rth * mult:
                        await self._fetch_darkpool(session)
                        self._last_darkpool = now

                    # Market tide
                    if now - self._last_tide >= self.settings.tide_interval_rth * mult:
                        await self._fetch_market_tide(session)
                        self._last_tide = now

                    # Sector ETFs
                    if now - self._last_sector >= self.settings.sector_interval_rth * mult:
                        await self._fetch_sector_etfs(session)
                        self._last_sector = now

                except Exception:
                    LOGGER.exception("UW REST poller loop error")

                # Sleep 5 s; shorter tick so we can react to stop quickly
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue

        LOGGER.info("UW REST poller loop exited")

    # ── Fetch helpers ────────────────────────────────────────────────────────

    async def _get(
        self, session: aiohttp.ClientSession, path: str, params: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """GET helper with rate-limit header tracking."""
        url = f"{self.BASE_URL}/{path.lstrip('/')}"
        try:
            async with session.get(url, params=params) as resp:
                # Track rate-limit headers
                try:
                    self._daily_req_count = int(resp.headers.get("x-uw-daily-req-count", self._daily_req_count))
                    self._minute_remaining = int(resp.headers.get("x-uw-req-per-minute-remaining", self._minute_remaining))
                except (ValueError, TypeError):
                    pass

                if resp.status == 429:
                    LOGGER.warning("UW API rate-limited (429); backing off 60s")
                    await asyncio.sleep(60)
                    return None
                if resp.status != 200:
                    LOGGER.debug("UW API %s → HTTP %s", path, resp.status)
                    return None
                return await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            LOGGER.debug("UW API request failed (%s): %s", path, exc)
            return None

    def _redis_pipe(self):
        """Return a Redis pipeline, or None if Redis is unavailable."""
        if not self.redis:
            return None
        try:
            return self.redis.client.pipeline()
        except Exception:
            return None

    def _publish(self, channel: str, payload: str) -> None:
        if not self.redis:
            return
        try:
            self.redis.client.publish(channel, payload)
        except Exception as exc:
            LOGGER.debug("Redis publish failed (%s): %s", channel, exc)

    # ── Sweep alerts ─────────────────────────────────────────────────────────

    async def _fetch_sweeps(self, session: aiohttp.ClientSession) -> None:
        data = await self._get(
            session,
            "option-trades/flow-alerts",
            params={"has_sweep": "true", "limit": self.settings.page_limit},
        )
        if not data:
            return
        items: List[Dict] = data.get("data", [])
        new_items = []
        for item in items:
            item_id = item.get("id", "")
            if item_id and item_id in self._seen_sweep_ids:
                continue
            # Apply premium filter
            try:
                premium = float(item.get("total_premium", 0))
            except (TypeError, ValueError):
                premium = 0.0
            if premium < self.settings.min_sweep_premium:
                continue
            # Enrich with computed fields
            item = self._enrich_alert(item)
            if item_id:
                self._seen_sweep_ids.add(item_id)
                # Keep seen set bounded
                if len(self._seen_sweep_ids) > 2000:
                    self._seen_sweep_ids = set(list(self._seen_sweep_ids)[-1000:])
            new_items.append(item)

        if not new_items:
            return

        LOGGER.info("UW sweeps: %d new (filtered from %d)", len(new_items), len(items))
        pipe = self._redis_pipe()
        for item in new_items:
            serialized = json.dumps(item, default=str)
            if pipe is not None:
                pipe.setex(SWEEP_LATEST_KEY, CACHE_TTL, serialized)
                pipe.lpush(SWEEP_HISTORY_KEY, serialized)
                pipe.ltrim(SWEEP_HISTORY_KEY, 0, HISTORY_LIMIT - 1)
            self._publish(SWEEP_STREAM_CHANNEL, serialized)
        if pipe is not None:
            try:
                pipe.execute()
            except Exception as exc:
                LOGGER.debug("Redis pipe execute failed: %s", exc)

    # ── Full flow alerts ──────────────────────────────────────────────────────

    async def _fetch_alerts(self, session: aiohttp.ClientSession) -> None:
        data = await self._get(
            session,
            "option-trades/flow-alerts",
            params={"limit": self.settings.page_limit},
        )
        if not data:
            return
        items: List[Dict] = data.get("data", [])
        new_items = []
        for item in items:
            item_id = item.get("id", "")
            if item_id and item_id in self._seen_alert_ids:
                continue
            item = self._enrich_alert(item)
            if item_id:
                self._seen_alert_ids.add(item_id)
                if len(self._seen_alert_ids) > 2000:
                    self._seen_alert_ids = set(list(self._seen_alert_ids)[-1000:])
            new_items.append(item)

        if not new_items:
            return

        pipe = self._redis_pipe()
        for item in new_items:
            serialized = json.dumps(item, default=str)
            if pipe is not None:
                pipe.setex(ALERT_LATEST_KEY, CACHE_TTL, serialized)
                pipe.lpush(ALERT_HISTORY_KEY, serialized)
                pipe.ltrim(ALERT_HISTORY_KEY, 0, HISTORY_LIMIT - 1)
            self._publish(ALERT_STREAM_CHANNEL, serialized)
        if pipe is not None:
            try:
                pipe.execute()
            except Exception as exc:
                LOGGER.debug("Redis pipe execute failed: %s", exc)

    # ── Market tide ───────────────────────────────────────────────────────────

    async def _fetch_market_tide(self, session: aiohttp.ClientSession) -> None:
        data = await self._get(session, "market/market-tide")
        if not data:
            return
        bars: List[Dict] = data.get("data", [])
        if not bars:
            return

        # Compute cumulative net and running bias for the latest bar
        enriched = self._enrich_market_tide(bars)
        serialized = json.dumps(enriched, default=str)

        pipe = self._redis_pipe()
        if pipe is not None:
            pipe.setex(TIDE_LATEST_KEY, CACHE_TTL, serialized)
            try:
                pipe.execute()
            except Exception as exc:
                LOGGER.debug("Redis pipe execute failed: %s", exc)
        self._publish(TIDE_STREAM_CHANNEL, serialized)
        LOGGER.debug("UW market-tide: %d bars, latest %s", len(bars), bars[-1].get("timestamp"))

    # ── Dark pool ─────────────────────────────────────────────────────────────

    async def _fetch_darkpool(self, session: aiohttp.ClientSession) -> None:
        data = await self._get(
            session,
            "darkpool/recent",
            params={"limit": self.settings.page_limit},
        )
        if not data:
            return
        items: List[Dict] = data.get("data", [])
        new_items = []
        for item in items:
            # Use tracking_id as dedup key
            tid = str(item.get("tracking_id", ""))
            if tid and tid in self._seen_darkpool_ids:
                continue
            if tid:
                self._seen_darkpool_ids.add(tid)
                if len(self._seen_darkpool_ids) > 5000:
                    self._seen_darkpool_ids = set(list(self._seen_darkpool_ids)[-2000:])
            new_items.append(item)

        if not new_items:
            return

        LOGGER.debug("UW darkpool: %d new prints", len(new_items))
        pipe = self._redis_pipe()
        for item in new_items:
            serialized = json.dumps(item, default=str)
            if pipe is not None:
                pipe.setex(DARKPOOL_LATEST_KEY, CACHE_TTL, serialized)
            self._publish(DARKPOOL_STREAM_CHANNEL, serialized)
        if pipe is not None:
            try:
                pipe.execute()
            except Exception as exc:
                LOGGER.debug("Redis pipe execute failed: %s", exc)

    # ── Sector ETFs ───────────────────────────────────────────────────────────

    async def _fetch_sector_etfs(self, session: aiohttp.ClientSession) -> None:
        data = await self._get(session, "market/sector-etfs")
        if not data:
            return
        items: List[Dict] = data.get("data", [])
        if not items:
            return

        enriched = self._enrich_sector_etfs(items)
        serialized = json.dumps(enriched, default=str)

        pipe = self._redis_pipe()
        if pipe is not None:
            pipe.setex(SECTOR_LATEST_KEY, CACHE_TTL, serialized)
            try:
                pipe.execute()
            except Exception as exc:
                LOGGER.debug("Redis pipe execute failed: %s", exc)
        self._publish(SECTOR_STREAM_CHANNEL, serialized)
        LOGGER.debug("UW sector-etfs: %d tickers", len(items))

    # ── Total options volume (daily baseline) ─────────────────────────────────

    async def _fetch_options_volume(self, session: aiohttp.ClientSession) -> None:
        data = await self._get(session, "market/total-options-volume")
        if not data:
            return
        items: List[Dict] = data.get("data", [])
        if not items:
            return

        serialized = json.dumps(items[0], default=str)
        pipe = self._redis_pipe()
        if pipe is not None:
            pipe.setex(OPTIONS_VOL_LATEST_KEY, CACHE_TTL, serialized)
            try:
                pipe.execute()
            except Exception as exc:
                LOGGER.debug("Redis pipe execute failed: %s", exc)
        self._options_vol_fetched_date = date.today()
        LOGGER.info("UW options-volume baseline: %s", items[0])

    # ── Enrichment helpers ────────────────────────────────────────────────────

    @staticmethod
    def _enrich_alert(item: Dict[str, Any]) -> Dict[str, Any]:
        """Add computed fields useful for sweep detection and routing."""
        item = dict(item)  # shallow copy

        try:
            ask_prem = float(item.get("total_ask_side_prem", 0) or 0)
            bid_prem = float(item.get("total_bid_side_prem", 0) or 0)
            total = ask_prem + bid_prem
            item["ask_side_ratio"] = round(ask_prem / total, 4) if total > 0 else None
        except (TypeError, ValueError):
            item["ask_side_ratio"] = None

        try:
            start_ms = int(item.get("start_time", 0) or 0)
            end_ms = int(item.get("end_time", 0) or 0)
            item["execution_ms"] = end_ms - start_ms if end_ms > start_ms else None
        except (TypeError, ValueError):
            item["execution_ms"] = None

        try:
            iv_start = float(item.get("iv_start", 0) or 0)
            iv_end = float(item.get("iv_end", 0) or 0)
            item["iv_change"] = round(iv_end - iv_start, 6) if iv_start and iv_end else None
        except (TypeError, ValueError):
            item["iv_change"] = None

        # Classify sweep conviction
        ask_ratio = item.get("ask_side_ratio")
        exec_ms = item.get("execution_ms")
        has_sweep = item.get("has_sweep", False)
        premium = float(item.get("total_premium", 0) or 0)
        vol_oi = float(item.get("volume_oi_ratio", 0) or 0)

        conviction = "low"
        if has_sweep and premium >= 500_000 and ask_ratio is not None and ask_ratio >= 0.75 and vol_oi >= 0.5:
            conviction = "high"
            if exec_ms is not None and exec_ms < 100:
                conviction = "very_high"
        elif has_sweep and premium >= 100_000 and ask_ratio is not None and ask_ratio >= 0.60:
            conviction = "medium"

        item["sweep_conviction"] = conviction
        item["_enriched_at"] = datetime.now(timezone.utc).isoformat()
        return item

    @staticmethod
    def _enrich_market_tide(bars: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add cumulative net premium and running bias classification."""
        cumulative_call = 0.0
        cumulative_put = 0.0
        enriched_bars = []
        for bar in bars:
            try:
                net_call = float(bar.get("net_call_premium", 0) or 0)
                net_put = float(bar.get("net_put_premium", 0) or 0)
                cumulative_call += net_call
                cumulative_put += net_put
                bar = dict(bar)
                bar["cum_net_call_premium"] = round(cumulative_call, 2)
                bar["cum_net_put_premium"] = round(cumulative_put, 2)
                # Bias: positive net_call dominant = call-side flow winning
                bar["bar_bias"] = "call" if net_call > net_put else "put" if net_put > net_call else "neutral"
            except (TypeError, ValueError):
                pass
            enriched_bars.append(bar)

        overall_bias = "neutral"
        if cumulative_call > 0 and cumulative_call > abs(cumulative_put):
            overall_bias = "call"
        elif cumulative_put > 0 and cumulative_put > abs(cumulative_call):
            overall_bias = "put"

        return {
            "bars": enriched_bars,
            "cum_net_call_premium": round(cumulative_call, 2),
            "cum_net_put_premium": round(cumulative_put, 2),
            "overall_bias": overall_bias,
            "bar_count": len(enriched_bars),
            "_fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _enrich_sector_etfs(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute per-sector call/put ratio vs 30-day average."""
        enriched = []
        for item in items:
            item = dict(item)
            try:
                call_vol = float(item.get("call_volume", 0) or 0)
                put_vol = float(item.get("put_volume", 0) or 0)
                item["call_put_vol_ratio"] = round(call_vol / put_vol, 4) if put_vol > 0 else None

                avg_call = float(item.get("avg30_call_volume", 0) or 0)
                avg_put = float(item.get("avg30_put_volume", 0) or 0)
                item["call_vol_vs_avg"] = round(call_vol / avg_call, 3) if avg_call > 0 else None
                item["put_vol_vs_avg"] = round(put_vol / avg_put, 3) if avg_put > 0 else None

                call_prem = float(item.get("call_premium", 0) or 0)
                put_prem = float(item.get("put_premium", 0) or 0)
                item["call_put_prem_ratio"] = round(call_prem / put_prem, 4) if put_prem > 0 else None
                item["prem_bias"] = "call" if call_prem > put_prem else "put" if put_prem > call_prem else "neutral"
            except (TypeError, ValueError):
                pass
            enriched.append(item)

        return {
            "tickers": enriched,
            "_fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Scheduling helpers ────────────────────────────────────────────────────

    def _is_rth_now(self) -> bool:
        now = datetime.now(tz=_ET)
        start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        end = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return start <= now <= end
