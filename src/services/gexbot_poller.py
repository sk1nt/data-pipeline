"""Background service for polling GEXBot API at fixed intervals."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from datetime import date, datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set
import threading
from zoneinfo import ZoneInfo

import aiohttp

from lib.redis_client import RedisClient
from .redis_timeseries import RedisTimeSeriesClient


LOGGER = logging.getLogger(__name__)
# Allow poller-only debug logging via env without touching global log level.
# Attach a dedicated handler so DEBUG messages are always emitted even when the
# root logger is set to INFO by uvicorn/basicConfig.
if os.getenv("GEXBOT_POLLER_DEBUG", "").lower() == "true":
    LOGGER.setLevel(logging.DEBUG)
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    LOGGER.addHandler(_handler)
    LOGGER.propagate = False
SNAPSHOT_KEY_PREFIX = "gex:snapshot:"
SNAPSHOT_PUBSUB_CHANNEL = "gex:snapshot:stream"


@dataclass
class GEXBotPollerSettings:
    api_key: str
    symbols: List[str] = field(
        default_factory=lambda: ["NQ_NDX", "ES_SPX", "SPY", "QQQ", "SPX", "NDX"]
    )
    interval_seconds: float = 60.0
    aggregation_period: str = "zero"
    rth_interval_seconds: float = 1.0
    off_hours_interval_seconds: float = 300.0
    dynamic_schedule: bool = True
    exclude_symbols: List[str] = field(default_factory=list)
    sierra_chart_output_path: Optional[str] = None
    auto_refresh_symbols: bool = True


class GEXBotPoller:
    """Poll GEXBot classic endpoints and cache latest snapshots."""

    def __init__(
        self,
        settings: GEXBotPollerSettings,
        *,
        redis_client: Optional[RedisClient] = None,
        ts_client: Optional[RedisTimeSeriesClient] = None,
        dynamic_key: str = "gexbot:symbols:dynamic",
        supported_key: str = "gexbot:symbols:supported",
    ) -> None:
        self.settings = settings
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self.latest: Dict[str, Dict[str, Any]] = {}
        self.redis = redis_client
        self.ts_client = ts_client
        self.dynamic_key = dynamic_key
        self.supported_key = supported_key
        self._base_symbols: Set[str] = {s.upper() for s in settings.symbols}
        # No dynamic symbols: always use supported symbol list (downloaded) or base symbols
        self._dynamic_symbols: Dict[str, datetime] = {}
        self._supported_symbols: Set[str] = set(self._base_symbols)
        self._last_supported_refresh: Optional[date] = None
        self.snapshot_count = 0
        self.last_snapshot_ts: Optional[str] = None
        # Track last written timestamp per symbol (ms) to ensure monotonic writes
        self._last_snapshot_ms_by_symbol: Dict[str, int] = {}
        self._snapshot_lock = threading.Lock()
        self._last_interval_setting: Optional[str] = None
        self._auto_refresh_symbols = getattr(
            self.settings, "auto_refresh_symbols", True
        )

    def start(self) -> None:
        if self._task and not self._task.done():
            LOGGER.warning("GEXBot poller already running")
            return
        if not self.settings.api_key:
            LOGGER.warning("GEXBot API key missing; poller will not start")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="gexbot-poller")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task:
            await self._task
            self._task = None

    async def _run(self) -> None:
        LOGGER.info(
            "Starting GEXBot poller (base_symbols=%s interval=%ss)",
            ",".join(sorted(self._base_symbols)),
            self.settings.interval_seconds,
        )
        timeout = aiohttp.ClientTimeout(total=12)
        connector = aiohttp.TCPConnector(limit=8, force_close=True)
        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            if self._auto_refresh_symbols:
                await self._refresh_supported_symbols(session)
            while not self._stop_event.is_set():
                loop_start = asyncio.get_event_loop().time()
                interval_seconds = self._current_interval_seconds()
                if self._auto_refresh_symbols and self._needs_supported_refresh():
                    await self._refresh_supported_symbols(session)
                # Always poll the canonical supported symbols (downloaded list); dynamic adds removed
                # For NQ poller (base symbols include NQ_NDX), prefer a very fast RTH poll of the
                # key symbols to reduce load: only ['SPX','NQ_NDX','ES_SPX'] during RTH; otherwise poll all.
                if "NQ_NDX" in self._base_symbols:
                    symbols = (
                        ["SPX", "NQ_NDX", "ES_SPX"]
                        if self._is_rth_now()
                        else sorted(self._supported_symbols or self._base_symbols)
                    )
                else:
                    symbols = sorted(self._supported_symbols or self._base_symbols)
                LOGGER.debug(
                    "poll-loop symbols=%s interval=%.3fs", symbols, interval_seconds
                )
                for symbol in symbols:
                    try:
                        fetch_start = asyncio.get_event_loop().time()
                        snapshot = await self._fetch_symbol(session, symbol)
                        if snapshot:
                            self.latest[symbol.upper()] = snapshot
                            await self._record_timeseries(snapshot)
                            LOGGER.debug(
                                "fetched %s ts=%s in %.3fs",
                                symbol,
                                snapshot.get("timestamp"),
                                asyncio.get_event_loop().time() - fetch_start,
                            )
                        else:
                            LOGGER.debug("no snapshot returned for %s", symbol)
                    except Exception:  # pragma: no cover - defensive logging
                        LOGGER.exception("Failed to poll GEXBot for %s", symbol)
                interval_seconds = self._current_interval_seconds()
                label = "RTH" if self._is_rth_now() else "off-hours"
                if label != self._last_interval_setting:
                    LOGGER.info(
                        "GEXBot poller interval set to %ss (%s)",
                        interval_seconds,
                        label,
                    )
                    self._last_interval_setting = label
                # compensate for work time so we don't drift and miss ticks
                elapsed = asyncio.get_event_loop().time() - loop_start
                remaining = max(0.05, interval_seconds - elapsed)
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    continue
        LOGGER.info("GEXBot poller stopped")

    def _current_interval_seconds(self) -> float:
        """Return the effective poll interval (seconds), clamped to >0."""

        def _clamp(value: float) -> float:
            return max(0.1, float(value))

        if not self.settings.dynamic_schedule:
            return _clamp(self.settings.interval_seconds)
        if self._is_rth_now():
            return _clamp(self.settings.rth_interval_seconds)
        return _clamp(self.settings.off_hours_interval_seconds)

    def _is_rth_now(self) -> bool:
        try:
            eastern = ZoneInfo("America/New_York")
        except Exception:  # pragma: no cover - zoneinfo fallback
            eastern = timezone.utc
        now = datetime.now(tz=eastern)
        start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        end = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return start <= now <= end

    # Dynamic enrollment removed — symbol set is derived from the supported symbols

    async def fetch_symbol_now(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch a symbol immediately and return the normalized snapshot."""
        normalized = symbol.upper().strip()
        if not normalized:
            return None
        timeout = aiohttp.ClientTimeout(total=12)
        connector = aiohttp.TCPConnector(limit=2, force_close=True)
        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            snapshot = await self._fetch_symbol(session, normalized)
        if snapshot:
            self.latest[normalized] = snapshot
            await self._record_timeseries(snapshot)
        return snapshot

    async def _fetch_symbol(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        base_url = f"https://api.gexbot.com/{symbol}/classic/{self.settings.aggregation_period}"

        async def _endpoint() -> Optional[Dict[str, Any]]:
            url = f"{base_url}?key={self.settings.api_key}"
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        LOGGER.debug(
                            "GEXBot %s returned %s",
                            symbol,
                            resp.status,
                        )
                        return None
                    return await resp.json()
            except asyncio.CancelledError:
                raise
            except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                LOGGER.debug("GEXBot %s request failed: %s", symbol, exc)
                return None

        zero = await _endpoint()
        if not zero:
            LOGGER.debug("GEXBot %s returned no data", symbol)
            return None

        result = self._combine_payloads(symbol, zero)
        result["raw"] = {"zero": zero}
        return result

    async def _record_timeseries(self, snapshot: Dict[str, Any]) -> None:
        if self.ts_client:
            await asyncio.to_thread(self._write_snapshot_series, snapshot)
        else:
            # Even without a timeseries sink, persist the snapshot for downstream consumers
            self._store_snapshot_blob(snapshot)
            self.snapshot_count += 1
            self.last_snapshot_ts = (
                snapshot.get("timestamp") or datetime.utcnow().isoformat()
            )

    def _combine_payloads(
        self,
        symbol: str,
        zero: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        def _first(*values):
            for value in values:
                if value not in (None, "", []):
                    return value
            return None

        zero_payload = zero or {}
        timestamp = _first(zero_payload.get("timestamp"))
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(
                float(timestamp), tz=timezone.utc
            ).isoformat()
        maxchange_payload = (
            zero_payload.get("maxchange") if isinstance(zero_payload, dict) else {}
        )
        maxchange_windows: Dict[str, list[float]] = {}
        if isinstance(maxchange_payload, dict):
            for window, data in maxchange_payload.items():
                if window in {"ticker", "timestamp"}:
                    continue
                if isinstance(data, (list, tuple)) and len(data) == 2:
                    try:
                        strike = float(data[0]) if data[0] is not None else None
                        delta = float(data[1]) if data[1] is not None else None
                    except (TypeError, ValueError):
                        continue
                    if strike is None or delta is None:
                        continue
                    maxchange_windows[window] = [strike, delta]

        net_gex = _first(
            zero_payload.get("net_gex"),
            zero_payload.get("net_gex_vol"),
            zero_payload.get("sum_gex_vol"),
        )
        net_gex_oi = _first(
            zero_payload.get("net_gex_oi"),
            zero_payload.get("sum_gex_oi"),
        )

        snapshot = {
            "symbol": symbol.upper(),
            "timestamp": timestamp,
            "spot": _first(
                zero_payload.get("spot"),
            ),
            "zero_gamma": _first(
                zero_payload.get("zero_gamma"),
            ),
            "net_gex": net_gex,
            "net_gex_oi": net_gex_oi,
            "sum_gex_vol": zero_payload.get("sum_gex_vol"),
            "sum_gex_oi": zero_payload.get("sum_gex_oi"),
            "major_pos_vol": _first(
                zero_payload.get("major_pos_vol"),
            ),
            "major_neg_vol": _first(
                zero_payload.get("major_neg_vol"),
            ),
            "major_pos_oi": _first(
                zero_payload.get("major_pos_oi"),
            ),
            "major_neg_oi": _first(
                zero_payload.get("major_neg_oi"),
            ),
            "major_pos_strike": _first(
                zero_payload.get("major_pos_strike"),
            ),
            "major_neg_strike": _first(
                zero_payload.get("major_neg_strike"),
            ),
            "delta_risk_reversal": _first(
                zero_payload.get("delta_risk_reversal"),
            ),
            "maxchange": maxchange_windows,
        }
        snapshot["net_gex_vol"] = snapshot["net_gex"]
        snapshot["major_pos"] = snapshot["major_pos_vol"]
        snapshot["major_neg"] = snapshot["major_neg_vol"]
        snapshot["ticker"] = snapshot["symbol"]
        snapshot["min_dte"] = zero_payload.get("min_dte")
        snapshot["sec_min_dte"] = zero_payload.get("sec_min_dte")
        snapshot["max_priors"] = zero_payload.get("max_priors")
        snapshot["strikes"] = zero_payload.get("strikes")
        return snapshot

    def _write_snapshot_series(self, snapshot: Dict[str, Any]) -> None:
        ts_client = self.ts_client
        if not ts_client:
            return
        symbol = snapshot.get("symbol", "UNKNOWN").upper()
        timestamp = snapshot.get("timestamp") or datetime.utcnow().isoformat()
        timestamp_ms = _timestamp_ms(timestamp)
        symbol = snapshot.get("symbol", "UNKNOWN").upper()
        # Ensure timestamps are strictly increasing for the symbol to avoid TS.ADD overwrites
        with self._snapshot_lock:
            last_ms = self._last_snapshot_ms_by_symbol.get(symbol)
            if last_ms is not None and timestamp_ms <= last_ms:
                timestamp_ms = last_ms + 1
            self._last_snapshot_ms_by_symbol[symbol] = timestamp_ms
        metrics = {
            "spot": snapshot.get("spot"),
            "zero_gamma": snapshot.get("zero_gamma"),
            "net_gex": snapshot.get("net_gex"),
            "net_gex_oi": snapshot.get("net_gex_oi"),
            "sum_gex_vol": snapshot.get("sum_gex_vol"),
            "sum_gex_oi": snapshot.get("sum_gex_oi"),
            "major_pos_vol": snapshot.get("major_pos_vol"),
            "major_neg_vol": snapshot.get("major_neg_vol"),
            "major_pos_oi": snapshot.get("major_pos_oi"),
            "major_neg_oi": snapshot.get("major_neg_oi"),
            "delta_risk_reversal": snapshot.get("delta_risk_reversal"),
        }
        samples = []
        for metric_name, value in metrics.items():
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            samples.append(
                (
                    f"ts:gex:{metric_name}:{symbol}",
                    timestamp_ms,
                    numeric,
                    {"symbol": symbol, "type": "gex", "field": metric_name},
                )
            )

        maxchange = snapshot.get("maxchange") or {}
        if isinstance(maxchange, dict):
            for window, data in maxchange.items():
                if isinstance(data, (list, tuple)) and len(data) == 2:
                    strike, delta = data
                    try:
                        samples.append(
                            (
                                f"ts:gex:maxchange:{symbol}:{window}:delta",
                                timestamp_ms,
                                float(delta or 0.0),
                                {
                                    "symbol": symbol,
                                    "type": "gex",
                                    "field": f"maxchange_{window}_delta",
                                },
                            )
                        )
                        samples.append(
                            (
                                f"ts:gex:maxchange:{symbol}:{window}:strike",
                                timestamp_ms,
                                float(strike or 0.0),
                                {
                                    "symbol": symbol,
                                    "type": "gex",
                                    "field": f"maxchange_{window}_strike",
                                },
                            )
                        )
                    except (TypeError, ValueError):
                        continue
        if samples:
            ts_client.multi_add(samples)
        self._store_snapshot_blob(snapshot)
        self.snapshot_count += 1
        self.last_snapshot_ts = (
            snapshot.get("timestamp") or datetime.utcnow().isoformat()
        )

    def _store_snapshot_blob(self, snapshot: Dict[str, Any]) -> None:
        symbol = snapshot.get("symbol") or snapshot.get("ticker")
        if not symbol:
            return
        key = f"{SNAPSHOT_KEY_PREFIX}{symbol.upper()}"
        
        # Don't overwrite good snapshots with incomplete data (after market close)
        # If net_gex is null/zero and major_pos_vol is zero, this is likely stale/incomplete
        net_gex = snapshot.get("net_gex")
        major_pos_vol = snapshot.get("major_pos_vol") or 0
        major_neg_vol = snapshot.get("major_neg_vol") or 0
        
        if net_gex is None and major_pos_vol == 0 and major_neg_vol == 0:
            # Check if we have a better snapshot already cached
            if self.redis:
                try:
                    existing = self.redis.client.get(key)
                    if existing:
                        existing_data = json.loads(existing)
                        existing_net_gex = existing_data.get("net_gex")
                        existing_pos_vol = existing_data.get("major_pos_vol") or 0
                        
                        # Preserve existing snapshot if it has volume data
                        if existing_net_gex is not None or existing_pos_vol > 0:
                            LOGGER.info(
                                "Preserving existing snapshot for %s (has volume data)",
                                symbol
                            )
                            return
                except Exception as e:
                    LOGGER.debug("Failed to check existing snapshot: %s", e)
        
        if self.redis:
            try:
                payload = json.dumps(snapshot)
                self.redis.client.set(key, payload)
                try:
                    # Publish full snapshot for downstream consumers (Discord feed, websocket, etc.)
                    self.redis.client.publish(SNAPSHOT_PUBSUB_CHANNEL, payload)
                except Exception:
                    LOGGER.debug(
                        "Failed to publish snapshot for %s", symbol, exc_info=True
                    )
            except Exception:
                LOGGER.warning(
                    "Failed to cache GEX snapshot for %s", symbol, exc_info=True
                )
        self._write_sierra_chart_bridge(snapshot)

    def _write_sierra_chart_bridge(self, snapshot: Dict[str, Any]) -> None:
        """Optionally write NQ_NDX sum_gex_vol into Sierra Chart JSON bridge."""
        path = getattr(self.settings, "sierra_chart_output_path", None)
        if not path:
            return
        symbol = (snapshot.get("symbol") or snapshot.get("ticker") or "").upper()
        if symbol != "NQ_NDX":
            return
        value = snapshot.get("sum_gex_vol")
        if value is None:
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            LOGGER.debug(
                "Skipping Sierra Chart write; sum_gex_vol not numeric: %s", value
            )
            return
        payload = {"sum_gex_vol": numeric}
        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception:
            LOGGER.warning(
                "Failed to write Sierra Chart bridge file at %s", path, exc_info=True
            )

    # dynamic symbol helpers removed

    # dynamic symbol persistence removed

    def _parse_expiry(self, raw: Any) -> Optional[datetime]:
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        if isinstance(raw, str):
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        return None

    async def _sync_dynamic_symbols(
        self, session: Optional[aiohttp.ClientSession]
    ) -> None:
        """Detect dynamic symbols added externally and fetch snapshots for newly added ones.

        This method reads the dynamic symbols list from Redis, computes any newly
        added symbols (relative to in-memory dynamic map), and makes an immediate
        fetch for each new symbol so a canonical snapshot is stored and timeseries
        are recorded.
        """
        # Dynamic synchronization disabled — this method is intentionally a no-op.
        return

    def _prune_expired_dynamic_symbols(self) -> None:
        # No dynamic enrollment; nothing to prune
        return

    def _seconds_until_midnight(self) -> int:
        now = datetime.utcnow()
        tomorrow = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return int((tomorrow - now).total_seconds())

    def _needs_supported_refresh(self) -> bool:
        today = datetime.utcnow().date()
        return self._last_supported_refresh != today

    async def _refresh_supported_symbols(self, session: aiohttp.ClientSession) -> None:
        if not self._auto_refresh_symbols:
            # Respect explicitly configured symbols without reaching the global tickers endpoint
            self._supported_symbols = set(self._base_symbols)
            self._dynamic_symbols = {}
            self._last_supported_refresh = datetime.utcnow().date()
            LOGGER.info(
                "Auto-refresh disabled; using configured symbols: %s",
                sorted(self._supported_symbols),
            )
            return
        symbols = await self._fetch_supported_symbol_list(session)
        if symbols:
            self._supported_symbols = {s.upper() for s in symbols}
            # Apply any configured exclusions (e.g., exclude NQ_NDX from main poller)
            if getattr(self.settings, "exclude_symbols", None):
                excluded = {s.upper() for s in self.settings.exclude_symbols}
                self._supported_symbols = {
                    s for s in self._supported_symbols if s not in excluded
                }
            if self.redis:
                ttl = self._seconds_until_midnight()
                self.redis.set_cached(
                    self.supported_key, sorted(self._supported_symbols), ttl_seconds=ttl
                )
        else:
            self._supported_symbols = {s.upper() for s in self.settings.symbols}
            # Apply configured exclusions even when falling back to base symbols
            if getattr(self.settings, "exclude_symbols", None):
                excluded = {s.upper() for s in self.settings.exclude_symbols}
                self._supported_symbols = {
                    s for s in self._supported_symbols if s not in excluded
                }
        # No dynamic symbols: clear any in-memory map (dynamic enrollment removed)
        self._dynamic_symbols = {}
        self._last_supported_refresh = datetime.utcnow().date()
        LOGGER.info(
            "Updated supported GEX symbols: %s", sorted(self._supported_symbols)
        )

    async def _fetch_supported_symbol_list(
        self, session: aiohttp.ClientSession
    ) -> Optional[List[str]]:
        if not self.settings.api_key:
            return None
        url = "https://api.gexbot.com/tickers"
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    LOGGER.debug("GEXBot tickers failed: %s", resp.status)
                    return None
                data = await resp.json()
        except Exception:
            LOGGER.exception("Failed to fetch GEX tickers")
            return None

        if isinstance(data, dict):
            all_symbols = []
            for key in ["stocks", "indexes", "futures"]:
                symbols = data.get(key)
                if isinstance(symbols, list):
                    all_symbols.extend(str(item) for item in symbols)
            if all_symbols:
                return all_symbols
        return None

    def status(self) -> Dict[str, Any]:
        running = self._task is not None and not self._task.done()
        return {
            "running": running,
            "snapshot_count": self.snapshot_count,
            "last_snapshot_ts": self.last_snapshot_ts,
            "base_symbols": sorted(self._base_symbols),
        }


def _timestamp_ms(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(float(value))
    if isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except ValueError:
            pass
    return int(datetime.utcnow().timestamp() * 1000)
