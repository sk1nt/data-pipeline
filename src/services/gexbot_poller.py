"""Background service for polling GEXBot API at fixed intervals."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Set

import aiohttp

from ..lib.redis_client import RedisClient
from .redis_timeseries import RedisTimeSeriesClient


LOGGER = logging.getLogger(__name__)
SNAPSHOT_KEY_PREFIX = "gex:snapshot:"


@dataclass
class GEXBotPollerSettings:
    api_key: str
    symbols: List[str] = field(default_factory=lambda: ["NQ_NDX", "ES_SPX", "SPY", "QQQ", "SPX", "NDX"])
    interval_seconds: int = 60
    aggregation_period: str = "zero"
    rth_interval_seconds: int = 1
    off_hours_interval_seconds: int = 300
    dynamic_schedule: bool = True


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
        self._dynamic_symbols: Set[str] = self._load_dynamic_symbols()
        self._supported_symbols: Set[str] = set(self._base_symbols)
        self._last_supported_refresh: Optional[date] = None
        self.snapshot_count = 0
        self.last_snapshot_ts: Optional[str] = None
        self._last_interval_setting: Optional[str] = None

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
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            await self._refresh_supported_symbols(session)
            while not self._stop_event.is_set():
                if self._needs_supported_refresh():
                    await self._refresh_supported_symbols(session)
                symbols = sorted(self._base_symbols | self._dynamic_symbols)
                for symbol in symbols:
                    try:
                        snapshot = await self._fetch_symbol(session, symbol)
                        if snapshot:
                            self.latest[symbol.upper()] = snapshot
                            await self._record_timeseries(snapshot)
                    except Exception:  # pragma: no cover - defensive logging
                        LOGGER.exception("Failed to poll GEXBot for %s", symbol)
                interval_seconds = self._current_interval_seconds()
                label = "RTH" if self._is_rth_now() else "off-hours"
                if label != self._last_interval_setting:
                    LOGGER.info("GEXBot poller interval set to %ss (%s)", interval_seconds, label)
                    self._last_interval_setting = label
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=interval_seconds
                    )
                except asyncio.TimeoutError:
                    continue
        LOGGER.info("GEXBot poller stopped")

    def _current_interval_seconds(self) -> int:
        if not self.settings.dynamic_schedule:
            return self.settings.interval_seconds
        if self._is_rth_now():
            return self.settings.rth_interval_seconds
        return self.settings.off_hours_interval_seconds

    def _is_rth_now(self) -> bool:
        try:
            eastern = ZoneInfo("America/New_York")
        except Exception:  # pragma: no cover - zoneinfo fallback
            eastern = timezone.utc
        now = datetime.now(tz=eastern)
        start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        end = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return start <= now <= end

    def add_symbol_for_day(self, symbol: str) -> None:
        """Auto-enroll a symbol for polling until the next midnight."""
        normalized = symbol.upper().strip()
        if not normalized or normalized in self._base_symbols:
            return
        if normalized in self._dynamic_symbols:
            return
        self._dynamic_symbols.add(normalized)
        LOGGER.info("Enrolled %s for GEX polling (dynamic set)", normalized)
        self._persist_dynamic_symbols()

    async def fetch_symbol_now(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch a symbol immediately and enroll it for the remainder of the day."""
        normalized = symbol.upper().strip()
        if not normalized:
            return None
        timeout = aiohttp.ClientTimeout(total=12)
        connector = aiohttp.TCPConnector(limit=2, force_close=True)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            snapshot = await self._fetch_symbol(session, normalized)
        if snapshot:
            self.latest[normalized] = snapshot
            self.add_symbol_for_day(normalized)
            await self._record_timeseries(snapshot)
        return snapshot

    async def _fetch_symbol(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        base_url = f"https://api.gexbot.com/{symbol}/classic/{self.settings.aggregation_period}"

        async def _endpoint(suffix: str = "") -> Optional[Dict[str, Any]]:
            url = f"{base_url}{suffix}?key={self.settings.api_key}"
            async with session.get(url) as resp:
                if resp.status != 200:
                    LOGGER.debug(
                        "GEXBot %s%s returned %s",
                        symbol,
                        suffix or "",
                        resp.status,
                    )
                    return None
                return await resp.json()

        zero = await _endpoint()
        majors = await _endpoint("/majors") if zero else None
        maxchange = await _endpoint("/maxchange") if zero else None
        if not zero and not majors and not maxchange:
            return None

        result = self._combine_payloads(symbol, zero, majors, maxchange)
        result["raw"] = {"zero": zero, "majors": majors, "maxchange": maxchange}
        return result

    async def _record_timeseries(self, snapshot: Dict[str, Any]) -> None:
        if not self.ts_client:
            return
        await asyncio.to_thread(self._write_snapshot_series, snapshot)

    def _combine_payloads(
        self,
        symbol: str,
        zero: Optional[Dict[str, Any]],
        majors: Optional[Dict[str, Any]],
        maxchange: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        def _first(*values):
            for value in values:
                if value not in (None, "", []):
                    return value
            return None

        timestamp = _first(
            zero.get("timestamp") if zero else None,
            majors.get("timestamp") if majors else None,
            maxchange.get("timestamp") if maxchange else None,
        )
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()

        zero_payload = zero or {}
        majors_payload = majors or {}
        maxchange_payload = maxchange or {}
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
            majors_payload.get("net_gex_vol"),
            majors_payload.get("net_gex"),
            zero_payload.get("net_gex"),
            zero_payload.get("net_gex_vol"),
            zero_payload.get("sum_gex_vol"),
        )
        net_gex_oi = _first(
            majors_payload.get("net_gex_oi"),
            zero_payload.get("net_gex_oi"),
            zero_payload.get("sum_gex_oi"),
        )

        snapshot = {
            "symbol": symbol.upper(),
            "timestamp": timestamp,
            "spot": _first(
                zero_payload.get("spot"),
                majors_payload.get("spot"),
                maxchange_payload.get("spot"),
            ),
            "zero_gamma": _first(
                zero_payload.get("zero_gamma"),
                maxchange_payload.get("zero_gamma"),
            ),
            "net_gex": net_gex,
            "net_gex_oi": net_gex_oi,
            "sum_gex_vol": zero_payload.get("sum_gex_vol"),
            "sum_gex_oi": zero_payload.get("sum_gex_oi"),
            "major_pos_vol": _first(
                majors_payload.get("major_pos_vol"),
                majors_payload.get("mpos_vol"),
                zero_payload.get("major_pos_vol"),
                zero_payload.get("mpos_vol"),
            ),
            "major_neg_vol": _first(
                majors_payload.get("major_neg_vol"),
                majors_payload.get("mneg_vol"),
                zero_payload.get("major_neg_vol"),
                zero_payload.get("mneg_vol"),
            ),
            "major_pos_oi": _first(
                majors_payload.get("major_pos_oi"),
                majors_payload.get("mpos_oi"),
                zero_payload.get("major_pos_oi"),
                zero_payload.get("mpos_oi"),
            ),
            "major_neg_oi": _first(
                majors_payload.get("major_neg_oi"),
                majors_payload.get("mneg_oi"),
                zero_payload.get("major_neg_oi"),
                zero_payload.get("mneg_oi"),
            ),
            "major_pos_strike": _first(
                majors_payload.get("major_pos_strike"),
                zero_payload.get("major_pos_strike"),
                maxchange_payload.get("major_pos_strike"),
            ),
            "major_neg_strike": _first(
                majors_payload.get("major_neg_strike"),
                zero_payload.get("major_neg_strike"),
                maxchange_payload.get("major_neg_strike"),
            ),
            "delta_risk_reversal": _first(
                zero_payload.get("delta_risk_reversal"),
                maxchange_payload.get("delta_risk_reversal"),
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
        for field, value in metrics.items():
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            samples.append((
                f"ts:gex:{field}:{symbol}",
                timestamp_ms,
                numeric,
                {"symbol": symbol, "type": "gex", "field": field},
            ))

        maxchange = snapshot.get("maxchange") or {}
        if isinstance(maxchange, dict):
            for window, data in maxchange.items():
                if isinstance(data, (list, tuple)) and len(data) == 2:
                    strike, delta = data
                    try:
                        samples.append((
                            f"ts:gex:maxchange:{symbol}:{window}:delta",
                            timestamp_ms,
                            float(delta or 0.0),
                            {"symbol": symbol, "type": "gex", "field": f"maxchange_{window}_delta"},
                        ))
                        samples.append((
                            f"ts:gex:maxchange:{symbol}:{window}:strike",
                            timestamp_ms,
                            float(strike or 0.0),
                            {"symbol": symbol, "type": "gex", "field": f"maxchange_{window}_strike"},
                        ))
                    except (TypeError, ValueError):
                        continue
        if samples:
            ts_client.multi_add(samples)
        self._store_snapshot_blob(snapshot)
        self.snapshot_count += 1
        self.last_snapshot_ts = snapshot.get("timestamp") or datetime.utcnow().isoformat()

    def _store_snapshot_blob(self, snapshot: Dict[str, Any]) -> None:
        if not self.redis:
            return
        symbol = snapshot.get("symbol") or snapshot.get("ticker")
        if not symbol:
            return
        key = f"{SNAPSHOT_KEY_PREFIX}{symbol.upper()}"
        try:
            payload = json.dumps(snapshot)
            self.redis.client.set(key, payload)
        except Exception:
            LOGGER.warning("Failed to cache GEX snapshot for %s", symbol, exc_info=True)

    def _load_dynamic_symbols(self) -> Set[str]:
        if not self.redis:
            return set()
        cached = self.redis.get_cached(self.dynamic_key) or []
        return {str(symbol).upper() for symbol in cached}

    def _persist_dynamic_symbols(self) -> None:
        if not self.redis:
            return
        ttl = max(60, self._seconds_until_midnight())
        payload = sorted(self._dynamic_symbols)
        self.redis.set_cached(self.dynamic_key, payload, ttl_seconds=ttl)

    def _seconds_until_midnight(self) -> int:
        now = datetime.utcnow()
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return int((tomorrow - now).total_seconds())

    def _needs_supported_refresh(self) -> bool:
        today = datetime.utcnow().date()
        return self._last_supported_refresh != today

    async def _refresh_supported_symbols(self, session: aiohttp.ClientSession) -> None:
        symbols = await self._fetch_supported_symbol_list(session)
        if symbols:
            self._supported_symbols = {s.upper() for s in symbols}
            if self.redis:
                ttl = self._seconds_until_midnight()
                self.redis.set_cached(self.supported_key, sorted(self._supported_symbols), ttl_seconds=ttl)
        else:
            self._supported_symbols = {s.upper() for s in self.settings.symbols}
        self._dynamic_symbols = self._load_dynamic_symbols()
        self._last_supported_refresh = datetime.utcnow().date()
        LOGGER.info("Updated supported GEX symbols: %s", sorted(self._supported_symbols))

    async def _fetch_supported_symbol_list(self, session: aiohttp.ClientSession) -> Optional[List[str]]:
        if not self.settings.api_key:
            return None
        url = f"https://api.gexbot.com/metadata/symbols?key={self.settings.api_key}"
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    LOGGER.debug("GEXBot symbol metadata failed: %s", resp.status)
                    return None
                data = await resp.json()
        except Exception:
            LOGGER.exception("Failed to fetch GEX symbol metadata")
            return None

        if isinstance(data, list):
            return [str(item) for item in data]
        if isinstance(data, dict):
            symbols = data.get("symbols") or data.get("data")
            if isinstance(symbols, list):
                return [str(item) for item in symbols]
        return None

    def status(self) -> Dict[str, Any]:
        running = self._task is not None and not self._task.done()
        return {
            "running": running,
            "snapshot_count": self.snapshot_count,
            "last_snapshot_ts": self.last_snapshot_ts,
            "base_symbols": sorted(self._base_symbols),
            "dynamic_symbols": sorted(self._dynamic_symbols),
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
