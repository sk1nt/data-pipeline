"""Background service for polling GEXBot API at fixed intervals."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set

import aiohttp

from ..lib.redis_client import RedisClient
from .redis_timeseries import RedisTimeSeriesClient


LOGGER = logging.getLogger(__name__)


@dataclass
class GEXBotPollerSettings:
    api_key: str
    symbols: List[str] = field(default_factory=lambda: ["NQ_NDX", "ES_SPX", "SPY", "QQQ", "SPX", "NDX"])
    interval_seconds: int = 60


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
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self.settings.interval_seconds
                    )
                except asyncio.TimeoutError:
                    continue
        LOGGER.info("GEXBot poller stopped")

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
        base_url = f"https://api.gexbot.com/{symbol}/classic"

        async def _endpoint(name: str) -> Optional[Dict[str, Any]]:
            url = f"{base_url}/{name}?key={self.settings.api_key}"
            async with session.get(url) as resp:
                if resp.status != 200:
                    LOGGER.debug("GEXBot %s %s returned %s", symbol, name, resp.status)
                    return None
                return await resp.json()

        zero = await _endpoint("zero")
        majors = await _endpoint("majors") if zero else None
        maxchange = await _endpoint("maxchange") if zero else None
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

        return {
            "symbol": symbol.upper(),
            "timestamp": timestamp,
            "spot": _first(zero.get("spot") if zero else None, majors.get("spot") if majors else None),
            "zero_gamma": _first(zero.get("zero_gamma") if zero else None, maxchange.get("zero_gamma") if maxchange else None),
            "net_gex": _first(zero.get("net_gex") if zero else None, zero.get("net_gex_vol") if zero else None),
            "major_pos_vol": _first(
                majors.get("major_pos_vol") if majors else None,
                maxchange.get("major_pos_vol") if maxchange else None,
            ),
            "major_neg_vol": _first(
                majors.get("major_neg_vol") if majors else None,
                maxchange.get("major_neg_vol") if maxchange else None,
            ),
            "major_pos_oi": _first(
                majors.get("major_pos_oi") if majors else None,
                maxchange.get("major_pos_oi") if maxchange else None,
            ),
            "major_neg_oi": _first(
                majors.get("major_neg_oi") if majors else None,
                maxchange.get("major_neg_oi") if maxchange else None,
            ),
            "major_pos_strike": _first(
                majors.get("major_pos_strike") if majors else None,
                maxchange.get("major_pos_strike") if maxchange else None,
            ),
            "major_neg_strike": _first(
                majors.get("major_neg_strike") if majors else None,
                maxchange.get("major_neg_strike") if maxchange else None,
            ),
            "delta_risk_reversal": _first(
                zero.get("delta_risk_reversal") if zero else None,
                maxchange.get("delta_risk_reversal") if maxchange else None,
            ),
            "maxchange": maxchange or {},
        }

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
        self.snapshot_count += 1
        self.last_snapshot_ts = snapshot.get("timestamp") or datetime.utcnow().isoformat()

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
        return {
            "running": self._task is not None and not self._task.done(),
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
