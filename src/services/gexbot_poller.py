"""Background service for polling GEXBot API at fixed intervals."""

from __future__ import annotations

import asyncio
import json
import math
import logging
import os
import socket
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
import threading
from zoneinfo import ZoneInfo

import aiohttp

from lib.redis_client import RedisClient
from .gex_duckdb_writer import GEXDuckDBWriter
from .gex_wall_utils import build_compact_wall_fields
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


def _sanitize_floats(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None for JSON compliance."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_floats(v) for v in obj]
    return obj


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None



@dataclass
class GEXBotPollerSettings:
    api_key: str
    symbols: List[str] = field(
        default_factory=lambda: ["NQ_NDX", "ES_SPX", "SPY", "QQQ", "NDX"]
    )
    interval_seconds: float = 60.0
    aggregation_period: str = "zero"
    rth_interval_seconds: float = 1.0
    off_hours_interval_seconds: float = 300.0
    dynamic_schedule: bool = True
    exclude_symbols: List[str] = field(default_factory=list)
    auto_refresh_symbols: bool = True
    rth_overlap_enabled: bool = False
    same_timestamp_retry_seconds: float = 0.25


class GEXBotPoller:
    """Poll GEXBot classic endpoints and cache latest snapshots."""

    def __init__(
        self,
        settings: GEXBotPollerSettings,
        *,
        redis_client: Optional[RedisClient] = None,
        ts_client: Optional[RedisTimeSeriesClient] = None,
        duckdb_writer: Optional[GEXDuckDBWriter] = None,
        dynamic_key: str = "gexbot:symbols:dynamic",
        supported_key: str = "gexbot:symbols:supported",
    ) -> None:
        self.settings = settings
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self.latest: Dict[str, Dict[str, Any]] = {}
        self.redis = redis_client
        self.ts_client = ts_client
        self.duckdb_writer = duckdb_writer
        self.dynamic_key = dynamic_key
        self.supported_key = supported_key
        self._base_symbols: Set[str] = {s.upper() for s in settings.symbols}
        self._dynamic_symbols: Dict[str, datetime] = self._load_dynamic_symbols()
        self._supported_symbols: Set[str] = set(self._base_symbols)
        self._last_supported_refresh: Optional[date] = None
        self.snapshot_count = 0
        self.last_snapshot_ts: Optional[str] = None
        self._last_cycle_started_ts: Optional[str] = None
        self._last_cycle_completed_ts: Optional[str] = None
        self._last_cycle_duration_seconds: Optional[float] = None
        self._last_cycle_interval_seconds: Optional[float] = None
        self._last_cycle_symbols: List[str] = []
        self._last_cycle_success_count = 0
        self._last_cycle_mode: Optional[str] = None
        self._last_cycle_fetch_duration_seconds: Optional[float] = None
        self._last_cycle_store_duration_seconds: Optional[float] = None
        self._last_cycle_redis_duration_seconds: Optional[float] = None
        self._last_cycle_ts_duration_seconds: Optional[float] = None
        self._last_cycle_fetch_max_seconds: Optional[float] = None
        self._last_cycle_fetch_avg_seconds: Optional[float] = None
        self._last_cycle_store_max_seconds: Optional[float] = None
        self._last_cycle_store_avg_seconds: Optional[float] = None
        self._last_cycle_symbol_timings: List[Dict[str, Any]] = []
        self._pending_ts_tasks: Set[asyncio.Task[None]] = set()
        # Track last written timestamp per symbol (ms) to ensure monotonic writes
        self._last_snapshot_ms_by_symbol: Dict[str, int] = {}
        self._snapshot_lock = threading.Lock()
        self._last_seen_snapshot_ms_by_symbol: Dict[str, int] = {}
        self._gex_history_by_symbol: Dict[str, Deque[Tuple[int, float]]] = {}
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
        await self.wait_for_pending_timeseries_writes()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _run(self) -> None:
        LOGGER.info(
            "Starting GEXBot poller (base_symbols=%s interval=%ss)",
            ",".join(sorted(self._base_symbols)),
            self.settings.interval_seconds,
        )
        timeout = aiohttp.ClientTimeout(total=12)
        connector = aiohttp.TCPConnector(
            limit=16,
            limit_per_host=8,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=True,
            family=socket.AF_INET,
        )
        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            if self._auto_refresh_symbols:
                await self._refresh_supported_symbols(session)
            pending_cycles: Set[asyncio.Task[None]] = set()
            next_rth_tick: Optional[float] = None
            while not self._stop_event.is_set():
                is_rth = self._is_rth_now()
                interval_seconds = self._current_interval_seconds()
                if is_rth and getattr(self.settings, "rth_overlap_enabled", False):
                    now = asyncio.get_event_loop().time()
                    if next_rth_tick is None:
                        next_rth_tick = now
                    if now >= next_rth_tick:
                        task = asyncio.create_task(
                            self._poll_cycle(session),
                            name="gexbot-rth-cycle",
                        )
                        pending_cycles.add(task)
                        task.add_done_callback(pending_cycles.discard)
                        next_rth_tick = now + 1.0
                    timeout = max(0.05, min(1.0, (next_rth_tick or now) - now))
                    try:
                        await asyncio.wait_for(self._stop_event.wait(), timeout=timeout)
                    except asyncio.TimeoutError:
                        continue
                    continue

                next_rth_tick = None
                loop_start = asyncio.get_event_loop().time()
                await self._poll_cycle(session)
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
            if pending_cycles:
                await asyncio.gather(*pending_cycles, return_exceptions=True)
        LOGGER.info("GEXBot poller stopped")

    async def _poll_cycle(self, session: aiohttp.ClientSession) -> None:
        loop_start = asyncio.get_event_loop().time()
        interval_seconds = self._current_interval_seconds()
        if self._auto_refresh_symbols and self._needs_supported_refresh():
            await self._refresh_supported_symbols(session)
        self._prune_expired_dynamic_symbols()
        await self._sync_dynamic_symbols(session)
        effective_symbols = self._effective_symbols()
        is_rth = self._is_rth_now()
        if "NQ_NDX" in self._base_symbols:
            fast_list = ["SPX", "NQ_NDX", "VIX"]
            symbols = [s for s in fast_list if s in set(effective_symbols)] if is_rth else effective_symbols
            mode = "rth_fast_path" if is_rth else "configured"
        else:
            symbols = effective_symbols
            mode = "configured"
        self._last_cycle_started_ts = datetime.now(timezone.utc).isoformat()
        self._last_cycle_symbols = list(symbols)
        self._last_cycle_interval_seconds = interval_seconds
        self._last_cycle_mode = mode
        LOGGER.debug(
            "poll-loop symbols=%s interval=%.3fs", symbols, interval_seconds
        )

        results = await asyncio.gather(
            *[self._fetch_and_store_symbol(session, sym) for sym in symbols]
        )
        self._last_cycle_success_count = sum(1 for item in results if item.get("ok"))
        fetch_durations = [
            float(item["fetch_seconds"])
            for item in results
            if item.get("fetch_seconds") is not None
        ]
        store_durations = [
            float(item["store_seconds"])
            for item in results
            if item.get("store_seconds") is not None
        ]
        self._last_cycle_fetch_max_seconds = max(fetch_durations) if fetch_durations else None
        self._last_cycle_fetch_avg_seconds = (
            sum(fetch_durations) / len(fetch_durations) if fetch_durations else None
        )
        self._last_cycle_store_max_seconds = max(store_durations) if store_durations else None
        self._last_cycle_store_avg_seconds = (
            sum(store_durations) / len(store_durations) if store_durations else None
        )
        self._last_cycle_symbol_timings = results
        self._last_cycle_completed_ts = datetime.now(timezone.utc).isoformat()
        self._last_cycle_duration_seconds = (
            asyncio.get_event_loop().time() - loop_start
        )

    async def _fetch_and_store_symbol(
        self, session: aiohttp.ClientSession, symbol: str
    ) -> Dict[str, Any]:
        symbol = symbol.upper()
        try:
            fetch_start = asyncio.get_event_loop().time()
            snapshot = await self._fetch_symbol(session, symbol)
            if snapshot:
                snapshot = await self._maybe_retry_same_timestamp(session, symbol, snapshot)
                store_start = asyncio.get_event_loop().time()
                self.latest[symbol] = snapshot
                timestamp_ms = _timestamp_ms(snapshot.get("timestamp"))
                self._last_seen_snapshot_ms_by_symbol[symbol] = timestamp_ms
                await self._record_timeseries(snapshot)
                fetch_duration = store_start - fetch_start
                store_duration = asyncio.get_event_loop().time() - store_start
                LOGGER.debug(
                    "fetched %s ts=%s in %.3fs (store=%.3fs)",
                    symbol,
                    snapshot.get("timestamp"),
                    fetch_duration,
                    store_duration,
                )
                return {
                    "symbol": symbol,
                    "ok": True,
                    "fetch_seconds": fetch_duration,
                    "store_seconds": store_duration,
                }
            LOGGER.debug("no snapshot returned for %s", symbol)
            return {
                "symbol": symbol,
                "ok": False,
                "fetch_seconds": None,
                "store_seconds": None,
            }
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.exception("Failed to poll GEXBot for %s", symbol)
            return {
                "symbol": symbol,
                "ok": False,
                "fetch_seconds": None,
                "store_seconds": None,
            }

    async def _maybe_retry_same_timestamp(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        retry_seconds = float(getattr(self.settings, "same_timestamp_retry_seconds", 0.25))
        if retry_seconds <= 0:
            return snapshot
        if not self._is_rth_now():
            return snapshot
        current_ts_ms = _timestamp_ms(snapshot.get("timestamp"))
        previous_ts_ms = self._last_seen_snapshot_ms_by_symbol.get(symbol)
        if previous_ts_ms is None or current_ts_ms != previous_ts_ms:
            return snapshot
        await asyncio.sleep(retry_seconds)
        retry_snapshot = await self._fetch_symbol(session, symbol)
        if retry_snapshot:
            return retry_snapshot
        return snapshot

    def _effective_symbols(self) -> List[str]:
        """Return the configured symbols, filtered by supported list if present.
        
        When base_symbols is empty and auto_refresh_symbols is enabled, poll all
        supported symbols instead of an empty list.
        """
        dynamic = set(self._dynamic_symbols.keys()) if self._auto_refresh_symbols else set()
        if self._supported_symbols:
            # If no base symbols configured but we have a supported list,
            # poll all supported symbols (this is the main poller's mode)
            if not self._base_symbols and self._auto_refresh_symbols:
                return sorted(self._supported_symbols | dynamic)
            filtered = self._base_symbols & self._supported_symbols
            if filtered:
                return sorted(filtered | dynamic)
        return sorted(self._base_symbols | dynamic)

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

    # Dynamic enrollment is available for auto-refresh pollers via the Redis-backed
    # enrollment set; fast pollers keep it disabled through auto_refresh_symbols=False.
    def add_symbol_for_day(self, symbol: str) -> None:
        """Enroll a symbol for polling for the next 24 hours."""

        if not self._auto_refresh_symbols:
            return
        normalized = symbol.upper().strip()
        if not normalized or normalized in self._base_symbols:
            return
        expires_at = datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(hours=24)
        existing = self._dynamic_symbols.get(normalized)
        if existing and existing > expires_at:
            return
        self._dynamic_symbols[normalized] = expires_at
        LOGGER.info("Enrolled %s for GEX polling (dynamic set)", normalized)
        self._persist_dynamic_symbols()

    async def fetch_symbol_now(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch a symbol immediately, enroll it, and return the normalized snapshot."""
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
            self.add_symbol_for_day(normalized)
            await self._record_timeseries(snapshot)
        return snapshot

    async def _fetch_symbol(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        base_url = f"https://api.gexbot.com/{symbol}/classic/{self.settings.aggregation_period}"
        headers = self._auth_headers()

        async def _fetch_json(url: str) -> Optional[Dict[str, Any]]:
            try:
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        LOGGER.debug("GEXBot %s returned %s for %s", symbol, resp.status, url)
                        return None
                    return await resp.json()
            except asyncio.CancelledError:
                raise
            except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                LOGGER.debug("GEXBot %s request failed for %s: %s", symbol, url, exc)
                return None

        zero = await _fetch_json(base_url)

        if not zero:
            LOGGER.debug("GEXBot %s returned no data", symbol)
            return None

        result = self._combine_payloads(symbol, zero)
        return result

    async def _record_timeseries(self, snapshot: Dict[str, Any]) -> None:
        if snapshot.get("gex_delta_15s") is None:
            snapshot["gex_delta_15s"] = self._compute_gex_delta_15s(snapshot)
            if snapshot.get("gex_delta_15s") is None:
                snapshot["gex_delta_15s"] = await asyncio.to_thread(
                    self._compute_gex_delta_15s_from_timeseries, snapshot
                )
        redis_start = asyncio.get_event_loop().time()
        self._store_snapshot_blob(snapshot)
        self._last_cycle_redis_duration_seconds = (
            asyncio.get_event_loop().time() - redis_start
        )
        ts_start = asyncio.get_event_loop().time()
        if self.ts_client:
            self._queue_timeseries_write(snapshot)
        self._last_cycle_ts_duration_seconds = (
            asyncio.get_event_loop().time() - ts_start
        )
        if self.duckdb_writer:
            self.duckdb_writer.enqueue_snapshot(snapshot)
        else:
            LOGGER.debug(
                "Skipping DuckDB enqueue for %s because no shared writer is attached",
                snapshot.get("symbol") or snapshot.get("ticker") or "UNKNOWN",
            )
        self.snapshot_count += 1
        self.last_snapshot_ts = (
            snapshot.get("timestamp") or datetime.utcnow().isoformat()
        )

    def _compute_gex_delta_15s(self, snapshot: Dict[str, Any]) -> Optional[float]:
        symbol = (snapshot.get("symbol") or snapshot.get("ticker") or "").upper()
        if not symbol:
            return None
        timestamp_ms = _timestamp_ms(snapshot.get("timestamp"))
        current_value = _to_float(snapshot.get("sum_gex_vol"))
        if current_value is None:
            return None

        history = self._gex_history_by_symbol.setdefault(symbol, deque(maxlen=240))
        history.append((timestamp_ms, current_value))
        cutoff = timestamp_ms - 15_000
        window = [point for point in history if point[0] >= cutoff]
        if len(window) < 2:
            return None

        diffs = [window[i][1] - window[i - 1][1] for i in range(1, len(window))]
        if not diffs:
            return None
        return sum(diffs) / len(diffs)

    def _compute_gex_delta_15s_from_timeseries(self, snapshot: Dict[str, Any]) -> Optional[float]:
        ts_client = self.ts_client
        if not ts_client:
            return None
        revrange = getattr(ts_client, "revrange", None)
        if not callable(revrange):
            return None

        symbol = (snapshot.get("symbol") or snapshot.get("ticker") or "").upper()
        if not symbol:
            return None
        timestamp_ms = _timestamp_ms(snapshot.get("timestamp"))
        key = f"ts:gex:sum_gex_vol:{symbol}"

        try:
            rows = revrange(key, "+", "-", count=100)
        except Exception:
            LOGGER.debug("Failed to read GEX timeseries for %s", symbol, exc_info=True)
            return None

        points: List[tuple[int, float]] = []
        for row in rows or []:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            try:
                points.append((int(row[0]), float(row[1])))
            except (TypeError, ValueError):
                continue

        if len(points) < 2:
            return None

        points.sort(key=lambda item: item[0])
        cutoff = timestamp_ms - 15_000
        window = [point for point in points if point[0] >= cutoff]
        if len(window) < 2:
            return None

        diffs = [window[i][1] - window[i - 1][1] for i in range(1, len(window))]
        if not diffs:
            return None
        return sum(diffs) / len(diffs)

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

        sum_gex_vol = _first(
            zero_payload.get("sum_gex_vol"),
            zero_payload.get("sum_gex"),
        )
        delta_risk_reversal = _first(
            zero_payload.get("delta_risk_reversal"),
            zero_payload.get("deltaRiskRev"),
            zero_payload.get("drr"),
            zero_payload.get("delta_rr"),
        )
        snapshot = {
            "symbol": symbol.upper(),
            "timestamp": timestamp,
            "spot": _to_float(_first(zero_payload.get("spot"))),
            "zero_gamma": _to_float(_first(zero_payload.get("zero_gamma"))),
            "sum_gex_vol": _to_float(sum_gex_vol),
            "sum_gex_oi": _to_float(zero_payload.get("sum_gex_oi")),
            "major_pos_vol": _to_float(_first(zero_payload.get("major_pos_vol"))),
            "major_neg_vol": _to_float(_first(zero_payload.get("major_neg_vol"))),
            "major_pos_oi": _to_float(_first(zero_payload.get("major_pos_oi"))),
            "major_neg_oi": _to_float(_first(zero_payload.get("major_neg_oi"))),
            "delta_risk_reversal": _to_float(delta_risk_reversal),
        }
        snapshot["major_pos"] = snapshot["major_pos_vol"]
        snapshot["major_neg"] = snapshot["major_neg_vol"]
        snapshot["ticker"] = snapshot["symbol"]
        snapshot["min_dte"] = zero_payload.get("min_dte")
        snapshot["sec_min_dte"] = zero_payload.get("sec_min_dte")
        snapshot["max_priors"] = zero_payload.get("max_priors")
        snapshot.update(build_compact_wall_fields({**snapshot, "strikes": zero_payload.get("strikes")}))
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
            "sum_gex_vol": snapshot.get("sum_gex_vol"),
            "sum_gex_oi": snapshot.get("sum_gex_oi"),
            "major_pos_vol": snapshot.get("major_pos_vol"),
            "major_neg_vol": snapshot.get("major_neg_vol"),
            "major_pos_oi": snapshot.get("major_pos_oi"),
            "major_neg_oi": snapshot.get("major_neg_oi"),
            "major_pos_vol_gamma": snapshot.get("major_pos_vol_gamma"),
            "major_neg_vol_gamma": snapshot.get("major_neg_vol_gamma"),
            "delta_risk_reversal": snapshot.get("delta_risk_reversal"),
            "pos_can1_strike": snapshot.get("pos_can1_strike"),
            "pos_can1_value": snapshot.get("pos_can1_value"),
            "pos_can1_pct": snapshot.get("pos_can1_pct"),
            "pos_can2_strike": snapshot.get("pos_can2_strike"),
            "pos_can2_value": snapshot.get("pos_can2_value"),
            "pos_can2_pct": snapshot.get("pos_can2_pct"),
            "neg_can1_strike": snapshot.get("neg_can1_strike"),
            "neg_can1_value": snapshot.get("neg_can1_value"),
            "neg_can1_pct": snapshot.get("neg_can1_pct"),
            "neg_can2_strike": snapshot.get("neg_can2_strike"),
            "neg_can2_value": snapshot.get("neg_can2_value"),
            "neg_can2_pct": snapshot.get("neg_can2_pct"),
        }
        # Hold the snapshot lock while assigning the timestamp AND writing to ts_client so
        # that insertion order into ts_client.samples always matches the timestamp order.
        # This prevents a race where the higher-timestamp thread writes before the lower one.
        with self._snapshot_lock:
            last_ms = self._last_snapshot_ms_by_symbol.get(symbol)
            if last_ms is not None:
                if timestamp_ms <= last_ms:
                    timestamp_ms = last_ms + 1
            self._last_snapshot_ms_by_symbol[symbol] = timestamp_ms
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
            if samples:
                try:
                    ts_client.multi_add(samples)
                except Exception:
                    LOGGER.warning(
                        "RedisTimeSeries write failed for %s; continuing to cache snapshot",
                        symbol,
                        exc_info=True,
                    )

    def _queue_timeseries_write(self, snapshot: Dict[str, Any]) -> None:
        async def _run() -> None:
            try:
                await asyncio.to_thread(self._write_snapshot_series, snapshot)
            except Exception:
                LOGGER.exception(
                    "GEX RedisTimeSeries write failed for %s",
                    snapshot.get("symbol") or snapshot.get("ticker") or "UNKNOWN",
                )

        task = asyncio.create_task(_run(), name="gexbot-ts-write")
        self._pending_ts_tasks.add(task)
        task.add_done_callback(self._pending_ts_tasks.discard)

    async def wait_for_pending_timeseries_writes(self) -> None:
        if not self._pending_ts_tasks:
            return
        pending = list(self._pending_ts_tasks)
        await asyncio.gather(*pending, return_exceptions=True)

    def _store_snapshot_blob(self, snapshot: Dict[str, Any]) -> None:
        symbol = snapshot.get("symbol") or snapshot.get("ticker")
        if not symbol:
            return
        key = f"{SNAPSHOT_KEY_PREFIX}{symbol.upper()}"
        
        sum_gex_vol = snapshot.get("sum_gex_vol")
        major_pos_vol = snapshot.get("major_pos_vol") or 0
        major_neg_vol = snapshot.get("major_neg_vol") or 0

        if sum_gex_vol is None and major_pos_vol == 0 and major_neg_vol == 0:
            if self.redis:
                try:
                    existing = self.redis.client.get(key)
                    if existing:
                        existing_data = json.loads(existing)
                        existing_sum_gex_vol = existing_data.get("sum_gex_vol")
                        existing_pos_vol = existing_data.get("major_pos_vol") or 0

                        if existing_sum_gex_vol is not None or existing_pos_vol > 0:
                            LOGGER.info(
                                "Preserving existing snapshot for %s (has volume data)",
                                symbol
                            )
                            return
                except Exception as e:
                    LOGGER.debug("Failed to check existing snapshot: %s", e)

        if self.redis:
            try:
                payload = json.dumps(_sanitize_floats(snapshot))
                self.redis.client.set(key, payload)
                try:
                    self.redis.client.publish(SNAPSHOT_PUBSUB_CHANNEL, payload)
                except Exception:
                    LOGGER.debug(
                        "Failed to publish snapshot for %s", symbol, exc_info=True
                    )
            except Exception:
                LOGGER.warning(
                    "Failed to cache GEX snapshot for %s", symbol, exc_info=True
                )

    def _load_dynamic_symbols(self) -> Dict[str, datetime]:
        if not self.redis:
            return {}
        try:
            cached = self.redis.get_cached(self.dynamic_key)
        except Exception:
            cached = None
        if cached is None:
            try:
                raw = self.redis.client.get(self.dynamic_key)
            except Exception:
                raw = None
            if raw:
                try:
                    cached = json.loads(raw)
                except Exception:
                    cached = None
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        result: Dict[str, datetime] = {}
        if isinstance(cached, list):
            for entry in cached:
                symbol: Optional[str] = None
                expires_at: Optional[datetime] = None
                if isinstance(entry, str):
                    symbol = entry.upper()
                    expires_at = now + timedelta(hours=24)
                elif isinstance(entry, dict):
                    entry_symbol = entry.get("symbol")
                    if entry_symbol:
                        symbol = str(entry_symbol).upper()
                    expires_raw = entry.get("expires_at") or entry.get("expiry")
                    expires_at = self._parse_expiry(expires_raw)
                if not symbol:
                    continue
                expiry = expires_at or (now + timedelta(hours=24))
                if expiry <= now:
                    continue
                result[symbol] = expiry
        return result

    def _persist_dynamic_symbols(self) -> None:
        if not self.redis:
            return
        if not self._dynamic_symbols:
            try:
                self.redis.delete_cached(self.dynamic_key)
            except Exception:
                pass
            return
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        payload = [
            {
                "symbol": symbol,
                "expires_at": expiry.astimezone(timezone.utc).isoformat(),
            }
            for symbol, expiry in sorted(self._dynamic_symbols.items())
        ]
        max_expiry = max(self._dynamic_symbols.values())
        ttl = max(60, int((max_expiry - now).total_seconds()))
        try:
            self.redis.set_cached(self.dynamic_key, payload, ttl_seconds=ttl)
        except Exception:
            LOGGER.debug("Failed to persist dynamic GEX symbols", exc_info=True)
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
        """Reconcile in-memory dynamic symbols with the Redis-backed enrollment set."""
        if not self._auto_refresh_symbols or not self.redis:
            return
        loaded = self._load_dynamic_symbols()
        if loaded == self._dynamic_symbols:
            return
        self._dynamic_symbols = loaded
        LOGGER.info("Loaded dynamic GEX symbols: %s", sorted(self._dynamic_symbols))

    def _prune_expired_dynamic_symbols(self) -> None:
        if not self._auto_refresh_symbols:
            return
        if not self._dynamic_symbols:
            return
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        expired = [
            symbol for symbol, expiry in self._dynamic_symbols.items() if expiry <= now
        ]
        if not expired:
            return
        for symbol in expired:
            self._dynamic_symbols.pop(symbol, None)
        self._persist_dynamic_symbols()

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
        self._dynamic_symbols = self._load_dynamic_symbols()
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
        headers = self._auth_headers()
        try:
            async with session.get(url, headers=headers) as resp:
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

    def _auth_headers(self) -> Dict[str, str]:
        if not self.settings.api_key:
            return {
                "User-Agent": "DataPipeline/2.0",
                "Accept": "application/json",
            }
        return {
            "Authorization": f"Bearer {self.settings.api_key}",
            "User-Agent": "DataPipeline/2.0",
            "Accept": "application/json",
        }

    def status(self) -> Dict[str, Any]:
        running = self._task is not None and not self._task.done()
        return {
            "running": running,
            "snapshot_count": self.snapshot_count,
            "last_snapshot_ts": self.last_snapshot_ts,
            "base_symbols": sorted(self._base_symbols),
            "dynamic_symbols": sorted(self._dynamic_symbols.keys()),
            "effective_interval_seconds": self._current_interval_seconds(),
            "last_cycle_started_ts": self._last_cycle_started_ts,
            "last_cycle_completed_ts": self._last_cycle_completed_ts,
            "last_cycle_duration_seconds": self._last_cycle_duration_seconds,
            "last_cycle_interval_seconds": self._last_cycle_interval_seconds,
            "last_cycle_mode": self._last_cycle_mode,
            "last_cycle_symbols": self._last_cycle_symbols,
            "last_cycle_success_count": self._last_cycle_success_count,
            "last_cycle_fetch_duration_seconds": self._last_cycle_fetch_duration_seconds,
            "last_cycle_store_duration_seconds": self._last_cycle_store_duration_seconds,
            "last_cycle_redis_duration_seconds": self._last_cycle_redis_duration_seconds,
            "last_cycle_ts_duration_seconds": self._last_cycle_ts_duration_seconds,
            "last_cycle_fetch_max_seconds": self._last_cycle_fetch_max_seconds,
            "last_cycle_fetch_avg_seconds": self._last_cycle_fetch_avg_seconds,
            "last_cycle_store_max_seconds": self._last_cycle_store_max_seconds,
            "last_cycle_store_avg_seconds": self._last_cycle_store_avg_seconds,
            "last_cycle_symbol_timings": self._last_cycle_symbol_timings,
            "pending_timeseries_writes": len(self._pending_ts_tasks),
            "is_rth_now": self._is_rth_now(),
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
