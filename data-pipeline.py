#!/usr/bin/env python3
"""Data pipeline orchestration entrypoint.

This module wires together the FastAPI surface area, background streamers,
Redis/DuckDB infrastructure, and auxiliary tools (Discord bot, Schwab trader,
etc.).  Historically we bolted legacy GEX HTTP handlers onto newer realtime
services, so the documentation here focuses on how everything fits together to
make operational debugging less painful.  When adding a new service, keep the
following mental model in mind:

1. "Service" objects are owned by :class:`ServiceManager` and should expose
   ``start``/``stop`` coroutines plus a ``status`` callable for observability.
2. FastAPI's ``lifespan`` hook starts the manager when the process boots and
   awaits shutdown so in-flight asyncio tasks can flush state to Redis.
3. Anything that touches Redis time-series should reuse ``RedisTimeSeriesClient``
   so history, lookup and monitoring endpoints continue to behave consistently.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config import settings  # noqa: E402
from src.import_gex_history import process_historical_imports  # noqa: E402
from src.lib.gex_history_queue import gex_history_queue  # noqa: E402
from src.lib.redis_client import RedisClient  # noqa: E402
from src.services.gexbot_poller import GEXBotPoller, GEXBotPollerSettings  # noqa: E402
from src.services.tastytrade_streamer import StreamerSettings, TastyTradeStreamer  # noqa: E402
from src.services.redis_timeseries import RedisTimeSeriesClient  # noqa: E402
from src.services.redis_flush_worker import FlushWorkerSettings, RedisFlushWorker  # noqa: E402
from src.services.discord_bot_service import DiscordBotService  # noqa: E402
from src.services.lookup_service import LookupService  # noqa: E402
from src.services.schwab_streamer import SchwabStreamClient, build_streamer  # noqa: E402

LOGGER = logging.getLogger("data_pipeline")
NOISY_STREAM_LOGGERS = [
    "tastytrade",
    "tastytrade.session",
    "tastytrade.utils",
    "httpx",
]
MARKET_DATA_METRICS_KEY = "metrics:market_data_counts"


class HistoryPayload(BaseModel):
    """Body schema for the manual history import endpoint."""

    url: str
    ticker: Optional[str] = None
    endpoint: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SchwabStreamingService:
    """Helper to run `SchwabStreamClient` inside a background thread."""

    def __init__(self, manager: "ServiceManager") -> None:
        self.manager = manager
        self.streamer: Optional[SchwabStreamClient] = None
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def is_running(self) -> bool:
        if self.streamer and getattr(self.streamer, "is_running", False):
            return True
        return bool(self.thread and self.thread.is_alive())

    def start(self) -> None:
        if not settings.schwab_enabled:
            LOGGER.warning("Schwab streaming disabled (set SCHWAB_ENABLED=true)")
            return
        if self.is_running:
            LOGGER.info("Schwab streamer already running")
            return
        if not self.manager.redis_client:
            self.manager._ensure_redis_clients()
        if not self.manager.redis_client:
            LOGGER.error("Redis client unavailable; cannot start Schwab streamer")
            return
        try:
            self.streamer = build_streamer(
                redis_client=self.manager.redis_client,
                tick_handler=self._on_tick,
                level2_handler=self._on_level2,
            )
        except RuntimeError as exc:
            LOGGER.error("Failed to build Schwab streamer: %s", exc)
            return
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._run_streamer, daemon=True, name="schwab-streamer")
        self.thread.start()
        LOGGER.info("Schwab streamer thread started")

    def stop(self) -> None:
        if self.streamer:
            self._stop_event.set()
            self.streamer.stop()
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
        self.streamer = None

    def _run_streamer(self) -> None:
        if not self.streamer:
            return
        try:
            self.streamer.start()
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Schwab streamer failed to start: %s", exc)
            return
        LOGGER.info("Schwab streamer running for symbols: %s", ",".join(self.streamer.symbols))
        while not self._stop_event.wait(timeout=1):
            if not self.streamer.is_running:
                LOGGER.warning("Schwab streamer stopped unexpectedly")
                break
        LOGGER.info("Schwab streamer loop exiting")

    def _on_tick(self, event) -> None:
        if not self.manager.loop:
            return
        payload = event.to_payload()
        asyncio.run_coroutine_threadsafe(
            self.manager._handle_trade_event(payload, source="schwab"),
            self.manager.loop,
        )

    def _on_level2(self, event) -> None:
        if not self.manager.loop:
            return
        payload = event.to_payload()
        asyncio.run_coroutine_threadsafe(
            self.manager._handle_depth_event(payload, source="schwab"),
            self.manager.loop,
        )


class ServiceManager:
    """Coordinate lifecycle and telemetry for the long-lived pipeline services."""

    def __init__(self) -> None:
        self.tastytrade: Optional[TastyTradeStreamer] = None
        self.gex_poller: Optional[GEXBotPoller] = None
        self.redis_client: Optional[RedisClient] = None
        self.rts: Optional[RedisTimeSeriesClient] = None
        self.flush_worker: Optional[RedisFlushWorker] = None
        self.discord_bot: Optional[DiscordBotService] = None
        self.lookup_service: Optional[LookupService] = None
        self.schwab_service = SchwabStreamingService(self)
        self.trade_count = 0
        self.depth_count = 0
        self.trade_counts: Dict[str, int] = {}
        self.depth_counts: Dict[str, int] = {}
        self.last_trade_ts: Optional[str] = None
        self.last_depth_ts: Optional[str] = None
        self.last_trade_timestamps: Dict[str, str] = {}
        self.last_depth_timestamps: Dict[str, str] = {}
        self.depth_snapshots: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.last_depth_comparison: Dict[str, Dict[str, Any]] = {}
        self.trade_counts_by_symbol: Dict[str, Dict[str, int]] = {}
        self.depth_counts_by_symbol: Dict[str, Dict[str, int]] = {}
        self._last_metrics_flush: float = 0.0
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _ensure_event_loop(self) -> None:
        if self._loop is not None:
            return
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            LOGGER.warning("No running asyncio loop detected; Schwab callbacks may be disabled")

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._loop

    def start(self) -> None:
        """Initialize shared clients and launch enabled background services."""
        self._ensure_event_loop()
        self._ensure_redis_clients()
        self._silence_streamer_logs()
        for service in ("tastytrade", "schwab", "gex_poller", "redis_flush", "discord_bot"):
            self.start_service(service)

    async def stop(self) -> None:
        """Stop all managed services in a best-effort fashion."""
        for service in ("tastytrade", "schwab", "gex_poller", "redis_flush", "discord_bot"):
            await self.stop_service(service)

    def status(self) -> Dict[str, Any]:
        """Expose a structured snapshot for the ``/status`` endpoint."""
        tasty_status = {
            "running": bool(self.tastytrade and self.tastytrade.is_running),
            "trade_samples": self.trade_counts.get("tastytrade", self.trade_count),
            "last_trade_ts": self.last_trade_timestamps.get("tastytrade", self.last_trade_ts),
            "depth_samples": self.depth_counts.get("tastytrade", self.depth_count),
            "last_depth_ts": self.last_depth_timestamps.get("tastytrade", self.last_depth_ts),
        }
        schwab_status = {
            "running": self.schwab_service.is_running,
            "trade_samples": self.trade_counts.get("schwab", 0),
            "depth_samples": self.depth_counts.get("schwab", 0),
            "last_trade_ts": self.last_trade_timestamps.get("schwab"),
            "last_depth_ts": self.last_depth_timestamps.get("schwab"),
            "symbols": settings.schwab_symbol_list,
        }
        return {
            "tastytrade_streamer": tasty_status,
            "schwab_streamer": schwab_status,
            "gex_poller": getattr(self.gex_poller, "status", lambda: {})(),
            "redis_flush_worker": getattr(self.flush_worker, "status", lambda: {})(),
            "discord_bot": getattr(
                self.discord_bot, "status", lambda: {"running": False, "enabled": settings.discord_bot_enabled}
            )(),
            "lookup_service": {
                "ready": bool(self.lookup_service),
                "recent_depth_diffs": list(self.last_depth_comparison.values())[:3],
            },
            "control_enabled": bool(settings.service_control_token),
            "market_data_metrics": self.metrics_snapshot(),
        }

    def _ensure_redis_clients(self) -> None:
        if not self.redis_client:
            self.redis_client = RedisClient(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
            )
        if not self.rts and self.redis_client:
            self.rts = RedisTimeSeriesClient(self.redis_client.client)
        if self.redis_client and self.rts and self.lookup_service is None:
            self.lookup_service = LookupService(self.redis_client, self.rts)

    def start_service(self, name: str) -> None:
        """Start a specific service by name if it is enabled via settings."""
        name = name.lower()
        self._ensure_event_loop()
        self._ensure_redis_clients()
        if name == "tastytrade" and settings.tastytrade_stream_enabled:
            if self.tastytrade and self.tastytrade.is_running:
                return
            self.tastytrade = TastyTradeStreamer(
                StreamerSettings(
                    client_id=settings.tastytrade_client_id or "",
                    client_secret=settings.tastytrade_client_secret or "",
                    refresh_token=settings.tastytrade_refresh_token or "",
                    symbols=settings.tastytrade_symbol_list,
                    depth_levels=settings.tastytrade_depth_levels,
                ),
                on_trade=self._handle_trade_event,
                on_depth=self._handle_depth_event,
            )
            self.tastytrade.start()
            LOGGER.info("TastyTrade streamer started")
        elif name == "schwab":
            self.schwab_service.start()
        elif name == "gex_poller" and settings.gex_polling_enabled and settings.gexbot_api_key:
            if self.gex_poller:
                return
            self.gex_poller = GEXBotPoller(
                GEXBotPollerSettings(
                    api_key=settings.gexbot_api_key,
                    symbols=settings.gex_symbol_list,
                    interval_seconds=settings.gex_poll_interval_seconds,
                    aggregation_period=settings.gex_poll_aggregation,
                    rth_interval_seconds=settings.gex_poll_rth_interval_seconds,
                    off_hours_interval_seconds=settings.gex_poll_off_hours_interval_seconds,
                    dynamic_schedule=settings.gex_poll_dynamic_schedule,
                ),
                redis_client=self.redis_client,
                ts_client=self.rts,
            )
            self.gex_poller.start()
            LOGGER.info("GEXBot poller started")
        elif name == "redis_flush":
            if self.flush_worker:
                return
            flush_settings = FlushWorkerSettings()
            self.flush_worker = RedisFlushWorker(self.redis_client, self.rts, flush_settings)
            self.flush_worker.start()
            LOGGER.info("Redis flush worker started")
        elif name == "discord_bot" and settings.discord_bot_enabled:
            if not self.discord_bot:
                script_path = PROJECT_ROOT / "discord-bot" / "run_discord_bot.py"
                self.discord_bot = DiscordBotService(script_path)
            self.discord_bot.start()
            LOGGER.info("Discord bot started")

    async def stop_service(self, name: str) -> None:
        """Stop a running service and clean up the local reference."""
        name = name.lower()
        if name == "tastytrade" and self.tastytrade:
            await self.tastytrade.stop()
            self.tastytrade = None
            LOGGER.info("TastyTrade streamer stopped")
        elif name == "schwab":
            self.schwab_service.stop()
            LOGGER.info("Schwab streamer stopped")
        elif name == "gex_poller" and self.gex_poller:
            await self.gex_poller.stop()
            self.gex_poller = None
            LOGGER.info("GEXBot poller stopped")
        elif name == "redis_flush" and self.flush_worker:
            await self.flush_worker.stop()
            self.flush_worker = None
            LOGGER.info("Redis flush worker stopped")
        elif name == "discord_bot" and self.discord_bot:
            await self.discord_bot.stop()
            LOGGER.info("Discord bot stopped")
            self.discord_bot = None

    async def restart_service(self, name: str) -> None:
        """Convenience helper for the ``/control`` endpoint."""
        await self.stop_service(name)
        self.start_service(name)

    def _silence_streamer_logs(self) -> None:
        for logger_name in NOISY_STREAM_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    async def _handle_trade_event(self, payload: Dict[str, Any], source: str = "tastytrade") -> None:
        """Persist trade ticks to RedisTimeSeries in a thread so I/O stays async friendly."""
        if not self.rts:
            return
        await asyncio.to_thread(self._write_trade_timeseries, payload, source)

    async def _handle_depth_event(self, payload: Dict[str, Any], source: str = "tastytrade") -> None:
        """Persist depth updates and compute cross-feed comparisons."""
        if not self.rts:
            return
        await asyncio.to_thread(self._write_depth_timeseries, payload, source)

    def _write_trade_timeseries(self, payload: Dict[str, Any], source: str) -> None:
        """Write trade price/size samples and keep per-source counters."""
        symbol = payload.get("symbol", "").upper() or "UNKNOWN"
        timestamp_ms = _timestamp_ms(payload.get("timestamp"))
        price = float(payload.get("price", 0.0))
        size = float(payload.get("size", 0.0))
        normalized_source = source.lower()
        samples = [
            (
                f"ts:trade:price:{symbol}:{normalized_source}",
                timestamp_ms,
                price,
                {"symbol": symbol, "type": "trade", "field": "price", "source": normalized_source},
            ),
            (
                f"ts:trade:size:{symbol}:{normalized_source}",
                timestamp_ms,
                size,
                {"symbol": symbol, "type": "trade", "field": "size", "source": normalized_source},
            ),
        ]
        if self.rts:
            self.rts.multi_add(samples)
        self.trade_count += 1
        self.trade_counts[normalized_source] = self.trade_counts.get(normalized_source, 0) + 1
        symbol_trade_counts = self.trade_counts_by_symbol.setdefault(symbol, {})
        symbol_trade_counts[normalized_source] = symbol_trade_counts.get(normalized_source, 0) + 1
        self.last_trade_ts = payload.get("timestamp") or datetime.utcnow().isoformat()
        self.last_trade_timestamps[normalized_source] = self.last_trade_ts
        self._maybe_persist_metrics()

    def _write_depth_timeseries(self, payload: Dict[str, Any], source: str) -> None:
        """Write depth ladders per level and update in-memory diffs."""
        symbol = payload.get("symbol", "").upper() or "UNKNOWN"
        timestamp_ms = _timestamp_ms(payload.get("timestamp"))
        bids = payload.get("bids") or []
        asks = payload.get("asks") or []
        normalized_source = source.lower()
        samples = []
        depth_levels = settings.tastytrade_depth_levels
        for idx, level in enumerate(bids[:depth_levels], start=1):
            price = float(level.get("price", 0.0))
            size = float(level.get("size", 0.0))
            prefix = f"ts:depth:{symbol}:{normalized_source}:bid:{idx}"
            samples.extend(
                [
                    (
                        f"{prefix}:price",
                        timestamp_ms,
                        price,
                        {
                            "symbol": symbol,
                            "type": "depth",
                            "side": "bid",
                            "level": str(idx),
                            "field": "price",
                            "source": normalized_source,
                        },
                    ),
                    (
                        f"{prefix}:size",
                        timestamp_ms,
                        size,
                        {
                            "symbol": symbol,
                            "type": "depth",
                            "side": "bid",
                            "level": str(idx),
                            "field": "size",
                            "source": normalized_source,
                        },
                    ),
                ]
            )
        for idx, level in enumerate(asks[:depth_levels], start=1):
            price = float(level.get("price", 0.0))
            size = float(level.get("size", 0.0))
            prefix = f"ts:depth:{symbol}:{normalized_source}:ask:{idx}"
            samples.extend(
                [
                    (
                        f"{prefix}:price",
                        timestamp_ms,
                        price,
                        {
                            "symbol": symbol,
                            "type": "depth",
                            "side": "ask",
                            "level": str(idx),
                            "field": "price",
                            "source": normalized_source,
                        },
                    ),
                    (
                        f"{prefix}:size",
                        timestamp_ms,
                        size,
                        {
                            "symbol": symbol,
                            "type": "depth",
                            "side": "ask",
                            "level": str(idx),
                            "field": "size",
                            "source": normalized_source,
                        },
                    ),
                ]
            )
        if samples and self.rts:
            self.rts.multi_add(samples)
            self.depth_count += 1
            self.depth_counts[normalized_source] = self.depth_counts.get(normalized_source, 0) + 1
            symbol_depth_counts = self.depth_counts_by_symbol.setdefault(symbol, {})
            symbol_depth_counts[normalized_source] = symbol_depth_counts.get(normalized_source, 0) + 1
            self.last_depth_ts = payload.get("timestamp") or datetime.utcnow().isoformat()
            self.last_depth_timestamps[normalized_source] = self.last_depth_ts
        self._record_depth_snapshot(symbol, normalized_source, payload)
        self._maybe_persist_metrics()

    def _record_depth_snapshot(self, symbol: str, source: str, payload: Dict[str, Any]) -> None:
        """Store the latest book for each feed and push comparisons into LookupService."""
        normalized_symbol = symbol.upper()
        symbol_snapshots = self.depth_snapshots.setdefault(normalized_symbol, {})
        symbol_snapshots[source] = payload
        if "tastytrade" not in symbol_snapshots or "schwab" not in symbol_snapshots:
            return
        summary = self._build_depth_comparison(
            normalized_symbol, symbol_snapshots["tastytrade"], symbol_snapshots["schwab"]
        )
        self.last_depth_comparison[normalized_symbol] = summary
        if self.lookup_service:
            self.lookup_service.store_depth_comparison(normalized_symbol, summary)

    @staticmethod
    def _build_depth_comparison(symbol: str, tasty: Dict[str, Any], schwab: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize spread/bbo variance between the two live sources."""
        tasty_best_bid = tasty.get("bids", [])[:1]
        tasty_best_ask = tasty.get("asks", [])[:1]
        schwab_best_bid = schwab.get("bids", [])[:1]
        schwab_best_ask = schwab.get("asks", [])[:1]
        def _best(entry):
            if entry:
                return entry[0].get("price"), entry[0].get("size")
            return None, None
        tasty_bid_price, tasty_bid_size = _best(tasty_best_bid)
        tasty_ask_price, tasty_ask_size = _best(tasty_best_ask)
        schwab_bid_price, schwab_bid_size = _best(schwab_best_bid)
        schwab_ask_price, schwab_ask_size = _best(schwab_best_ask)
        bid_levels = min(len(tasty.get("bids", [])), len(schwab.get("bids", [])), settings.tastytrade_depth_levels)
        ask_levels = min(len(tasty.get("asks", [])), len(schwab.get("asks", [])), settings.tastytrade_depth_levels)
        return {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "bid": {
                "tasty_price": tasty_bid_price,
                "schwab_price": schwab_bid_price,
                "tasty_size": tasty_bid_size,
                "schwab_size": schwab_bid_size,
                "best_diff": None if tasty_bid_price is None or schwab_bid_price is None else tasty_bid_price - schwab_bid_price,
                "avg_diff": ServiceManager._avg_price_diff(tasty.get("bids", []), schwab.get("bids", []), bid_levels),
                "compared_levels": bid_levels,
            },
            "ask": {
                "tasty_price": tasty_ask_price,
                "schwab_price": schwab_ask_price,
                "tasty_size": tasty_ask_size,
                "schwab_size": schwab_ask_size,
                "best_diff": None if tasty_ask_price is None or schwab_ask_price is None else tasty_ask_price - schwab_ask_price,
                "avg_diff": ServiceManager._avg_price_diff(tasty.get("asks", []), schwab.get("asks", []), ask_levels),
                "compared_levels": ask_levels,
            },
        }

    def metrics_snapshot(self) -> Dict[str, Any]:
        """Return in-memory counts for trades and depth samples."""
        return {
            "total_trades": self.trade_count,
            "trades_by_source": dict(self.trade_counts),
            "trades_by_symbol": {k: dict(v) for k, v in self.trade_counts_by_symbol.items()},
            "total_level2_samples": self.depth_count,
            "level2_by_source": dict(self.depth_counts),
            "level2_by_symbol": {k: dict(v) for k, v in self.depth_counts_by_symbol.items()},
            "last_trade_timestamps": dict(self.last_trade_timestamps),
            "last_depth_timestamps": dict(self.last_depth_timestamps),
            "redis_key": MARKET_DATA_METRICS_KEY,
        }

    def _maybe_persist_metrics(self) -> None:
        """Serialize metrics into Redis at a throttled cadence for dashboards."""
        if not self.redis_client:
            return
        now = time.monotonic()
        if now - self._last_metrics_flush < 1.0:
            return
        self._last_metrics_flush = now
        snapshot = self.metrics_snapshot()
        snapshot["updated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            self.redis_client.client.set(MARKET_DATA_METRICS_KEY, json.dumps(snapshot))
        except Exception:
            LOGGER.debug("Unable to persist market metrics to Redis", exc_info=True)

    @staticmethod
    def _avg_price_diff(a: List[Dict[str, Any]], b: List[Dict[str, Any]], levels: int) -> Optional[float]:
        """Return the mean price delta for overlapping depth levels."""
        total = 0.0
        count = 0
        for idx in range(levels):
            if idx >= len(a) or idx >= len(b):
                break
            a_price = a[idx].get("price")
            b_price = b[idx].get("price")
            if a_price is None or b_price is None:
                continue
            total += float(a_price) - float(b_price)
            count += 1
        return total / count if count else None


service_manager = ServiceManager()

UW_OPTION_LATEST_KEY = "uw:option_trades_super_algo:latest"
UW_OPTION_HISTORY_KEY = "uw:option_trades_super_algo:history"
UW_MARKET_LATEST_KEY = "uw:market_agg_socket:latest"
UW_MARKET_HISTORY_KEY = "uw:market_agg_socket:history"
UW_OPTION_STREAM_CHANNEL = "uw:option_trades_super_algo:stream"
UW_MARKET_STREAM_CHANNEL = "uw:market_agg_socket:stream"
UW_HISTORY_LIMIT = 200
UW_CACHE_TTL_SECONDS = 900
SUPPORTED_WEBHOOK_TOPICS = {
    "option_trades_super_algo",
    "market_agg_socket",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure ServiceManager starts before handling requests and shuts down cleanly."""
    LOGGER.info("Starting services during FastAPI lifespan")
    service_manager.start()
    try:
        yield
    finally:
        LOGGER.info("Stopping services during FastAPI lifespan")
        await service_manager.stop()


app = FastAPI(title="Data Pipeline", lifespan=lifespan)


@app.get("/")
async def root() -> Dict[str, str]:
    """Basic readiness probe for load balancers."""
    return {"status": "ok"}


@app.get("/health")
async def health() -> Dict[str, str]:
    """Explicit health endpoint used by Kubernetes and dashboards."""
    return {"status": "healthy"}


@app.get("/status")
async def status() -> Dict[str, Any]:
    """Expose the aggregated ServiceManager telemetry."""
    return service_manager.status()


@app.get("/metrics/market_data")
async def market_data_metrics() -> Dict[str, Any]:
    """Expose trade + level2 counters for quick Schwab/TastyTrade comparisons."""
    return service_manager.metrics_snapshot()


CONTROL_ACTIONS = {"start", "stop", "restart"}
CONTROL_SERVICES = {"tastytrade", "schwab", "gex_poller", "redis_flush", "discord_bot"}


@app.post("/control/{service}/{action}")
async def control_service(service: str, action: str, request: Request):
    """Start/stop/restart managed services after verifying the shared token."""
    if not settings.service_control_token:
        raise HTTPException(status_code=403, detail="Service control disabled")
    token = request.headers.get("X-Service-Token") or request.query_params.get("token")
    if token != settings.service_control_token:
        raise HTTPException(status_code=403, detail="Invalid control token")
    service = service.lower()
    if service not in CONTROL_SERVICES:
        raise HTTPException(status_code=400, detail="Unknown service")
    action = action.lower()
    if action not in CONTROL_ACTIONS:
        raise HTTPException(status_code=400, detail="Unsupported action")
    if action == "start":
        service_manager.start_service(service)
    elif action == "stop":
        await service_manager.stop_service(service)
    elif action == "restart":
        await service_manager.restart_service(service)
    return {"status": "ok", "service": service, "action": action, "state": service_manager.status()}


STATUS_PAGE = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <title>Data Pipeline Status</title>
  <style>
    body { font-family: Arial, sans-serif; background: #0f1115; color: #f1f1f1; margin: 0; padding: 2rem; }
    pre { background: #1e232b; padding: 1rem; border-radius: 8px; min-height: 200px; }
    .warning { color: #ffcc00; }
    .controls { margin: 1rem 0; display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }
    button { background: #2563eb; color: #fff; border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; }
    button:hover { background: #1d4ed8; }
    input { padding: 0.5rem; border-radius: 4px; border: 1px solid #333; background: #111; color: #fff; }
  </style>
</head>
<body>
  <h1>Data Pipeline Status</h1>
  <p class=\"warning\">Dashboard auto-refreshes every 3 seconds.</p>
  <div class=\"controls\">
    <input type=\"password\" id=\"control-token\" placeholder=\"Control token\" />
    <button onclick=\"controlService('discord_bot','restart')\">Restart Discord Bot</button>
    <button onclick=\"controlService('tastytrade','restart')\">Restart TastyTrade</button>
    <button onclick=\"controlService('gex_poller','restart')\">Restart GEX Poller</button>
    <button onclick="controlService('schwab','restart')">Restart Schwab Streamer</button>
    <button onclick=\"controlService('redis_flush','restart')\">Restart Flush Worker</button>
  </div>
  <pre id=\"status\">Loading...</pre>
  <h2>Market Data Metrics</h2>
  <pre id=\"metrics\">Loading metrics...</pre>
  <script>
    async function refresh() {
      try {
        const res = await fetch('/status');
        const data = await res.json();
        document.getElementById('status').textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        document.getElementById('status').textContent = 'Error: ' + err;
      }
      try {
        const res = await fetch('/metrics/market_data');
        const data = await res.json();
        document.getElementById('metrics').textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        document.getElementById('metrics').textContent = 'Error: ' + err;
      }
    }
    async function controlService(service, action) {
      const token = document.getElementById('control-token').value.trim();
      if (!token) {
        alert('Enter control token first');
        return;
      }
      try {
        const res = await fetch(`/control/${service}/${action}`, {
          method: 'POST',
          headers: { 'X-Service-Token': token }
        });
        const data = await res.json();
        if (!res.ok) {
          alert(data.detail || 'Request failed');
        } else {
          alert(`Action ${action} queued for ${service}`);
          refresh();
        }
      } catch (err) {
        alert('Control error: ' + err);
      }
    }
    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>
"""


@app.get("/status.html", response_class=HTMLResponse)
async def status_page() -> str:
    """Serve a lightweight HTML dashboard for ops users."""
    return STATUS_PAGE


@app.post("/gex_history_url")
async def gex_history_endpoint(request: Request, background_tasks: BackgroundTasks):
    """Permissive endpoint for historical GEX imports.

    Contract: accept any three-field JSON body where ``url`` points to
    ``https://hist.gex.bot/...``, another field carries a string matching
    ``gex_*`` (case-insensitive), and the remaining string is treated as
    the ticker. This keeps compatibility with bespoke clients that don't
    use consistent key names.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Invalid payload format")

    url = (body.get('url') or '').strip()
    if not url:
        raise HTTPException(status_code=422, detail="Missing url field")
    if not url.startswith("https://hist.gex.bot/"):
        raise HTTPException(status_code=422, detail="URL must start with https://hist.gex.bot/")

    # Surface payloads in logs (with large values sanitized) to mirror caller schema.
    try:
        log_snapshot = {k: (v if isinstance(v, (str, int, float, bool)) else type(v).__name__) for k, v in body.items()}
        LOGGER.debug("/gex_history_url payload: %s", log_snapshot)
    except Exception:
        LOGGER.debug("/gex_history_url payload logging failed")

    endpoint = None
    for value in body.values():
        if isinstance(value, str):
            trimmed = value.strip()
            lowered = trimmed.lower()
            if lowered.startswith("gex_") or lowered.startswith("gex"):
                endpoint = trimmed.lower()
                break
    if endpoint is None:
        inferred = _infer_endpoint(url)
        endpoint = inferred or 'gex_zero'

    ticker = None
    preferred_keys = ("ticker", "symbol", "underlying")
    for key in preferred_keys:
        value = body.get(key)
        if isinstance(value, str):
            candidate = value.strip()
            if candidate and not candidate.lower().startswith("gex_"):
                ticker = candidate.upper()
                break
    if not ticker:
        for key, value in body.items():
            if key == 'url':
                continue
            if isinstance(value, str):
                candidate = value.strip()
                if candidate and not candidate.lower().startswith("gex_"):
                    ticker = candidate.upper()
                    break

    # Always prefer ticker inferred from the URL so legacy naming patterns (e.g., NQ_NDX)
    # propagate consistently even if the client payload uses a shorthand ticker.
    if url:
        inferred = _extract_ticker_from_url(url)
        if inferred:
            ticker = inferred.upper()

    if not ticker:
        inferred = _extract_ticker_from_url(url)
        ticker = (inferred or "UNKNOWN").upper()

    # Accept metadata under several possible keys
    metadata = None
    for k in ('metadata', 'payload', 'data'):
        v = body.get(k)
        if isinstance(v, dict):
            metadata = v
            break
    if metadata is None:
        metadata = {}

    # Normalize endpoint default
    endpoint = endpoint or _infer_endpoint(url) or 'gex_zero'

    try:
        queue_id = gex_history_queue.enqueue_request(
            url=url,
            ticker=ticker,
            endpoint=endpoint,
            payload=metadata or {},
        )
    except Exception as exc:
        LOGGER.exception("Failed to enqueue history request")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    background_tasks.add_task(_trigger_queue_processing)

    return {
        "status": "queued",
        "id": queue_id,
        "url": url,
        "ticker": ticker,
        "endpoint": endpoint,
    }


@app.post("/uw")
async def universal_webhook(request: Request):
    """Accept payloads from monkeyscript/Unusual Whales style webhooks."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")

    topic = _extract_webhook_topic(body)
    if not topic:
        raise HTTPException(status_code=422, detail="Missing topic/object field")

    normalized_topic = topic.lower()
    if normalized_topic not in SUPPORTED_WEBHOOK_TOPICS:
        LOGGER.info("/uw received unsupported topic: %s", topic)
        return {"status": "ignored", "topic": topic}

    redis_client = _get_redis_client()

    stamped_payload = {
        "topic": normalized_topic,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "data": body,
    }

    if normalized_topic == "option_trades_super_algo":
        _cache_option_trade(redis_client, stamped_payload)
    elif normalized_topic == "market_agg_socket":
        _cache_market_agg(redis_client, stamped_payload)

    return {"status": "received", "topic": normalized_topic}


@app.get("/lookup/trades")
async def lookup_trades(symbol: str, source: str = "tastytrade", limit: int = 100) -> Dict[str, Any]:
    """Return recent trades for a given symbol/source pair."""
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    lookup = service_manager.lookup_service
    if not lookup:
        raise HTTPException(status_code=503, detail="Lookup service unavailable")
    normalized_source = source.lower()
    if normalized_source not in {"tastytrade", "schwab"}:
        raise HTTPException(status_code=400, detail="Unsupported source")
    capped_limit = max(1, min(limit, 500))
    history = lookup.trade_history(symbol.upper(), normalized_source, limit=capped_limit)
    return {"symbol": symbol.upper(), "source": normalized_source, "count": len(history), "history": history}


@app.get("/lookup/history")
async def lookup_history(
    symbol: str,
    limit: int = 100,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> Dict[str, Any]:
    """Return serialized import records for the requested symbol."""
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    lookup = service_manager.lookup_service
    if not lookup:
        raise HTTPException(status_code=503, detail="Lookup service unavailable")
    capped_limit = max(1, min(limit, 1000))
    history = lookup.lookup_history(symbol.upper(), limit=capped_limit, start_time=start_time, end_time=end_time)
    return {
        "symbol": symbol.upper(),
        "count": len(history),
        "history": history,
    }


@app.get("/lookup/depth_diff")
async def lookup_depth_diff(symbol: str) -> Dict[str, Any]:
    """Fetch the latest cross-feed depth comparison snapshot."""
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    lookup = service_manager.lookup_service
    if not lookup:
        raise HTTPException(status_code=503, detail="Lookup service unavailable")
    summary = lookup.get_depth_comparison(symbol.upper())
    if not summary:
        raise HTTPException(status_code=404, detail="Depth comparison not available for symbol")
    return summary


async def _trigger_queue_processing() -> None:
    """Kick processing of any queued historical imports on a worker thread."""
    try:
        await asyncio.to_thread(process_historical_imports)
    except Exception:
        LOGGER.exception("Background import processing failed")


def _normalize_string(value: Optional[Any]) -> str:
    """Trim and stringify optionally missing payload values."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_webhook_topic(payload: Dict[str, Any]) -> Optional[str]:
    """Pull the topic/object indicator out of a webhook payload."""
    candidate_fields = ("topic", "object", "type", "feed", "kind")
    for field in candidate_fields:
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    nested = payload.get('payload') or payload.get('data')
    if isinstance(nested, dict):
        for field in candidate_fields:
            value = nested.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _get_redis_client():
    """Ensure the Redis client exists and return the underlying connection."""
    try:
        service_manager._ensure_redis_clients()
    except Exception:
        LOGGER.exception("Failed to initialize Redis client")
    wrapper = service_manager.redis_client
    if not wrapper:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    return wrapper.client


def _cache_option_trade(redis_conn, payload: Dict[str, Any]) -> None:
    serialized = json.dumps(payload, default=str)
    pipe = redis_conn.pipeline()
    pipe.setex(UW_OPTION_LATEST_KEY, UW_CACHE_TTL_SECONDS, serialized)
    pipe.lpush(UW_OPTION_HISTORY_KEY, serialized)
    pipe.ltrim(UW_OPTION_HISTORY_KEY, 0, UW_HISTORY_LIMIT - 1)
    pipe.execute()
    try:
        redis_conn.publish(UW_OPTION_STREAM_CHANNEL, serialized)
    except Exception:
        LOGGER.exception("Failed to publish option trade alert")


def _cache_market_agg(redis_conn, payload: Dict[str, Any]) -> None:
    serialized = json.dumps(payload, default=str)
    pipe = redis_conn.pipeline()
    pipe.setex(UW_MARKET_LATEST_KEY, UW_CACHE_TTL_SECONDS, serialized)
    pipe.lpush(UW_MARKET_HISTORY_KEY, serialized)
    pipe.ltrim(UW_MARKET_HISTORY_KEY, 0, UW_HISTORY_LIMIT - 1)
    pipe.execute()
    try:
        redis_conn.publish(UW_MARKET_STREAM_CHANNEL, serialized)
    except Exception:
        LOGGER.exception("Failed to publish market aggregate update")


def _infer_endpoint(url: str) -> str:
    """Guess the GEX endpoint slug from a legacy URL."""
    import re

    match = re.search(r"_((?:gex_zero|gex_one|gex_full))\\.json", url)
    if match:
        return match.group(1)
    return "gex_zero"


def _extract_ticker_from_url(url: str) -> str:
    """Parse a ticker symbol from classic history URLs."""
    import re
    # Match DATE_TICKER_classic where ticker may contain underscores/digits
    match = re.search(r"/(\d{4}-\d{2}-\d{2})_([A-Z0-9_]+)_classic", url)
    if match:
        return match.group(2)
    # Fallback: capture token before '_classic'
    match2 = re.search(r"/([A-Z0-9_]{1,12})_classic", url)
    if match2:
        return match2.group(1)
    return ""


def _timestamp_ms(value: Optional[str]) -> int:
    """Normalize timestamps from strings or epoch numbers into ms precision."""
    if isinstance(value, (int, float)):
        return int(float(value))
    if isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except ValueError:
            pass
    return int(datetime.utcnow().timestamp() * 1000)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for running the uvicorn server."""
    parser = argparse.ArgumentParser(description="Run data pipeline server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    """Entry point used both by ``python data-pipeline.py`` and packaging."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
