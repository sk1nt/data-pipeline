#!/usr/bin/env python3
# ruff: noqa: E402
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
from logging.handlers import RotatingFileHandler
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket
from fastapi.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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
from src.services.gexbot_poller import (
    GEXBotPoller,
    GEXBotPollerSettings,
    SNAPSHOT_KEY_PREFIX,
    SNAPSHOT_PUBSUB_CHANNEL,
)  # noqa: E402
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
TASTYTRADE_TRADE_CHANNEL = "market_data:tastytrade:trades"


class HistoryPayload(BaseModel):
    """Body schema for the manual history import endpoint."""

    url: str
    ticker: Optional[str] = None
    endpoint: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MLTradePayload(BaseModel):
    """Schema for machine learning driven trade signals."""

    symbol: str
    action: str
    direction: str
    price: float
    confidence: float
    position_before: int
    position_after: int
    pnl: float
    total_pnl: float
    total_trades: int
    timestamp: datetime
    simulated: bool = True


MLTradePayload.model_rebuild()


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
        if settings.schwab_stream_paused:
            LOGGER.warning(
                "Schwab streaming paused (set SCHWAB_STREAM_PAUSED=false to resume)"
            )
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
        self.thread = threading.Thread(
            target=self._run_streamer, daemon=True, name="schwab-streamer"
        )
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
        LOGGER.info(
            "Schwab streamer running for symbols: %s", ",".join(self.streamer.symbols)
        )
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
        self.gex_nq_poller: Optional[GEXBotPoller] = None
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
            LOGGER.warning(
                "No running asyncio loop detected; Schwab callbacks may be disabled"
            )

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._loop

    def start(self) -> None:
        """Initialize shared clients and launch enabled background services."""
        self._ensure_event_loop()
        self._ensure_redis_clients()
        self._silence_streamer_logs()
        for service in (
            "tastytrade",
            "schwab",
            "gex_poller",
            "gex_nq_poller",
            "redis_flush",
            "discord_bot",
        ):
            self.start_service(service)

    async def stop(self) -> None:
        """Stop all managed services in a best-effort fashion."""
        for service in (
            "tastytrade",
            "schwab",
            "gex_poller",
            "gex_nq_poller",
            "redis_flush",
            "discord_bot",
        ):
            await self.stop_service(service)

    def status(self) -> Dict[str, Any]:
        """Expose a structured snapshot for the ``/status`` endpoint."""
        tasty_status = {
            "running": bool(self.tastytrade and self.tastytrade.is_running),
            "trade_samples": self.trade_counts.get("tastytrade", self.trade_count),
            "last_trade_ts": self.last_trade_timestamps.get(
                "tastytrade", self.last_trade_ts
            ),
            "depth_samples": self.depth_counts.get("tastytrade", self.depth_count),
            "last_depth_ts": self.last_depth_timestamps.get(
                "tastytrade", self.last_depth_ts
            ),
        }
        schwab_status = {
            "running": self.schwab_service.is_running,
            "enabled": settings.schwab_enabled,
            "paused": settings.schwab_stream_paused,
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
            "gex_nq_poller": getattr(self.gex_nq_poller, "status", lambda: {})(),
            "redis_flush_worker": getattr(self.flush_worker, "status", lambda: {})(),
            "discord_bot": getattr(
                self.discord_bot,
                "status",
                lambda: {"running": False, "enabled": settings.discord_bot_enabled},
            )(),
            "lookup_service": {
                "ready": bool(self.lookup_service),
                "recent_depth_diffs": list(self.last_depth_comparison.values())[:3],
            },
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
                    depth_levels=settings.tastytrade_depth_cap,
                    enable_depth=getattr(settings, "tastytrade_enable_depth", False),
                ),
                on_trade=self._handle_trade_event,
                on_depth=self._handle_depth_event,
            )
            self.tastytrade.start()
            LOGGER.info("TastyTrade streamer started")
        elif name == "schwab":
            self.schwab_service.start()
        elif (
            name == "gex_poller"
            and settings.gex_polling_enabled
            and settings.gexbot_api_key
        ):
            if self.gex_poller:
                return
            self.gex_poller = GEXBotPoller(
                GEXBotPollerSettings(
                    api_key=settings.gexbot_api_key,
                    symbols=[],
                    interval_seconds=settings.gex_poll_interval_seconds,
                    aggregation_period=settings.gex_poll_aggregation,
                    # Main poller should poll at 5s during RTH regardless of .env
                    rth_interval_seconds=5.0,
                    off_hours_interval_seconds=settings.gex_poll_off_hours_interval_seconds,
                    dynamic_schedule=settings.gex_poll_dynamic_schedule,
                    # Exclude the NQ poller symbols from the main poller
                    exclude_symbols=settings.gex_nq_poll_symbol_list,
                ),
                redis_client=self.redis_client,
                ts_client=self.rts,
            )
            self.gex_poller.start()
            LOGGER.info("GEXBot poller started")
        elif (
            name == "gex_nq_poller"
            and settings.gex_nq_polling_enabled
            and settings.gexbot_api_key
        ):
            if self.gex_nq_poller:
                return
            symbols = settings.gex_nq_poll_symbol_list
            if not symbols:
                LOGGER.warning("No symbols configured for NQ poller; skipping start")
                return
            self.gex_nq_poller = GEXBotPoller(
                GEXBotPollerSettings(
                    api_key=settings.gexbot_api_key,
                    symbols=symbols,
                    interval_seconds=settings.gex_nq_poll_interval_seconds,
                    aggregation_period=settings.gex_nq_poll_aggregation,
                    rth_interval_seconds=settings.gex_nq_poll_rth_interval_seconds,
                    off_hours_interval_seconds=settings.gex_nq_poll_off_hours_interval_seconds,
                    dynamic_schedule=settings.gex_nq_poll_dynamic_schedule,
                    sierra_chart_output_path=settings.sierra_chart_output_path,
                    auto_refresh_symbols=False,
                ),
                redis_client=self.redis_client,
                ts_client=self.rts,
            )
            self.gex_nq_poller.start()
            LOGGER.info("GEXBot NQ poller started")
        elif name == "redis_flush":
            if self.flush_worker:
                return
            flush_settings = FlushWorkerSettings()
            self.flush_worker = RedisFlushWorker(
                self.redis_client, self.rts, flush_settings
            )
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
        elif name == "gex_nq_poller" and self.gex_nq_poller:
            await self.gex_nq_poller.stop()
            self.gex_nq_poller = None
            LOGGER.info("GEXBot NQ poller stopped")
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

    async def _handle_trade_event(
        self, payload: Dict[str, Any], source: str = "tastytrade"
    ) -> None:
        """Persist trade ticks to RedisTimeSeries in a thread so I/O stays async friendly."""
        if not self.rts:
            return
        await asyncio.to_thread(self._write_trade_timeseries, payload, source)

    async def _handle_depth_event(
        self, payload: Dict[str, Any], source: str = "tastytrade"
    ) -> None:
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
                {
                    "symbol": symbol,
                    "type": "trade",
                    "field": "price",
                    "source": normalized_source,
                },
            ),
            (
                f"ts:trade:size:{symbol}:{normalized_source}",
                timestamp_ms,
                size,
                {
                    "symbol": symbol,
                    "type": "trade",
                    "field": "size",
                    "source": normalized_source,
                },
            ),
        ]
        if self.rts:
            self.rts.multi_add(samples)
        if normalized_source == "tastytrade":
            self._publish_tastytrade_trade(symbol, price, size, timestamp_ms, payload)
        self.trade_count += 1
        self.trade_counts[normalized_source] = (
            self.trade_counts.get(normalized_source, 0) + 1
        )
        symbol_trade_counts = self.trade_counts_by_symbol.setdefault(symbol, {})
        symbol_trade_counts[normalized_source] = (
            symbol_trade_counts.get(normalized_source, 0) + 1
        )
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
        depth_levels = settings.tastytrade_depth_cap
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
            self.depth_counts[normalized_source] = (
                self.depth_counts.get(normalized_source, 0) + 1
            )
            symbol_depth_counts = self.depth_counts_by_symbol.setdefault(symbol, {})
            symbol_depth_counts[normalized_source] = (
                symbol_depth_counts.get(normalized_source, 0) + 1
            )
            self.last_depth_ts = (
                payload.get("timestamp") or datetime.utcnow().isoformat()
            )
            self.last_depth_timestamps[normalized_source] = self.last_depth_ts
        self._record_depth_snapshot(symbol, normalized_source, payload)
        self._maybe_persist_metrics()

    def _publish_tastytrade_trade(
        self,
        symbol: str,
        price: float,
        size: float,
        timestamp_ms: int,
        payload: Dict[str, Any],
    ) -> None:
        if not self.redis_client:
            return
        message = {
            "symbol": symbol,
            "price": price,
            "size": size,
            "timestamp": payload.get("timestamp") or datetime.utcnow().isoformat(),
            "ts_ms": timestamp_ms,
            "source": "tastytrade",
        }
        extra_fields = (
            "action",
            "direction",
            "confidence",
            "position_before",
            "position_after",
        )
        for key in extra_fields:
            if key in payload:
                message[key] = payload[key]
        try:
            self.redis_client.client.publish(
                TASTYTRADE_TRADE_CHANNEL, json.dumps(message, default=str)
            )
        except Exception:
            LOGGER.debug("Failed to publish tastytrade trade", exc_info=True)

    def _record_depth_snapshot(
        self, symbol: str, source: str, payload: Dict[str, Any]
    ) -> None:
        """Store the latest book for each feed and push comparisons into LookupService."""
        normalized_symbol = symbol.upper()
        symbol_snapshots = self.depth_snapshots.setdefault(normalized_symbol, {})
        symbol_snapshots[source] = payload
        if "tastytrade" not in symbol_snapshots or "schwab" not in symbol_snapshots:
            return
        summary = self._build_depth_comparison(
            normalized_symbol,
            symbol_snapshots["tastytrade"],
            symbol_snapshots["schwab"],
        )
        self.last_depth_comparison[normalized_symbol] = summary
        if self.lookup_service:
            self.lookup_service.store_depth_comparison(normalized_symbol, summary)

    @staticmethod
    def _build_depth_comparison(
        symbol: str, tasty: Dict[str, Any], schwab: Dict[str, Any]
    ) -> Dict[str, Any]:
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
        bid_levels = min(
            len(tasty.get("bids", [])),
            len(schwab.get("bids", [])),
            settings.tastytrade_depth_cap,
        )
        ask_levels = min(
            len(tasty.get("asks", [])),
            len(schwab.get("asks", [])),
            settings.tastytrade_depth_cap,
        )
        return {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "bid": {
                "tasty_price": tasty_bid_price,
                "schwab_price": schwab_bid_price,
                "tasty_size": tasty_bid_size,
                "schwab_size": schwab_bid_size,
                "best_diff": None
                if tasty_bid_price is None or schwab_bid_price is None
                else tasty_bid_price - schwab_bid_price,
                "avg_diff": ServiceManager._avg_price_diff(
                    tasty.get("bids", []), schwab.get("bids", []), bid_levels
                ),
                "compared_levels": bid_levels,
            },
            "ask": {
                "tasty_price": tasty_ask_price,
                "schwab_price": schwab_ask_price,
                "tasty_size": tasty_ask_size,
                "schwab_size": schwab_ask_size,
                "best_diff": None
                if tasty_ask_price is None or schwab_ask_price is None
                else tasty_ask_price - schwab_ask_price,
                "avg_diff": ServiceManager._avg_price_diff(
                    tasty.get("asks", []), schwab.get("asks", []), ask_levels
                ),
                "compared_levels": ask_levels,
            },
        }

    def metrics_snapshot(self) -> Dict[str, Any]:
        """Return in-memory counts for trades and depth samples."""
        return {
            "total_trades": self.trade_count,
            "trades_by_source": dict(self.trade_counts),
            "trades_by_symbol": {
                k: dict(v) for k, v in self.trade_counts_by_symbol.items()
            },
            "total_level2_samples": self.depth_count,
            "level2_by_source": dict(self.depth_counts),
            "level2_by_symbol": {
                k: dict(v) for k, v in self.depth_counts_by_symbol.items()
            },
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
    def _avg_price_diff(
        a: List[Dict[str, Any]], b: List[Dict[str, Any]], levels: int
    ) -> Optional[float]:
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

UW_OPTION_LATEST_KEY = "uw:options-trade:latest"
UW_OPTION_HISTORY_KEY = "uw:options-trade:history"
UW_MARKET_LATEST_KEY = "uw:market-state:latest"
UW_MARKET_HISTORY_KEY = "uw:market-state:history"
UW_OPTION_STREAM_CHANNEL = "uw:options-trade:stream"
UW_MARKET_STREAM_CHANNEL = "uw:market-state:stream"
UW_HISTORY_LIMIT = 200
UW_CACHE_TTL_SECONDS = 900
ML_TRADE_LATEST_KEY = "trade:ml-bot:latest"
ML_TRADE_HISTORY_KEY = "trade:ml-bot"
ML_TRADE_STREAM_CHANNEL = "trade:ml-bot:stream"
ML_TRADE_HISTORY_LIMIT = 500
TASTYTRADE_TRADE_CHANNEL = "market_data:tastytrade:trades"
SUPPORTED_WEBHOOK_TOPICS = {
    "options-trade",
    "market-state",
}
TOPIC_ALIASES = {
    # Alias used by Unusual Whales streaming payloads
    "option_trades_super_algo": "options-trade",
}
TOPIC_SYMBOL_SEPARATORS = (":", "|")


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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://unusualwhales.com"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, str]:
    """Basic readiness probe for load balancers."""
    return {"status": "ok"}


@app.get("/health")
async def health() -> Dict[str, str]:
    """Explicit health endpoint used by Kubernetes and dashboards."""
    return {"status": "healthy"}


@app.get("/status")
async def status(request: Request) -> Dict[str, Any]:
    """Expose the aggregated ServiceManager telemetry."""
    # Log status access via dedicated status logger
    logging.getLogger("data_pipeline.status").info(
        "Status endpoint requested from %s", getattr(request.client, "host", "unknown")
    )
    return service_manager.status()


@app.post("/control/{service_name}/start")
async def control_start(service_name: str) -> Dict[str, Any]:
    """Start one of the managed services by name (no auth)."""
    try:
        service_manager.start_service(service_name)
        return {"status": "started", "service": service_name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/control/{service_name}/stop")
async def control_stop(service_name: str) -> Dict[str, Any]:
    """Stop one of the managed services by name (no auth)."""
    try:
        await service_manager.stop_service(service_name)
        return {"status": "stopped", "service": service_name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/control/{service_name}/restart")
async def control_restart(service_name: str) -> Dict[str, Any]:
    """Restart a managed service (no auth)."""
    try:
        await service_manager.restart_service(service_name)
        return {"status": "restarted", "service": service_name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/control/{service_name}/status")
async def control_status(service_name: str) -> Dict[str, Any]:
    """Return a service-specific status snapshot (no auth)."""
    try:
        svc = getattr(service_manager, service_name, None)
        if svc and hasattr(svc, "status"):
            return getattr(svc, "status")()
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics/market_data")
async def market_data_metrics() -> Dict[str, Any]:
    """Expose trade + level2 counters for quick Schwab/TastyTrade comparisons."""
    return service_manager.metrics_snapshot()


@app.post("/ml-trade")
async def ingest_ml_trade(trade: MLTradePayload) -> Dict[str, Any]:
    """Accept ML-driven trade signals and persist them for downstream alerts."""
    redis_conn = _get_redis_client()
    payload = trade.model_dump()
    payload["symbol"] = payload.get("symbol", "").upper()
    payload["action"] = (payload.get("action") or "").lower()
    payload["direction"] = (payload.get("direction") or "").lower()
    trade_ts = (
        trade.timestamp
        if isinstance(trade.timestamp, datetime)
        else datetime.now(timezone.utc)
    )
    payload["timestamp"] = trade_ts.astimezone(timezone.utc).isoformat()
    payload["received_at"] = datetime.now(timezone.utc).isoformat()
    _cache_ml_trade(redis_conn, payload)
    return {
        "status": "received",
        "symbol": payload["symbol"],
        "simulated": payload["simulated"],
    }


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
  </style>
</head>
<body>
  <h1>Data Pipeline Status</h1>
    <p class=\"warning\">Dashboard auto-refreshes every 1 second.</p>
  <pre id=\"status\">Loading...</pre>
  <script>
    async function refresh() {
      try {
        const res = await fetch('/status');
        const data = await res.json();
        document.getElementById('status').textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        document.getElementById('status').textContent = 'Error: ' + err;
      }
    }
        refresh();
        setInterval(refresh, 1000);

        // service control helpers
        async function restartService(service) {
            try {
                const res = await fetch(`/control/${service}/restart`, { method: 'POST' });
                if (!res.ok) {
                    const text = await res.text();
                    alert(`Failed to restart ${service}: ${res.status} ${text}`);
                    return;
                }
                const data = await res.json();
                alert(`Restarted ${service}: ${JSON.stringify(data)}`);
            } catch (err) {
                alert(`Error restarting ${service}: ${err}`);
            }
        }

        function renderControls() {
            const services = ['tastytrade', 'schwab', 'gex_poller', 'gex_nq_poller', 'redis_flush', 'discord_bot'];
            const div = document.createElement('div');
            div.style.marginTop = '1rem';
            services.forEach(s => {
                const btn = document.createElement('button');
                btn.textContent = `Restart ${s}`;
                btn.style.marginRight = '0.5rem';
                btn.onclick = () => restartService(s);
                div.appendChild(btn);
            });
            document.body.insertBefore(div, document.getElementById('status'));
        }
        renderControls();
  </script>
</body>
</html>
"""


@app.get("/status.html", response_class=HTMLResponse)
async def status_page(request: Request) -> str:
    """Serve a lightweight HTML dashboard for ops users."""
    logging.getLogger("data_pipeline.status").info(
        "Status page requested from %s", getattr(request.client, "host", "unknown")
    )
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

    url = (body.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=422, detail="Missing url field")
    if not url.startswith("https://hist.gex.bot/"):
        raise HTTPException(
            status_code=422, detail="URL must start with https://hist.gex.bot/"
        )

    # Surface payloads in logs (with large values sanitized) to mirror caller schema.
    try:
        log_snapshot = {
            k: (v if isinstance(v, (str, int, float, bool)) else type(v).__name__)
            for k, v in body.items()
        }
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
        endpoint = inferred or "gex_zero"

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
            if key == "url":
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
    for k in ("metadata", "payload", "data"):
        v = body.get(k)
        if isinstance(v, dict):
            metadata = v
            break
    if metadata is None:
        metadata = {}

    # Normalize endpoint default
    endpoint = endpoint or _infer_endpoint(url) or "gex_zero"

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


@app.api_route(
    "/uw", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
)
async def universal_webhook(request: Request):
    """Accept UW webhook payloads including websocket array format."""
    raw_body = await request.body()

    raw_payload: Any
    try:
        raw_payload = json.loads(raw_body) if raw_body else {}
    except json.JSONDecodeError:
        raw_payload = {
            "__raw_text__": raw_body.decode("utf-8", errors="replace"),
        }

    if not raw_payload and request.method == "GET":
        raw_payload = dict(request.query_params.multi_items())

    # Check if this is a websocket array format message
    # Format: [null,null,"message_type",\"topic\",{\"data\":{...}}]
    if isinstance(raw_payload, list) and len(raw_payload) >= 5:
        try:
            from src.services.uw_message_service import UWMessageService
            # Get the RedisClient wrapper, not the raw connection
            redis_client_wrapper = service_manager.redis_client
            if not redis_client_wrapper:
                raise HTTPException(status_code=503, detail="Redis unavailable")
            uw_service = UWMessageService(redis_client_wrapper)
            result = uw_service.process_raw_message(raw_payload)
            if result:
                return result
            else:
                return {"status": "error", "reason": "failed_to_parse_uw_message"}
        except Exception as e:
            LOGGER.exception("Failed to process UW websocket message: %s", e)
            return {"status": "error", "reason": str(e)}

    # Legacy webhook format handling
    payload = _coerce_webhook_payload(raw_payload)

    topic = _extract_webhook_topic(payload)
    normalized_topic, topic_symbol = _normalize_topic(topic)

    if not normalized_topic:
        LOGGER.info("/uw temporarily accepting payload without topic: %s", payload)
        return {
            "status": "received",
            "topic": "unknown",
            "note": "topic missing; captured for review",
        }

    redis_client = _get_redis_client()

    stamped_payload: Dict[str, Any] = {
        "topic": normalized_topic,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "data": payload,
        "method": request.method,
    }
    if topic_symbol and isinstance(payload, dict):
        payload.setdefault("symbol", topic_symbol)
        payload.setdefault("ticker", topic_symbol)
        nested = payload.get("data")
        if isinstance(nested, dict):
            nested.setdefault("symbol", topic_symbol)
            nested.setdefault("ticker", topic_symbol)
        stamped_payload["symbol"] = topic_symbol
        stamped_payload["ticker"] = topic_symbol

    if normalized_topic in SUPPORTED_WEBHOOK_TOPICS:
        if normalized_topic == "options-trade":
            _cache_option_trade(redis_client, stamped_payload)
        elif normalized_topic == "market-state":
            _cache_market_state(redis_client, stamped_payload)
    else:
        LOGGER.info(
            "/uw temporarily accepting unsupported topic %s via %s",
            normalized_topic,
            request.method,
        )

    return {"status": "received", "topic": normalized_topic}


@app.get("/lookup/trades")
async def lookup_trades(
    symbol: str, source: str = "tastytrade", limit: int = 100
) -> Dict[str, Any]:
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
    history = lookup.trade_history(
        symbol.upper(), normalized_source, limit=capped_limit
    )
    return {
        "symbol": symbol.upper(),
        "source": normalized_source,
        "count": len(history),
        "history": history,
    }


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
    history = lookup.lookup_history(
        symbol.upper(), limit=capped_limit, start_time=start_time, end_time=end_time
    )
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
        raise HTTPException(
            status_code=404, detail="Depth comparison not available for symbol"
        )
    return summary


@app.get("/lookup/gex_snapshot")
async def lookup_gex_snapshot(symbol: str) -> Dict[str, Any]:
    """Return the latest cached GEX snapshot for a symbol from Redis."""
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    redis_conn = _get_redis_client()
    key = f"{SNAPSHOT_KEY_PREFIX}{symbol.upper()}"
    raw = redis_conn.get(key)
    if not raw:
        raise HTTPException(status_code=404, detail="Snapshot not available for symbol")
    try:
        snapshot = json.loads(raw)
    except Exception:
        try:
            snapshot = json.loads(raw.decode("utf-8"))
        except Exception:
            snapshot = {"raw": raw.decode("utf-8", errors="replace")}
    return {
        "symbol": symbol.upper(),
        "sum_gex_vol": snapshot.get("sum_gex_vol")
        if isinstance(snapshot, dict)
        else None,
        "snapshot": snapshot,
    }


@app.get("/sc")
async def sierra_chart_bridge(symbol: str = "NQ_NDX") -> Dict[str, Any]:
    """Expose the latest sum_gex_vol for Sierra Chart polling."""
    normalized = symbol.upper()
    redis_conn = _get_redis_client()
    key = f"{SNAPSHOT_KEY_PREFIX}{normalized}"
    raw = redis_conn.get(key)
    if not raw:
        raise HTTPException(
            status_code=404, detail=f"Snapshot not available for {normalized}"
        )
    try:
        snapshot = json.loads(raw)
    except Exception:
        try:
            snapshot = json.loads(raw.decode("utf-8"))
        except Exception:
            snapshot = {"raw": raw.decode("utf-8", errors="replace")}
    if not isinstance(snapshot, dict):
        raise HTTPException(status_code=502, detail="Snapshot payload malformed")
    value = snapshot.get("sum_gex_vol")
    if value is None:
        raise HTTPException(
            status_code=404, detail=f"sum_gex_vol missing for {normalized}"
        )
    return {
        "symbol": normalized,
        "sum_gex_vol": value,
        "timestamp": snapshot.get("timestamp"),
    }


@app.websocket("/ws/sc")
async def sierra_chart_websocket(websocket: WebSocket, symbol: str = "NQ_NDX") -> None:
    """Stream sum_gex_vol updates via Redis pubsub; filters to the requested symbol."""
    await websocket.accept()
    normalized = (symbol or "NQ_NDX").upper()
    redis_conn = _get_redis_client()
    pubsub = redis_conn.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(SNAPSHOT_PUBSUB_CHANNEL)

    async def _send_snapshot(payload: Dict[str, Any]) -> None:
        value = payload.get("sum_gex_vol")
        if value is None:
            return
        await websocket.send_json(
            {
                "symbol": normalized,
                "sum_gex_vol": value,
                "timestamp": payload.get("timestamp"),
            }
        )

    # Send the latest cached value immediately (matches /sc behavior)
    try:
        key = f"{SNAPSHOT_KEY_PREFIX}{normalized}"
        raw = redis_conn.get(key)
        if raw:
            try:
                snapshot = json.loads(raw)
            except Exception:
                try:
                    snapshot = json.loads(raw.decode("utf-8"))
                except Exception:
                    snapshot = {"raw": raw.decode("utf-8", errors="replace")}
            if isinstance(snapshot, dict) and snapshot.get("sum_gex_vol") is not None:
                await _send_snapshot(snapshot)
    except Exception:
        LOGGER.debug("Failed to send initial /ws/sc snapshot", exc_info=True)

    try:
        while True:
            try:
                message = await asyncio.to_thread(pubsub.get_message, timeout=1.5)
            except Exception:
                message = None
            if not message or message.get("type") != "message":
                await asyncio.sleep(0.1)
                continue
            payload_raw = message.get("data")
            payload: Dict[str, Any]
            try:
                if isinstance(payload_raw, (bytes, bytearray)):
                    payload = json.loads(payload_raw.decode("utf-8"))
                elif isinstance(payload_raw, str):
                    payload = json.loads(payload_raw)
                elif isinstance(payload_raw, dict):
                    payload = payload_raw
                else:
                    continue
            except Exception:
                LOGGER.debug(
                    "Skipping malformed pubsub payload for /ws/sc", exc_info=True
                )
                continue
            if (payload.get("symbol") or "").upper() != normalized:
                continue
            await _send_snapshot(payload)
    except WebSocketDisconnect:
        return
    finally:
        try:
            pubsub.close()
        except Exception:
            pass


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


def _coerce_webhook_payload(raw_payload: Any) -> Dict[str, Any]:
    """Normalize webhook payloads into a dict so topics can be extracted."""
    if isinstance(raw_payload, dict):
        return raw_payload

    if isinstance(raw_payload, list):
        topic = raw_payload[2] if len(raw_payload) > 2 else None
        event_type = raw_payload[3] if len(raw_payload) > 3 else None
        data = raw_payload[4] if len(raw_payload) > 4 else None

        coerced: Dict[str, Any] = {"__raw__": raw_payload}
        if isinstance(topic, str) and topic.strip():
            coerced["topic"] = topic.strip()
        if isinstance(event_type, str) and event_type.strip():
            coerced["event_type"] = event_type.strip()
        if data is not None:
            coerced["data"] = data if isinstance(data, dict) else {"value": data}
        return coerced

    return {"__raw_payload__": raw_payload}


def _extract_webhook_topic(payload: Dict[str, Any]) -> Optional[str]:
    """Pull the topic/object indicator out of a webhook payload."""
    candidate_fields = ("topic", "object", "type", "feed", "kind")
    for field in candidate_fields:
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    nested = payload.get("payload") or payload.get("data")
    if isinstance(nested, dict):
        for field in candidate_fields:
            value = nested.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _normalize_topic(topic: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Lowercase and map known aliases to canonical webhook topics plus optional symbol."""
    if not topic:
        return None, None
    normalized = topic.strip().lower()

    symbol: Optional[str] = None
    for sep in TOPIC_SYMBOL_SEPARATORS:
        if sep in normalized:
            parts = normalized.split(sep, 1)
            normalized = parts[0].strip()
            symbol = (parts[1] or "").strip().upper() or None
            break

    canonical = TOPIC_ALIASES.get(normalized, normalized)
    return canonical, symbol


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


def _cache_market_state(redis_conn, payload: Dict[str, Any]) -> None:
    serialized = json.dumps(payload, default=str)
    pipe = redis_conn.pipeline()
    pipe.setex(UW_MARKET_LATEST_KEY, UW_CACHE_TTL_SECONDS, serialized)
    pipe.lpush(UW_MARKET_HISTORY_KEY, serialized)
    pipe.ltrim(UW_MARKET_HISTORY_KEY, 0, UW_HISTORY_LIMIT - 1)
    pipe.execute()
    try:
        redis_conn.publish(UW_MARKET_STREAM_CHANNEL, serialized)
    except Exception:
        LOGGER.exception("Failed to publish market state update")


def _cache_ml_trade(redis_conn, payload: Dict[str, Any]) -> None:
    serialized = json.dumps(payload, default=str)
    pipe = redis_conn.pipeline()
    pipe.set(ML_TRADE_LATEST_KEY, serialized)
    pipe.lpush(ML_TRADE_HISTORY_KEY, serialized)
    pipe.ltrim(ML_TRADE_HISTORY_KEY, 0, ML_TRADE_HISTORY_LIMIT - 1)
    pipe.execute()
    try:
        redis_conn.publish(ML_TRADE_STREAM_CHANNEL, serialized)
    except Exception:
        LOGGER.exception("Failed to publish ML trade alert")


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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--timeout-keep-alive", type=int, default=30)
    parser.add_argument("--ssl-certfile", help="Path to SSL certificate file")
    parser.add_argument("--ssl-keyfile", help="Path to SSL private key file")
    return parser.parse_args()


def configure_logging(log_level: int = logging.INFO, log_dir: str = "logs") -> None:
    """Set up basic logging configuration for the data pipeline.

    - Ensure the log directory exists
    - Create a rotating file handler for the main process
    - Create a rotating file handler for the status page
    - Keep stream handler for console
    """
    # Ensure directory exists
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to create log directory {log_dir}: {exc}")

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    def _has_file_handler(logger: logging.Logger) -> bool:
        return any(
            isinstance(h, RotatingFileHandler)
            and getattr(h, "baseFilename", "").endswith("data-pipeline.log")
            for h in logger.handlers
        )

    root = logging.getLogger()
    root.setLevel(log_level)
    # Attach file handler if missing
    if not _has_file_handler(root):
        file_path = Path(log_dir) / "data-pipeline.log"
        fh = RotatingFileHandler(
            str(file_path), maxBytes=10 * 1024 * 1024, backupCount=5
        )
        fh.setLevel(log_level)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Disable console logging when file logging is active; otherwise keep a console handler
    if _has_file_handler(root):
        for handler in list(root.handlers):
            if isinstance(handler, logging.StreamHandler):
                root.removeHandler(handler)
    elif not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    # Configure a separate logger for status page
    status_logger = logging.getLogger("data_pipeline.status")
    status_logger.setLevel(logging.INFO)
    try:
        status_fh = RotatingFileHandler(
            str(Path(log_dir) / "status.log"), maxBytes=5 * 1024 * 1024, backupCount=3
        )
        status_fh.setLevel(logging.INFO)
        status_fh.setFormatter(fmt)
        status_logger.addHandler(status_fh)
    except Exception:
        # Don't fail startup if status handler cannot be created
        LOGGER.warning("Could not create status.log handler; continuing without it")


def main() -> None:
    """Entry point used both by ``python data-pipeline.py`` and packaging."""
    args = parse_args()
    # Configure logging early so any modules started by this process inherit the config
    configure_logging(
        log_level=getattr(logging, args.log_level.upper(), logging.INFO),
        log_dir=args.log_dir,
    )
    
    # Build uvicorn config
    uvicorn_config = {
        "app": app,
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level.lower(),
        "timeout_keep_alive": args.timeout_keep_alive,
    }
    
    # Add SSL support if certificates provided
    if args.ssl_certfile and args.ssl_keyfile:
        uvicorn_config["ssl_certfile"] = args.ssl_certfile
        uvicorn_config["ssl_keyfile"] = args.ssl_keyfile
        LOGGER.info(
            "SSL enabled: cert=%s, key=%s", args.ssl_certfile, args.ssl_keyfile
        )
    
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
