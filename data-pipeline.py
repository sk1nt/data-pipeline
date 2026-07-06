#!/usr/bin/env python3
# ruff: noqa: E402
"""Data pipeline orchestration entrypoint.
import math

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
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
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
from src.services.tastytrade_auth_service import get_tastytrade_auth_service  # noqa: E402
from src.services.redis_timeseries import RedisTimeSeriesClient  # noqa: E402
from src.services.redis_flush_worker import FlushWorkerSettings, RedisFlushWorker  # noqa: E402
from src.services.lookup_service import LookupService  # noqa: E402
from src.services.trin_history_recorder import (  # noqa: E402
    TrinHistoryRecorder,
    TrinHistoryRecorderSettings,
)
from src.services.ivr_service import IVRService, IVRServiceSettings  # noqa: E402
from src.services.schwab_streamer import SchwabStreamClient, build_streamer  # noqa: E402
from src.services.social_feed_service import SocialFeedService, FeedConfig  # noqa: E402
from src.services.social_feed_service import SOCIAL_ALL_EVENTS_CHANNEL, SOCIAL_HISTORY_KEY  # noqa: E402
from src.services.correlation_engine import CorrelationEngine  # noqa: E402
from src.services.economic_calendar_service import EconomicCalendarService, CALENDAR_REDIS_KEY  # noqa: E402
from src.services.market_mover_analyzer import MarketMoverAnalyzer, MarketMoverResult  # noqa: E402
from src.services.uw_rest_poller import UWRestPoller, UWRestPollerSettings  # noqa: E402
from src.models.social_event import SocialSource  # noqa: E402
from src.api.routes.datastores import router as datastores_router  # noqa: E402

# Trading panel router
from backend.src.api.trading import router as trading_router  # noqa: E402

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

    def status(self) -> Dict[str, Any]:
        symbols = list(self.streamer.symbols) if self.streamer else []
        return {
            "running": self.is_running,
            "enabled": settings.schwab_enabled,
            "paused": settings.schwab_stream_paused,
            "symbols": symbols,
            "thread_alive": bool(self.thread and self.thread.is_alive()),
            "trade_samples": self.manager.trade_counts.get("schwab", 0),
            "last_trade_ts": self.manager.last_trade_timestamps.get("schwab"),
            "auth": self.streamer.auth_client.status()
            if self.streamer
            else {
                "needs_reauth": False,
                "last_error": None,
                "auto_refresh_running": False,
            },
        }

    def _run_streamer(self) -> None:
        if not self.streamer:
            return
        try:
            self.streamer.start()
        except Exception as exc:  # pragma: no cover - defensive
            auth_status = {}
            try:
                auth_status = self.streamer.auth_client.status()
            except Exception:
                pass
            if auth_status.get("needs_reauth"):
                LOGGER.error(
                    "Schwab streamer failed to start: refresh token is invalid or revoked (%s)",
                    exc,
                )
            else:
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
        self.tastytrade_auth = None
        self.automated_options_service: Optional[Any] = None
        self.trin_history: Optional[TrinHistoryRecorder] = None
        self.ivr_service: Optional[IVRService] = None
        self._ivr_task: Optional[asyncio.Task[None]] = None
        self.gex_poller: Optional[GEXBotPoller] = None
        self.gex_nq_poller: Optional[GEXBotPoller] = None
        self.redis_client: Optional[RedisClient] = None
        self.rts: Optional[RedisTimeSeriesClient] = None
        self.flush_worker: Optional[RedisFlushWorker] = None
        self.lookup_service: Optional[LookupService] = None
        self.schwab_service = SchwabStreamingService(self)
        self.social_feed: Optional[SocialFeedService] = None
        self.correlation_engine: Optional[CorrelationEngine] = None
        self.calendar_service: Optional[EconomicCalendarService] = None
        self.uw_rest_poller: Optional[UWRestPoller] = None
        self._discord_bot_proc: Optional[Any] = None  # subprocess.Popen
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
        # Adopt an already-running discord bot process so status dot and restart work
        if self._discord_bot_proc is None:
            try:
                import subprocess as _sp
                result = _sp.run(
                    ["pgrep", "-f", "run_discord_bot.py"], capture_output=True, text=True
                )
                pid_str = (result.stdout.strip().split() or [None])[0]
                if pid_str:
                    import psutil
                    p = psutil.Process(int(pid_str))
                    if p.is_running():
                        self._discord_bot_proc = p
                        LOGGER.info("Adopted existing discord bot process (pid=%s)", pid_str)
            except Exception:
                pass
        for service in (
            "tastytrade",
            "schwab",
            "gex_poller",
            "gex_nq_poller",
            "redis_flush",
            "social_feed",
            "correlation",
            "calendar",
            "uw_rest",
            "ivr",
        ):
            self.start_service(service)

    async def stop(self) -> None:
        """Stop all managed services in a best-effort fashion."""
        for service in (
            "ivr",
            "correlation",
            "calendar",
            "uw_rest",
            "social_feed",
            "tastytrade",
            "tastytrade_auth",
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
        return {
            "tastytrade_auth": (
                self.tastytrade_auth.status() if self.tastytrade_auth else {}
            ),
            "tastytrade_order_replay": {
                "running": bool(
                    self.automated_options_service
                    and self.automated_options_service._replay_task
                    and not self.automated_options_service._replay_task.done()
                )
            },
            "tastytrade_streamer": tasty_status,
            "schwab_streamer": self.schwab_service.status(),
            "gex_poller": getattr(self.gex_poller, "status", lambda: {})(),
            "gex_nq_poller": getattr(self.gex_nq_poller, "status", lambda: {})(),
            "redis_flush_worker": getattr(self.flush_worker, "status", lambda: {})(),
            "uw_rest_poller": getattr(self.uw_rest_poller, "status", lambda: {})(),
            "trin_history_recorder": {
                "running": bool(self.trin_history and self.trin_history.is_running),
                "db_path": settings.tastytrade_trin_history_db_path,
                "parquet_dir": settings.tastytrade_trin_history_parquet_dir,
            },
            "ivr_service": {
                "running": bool(self._ivr_task and not self._ivr_task.done()),
            },
            "lookup_service": {
                "ready": bool(self.lookup_service),
                "recent_depth_diffs": list(self.last_depth_comparison.values())[:3],
            },
            "market_data_metrics": self._status_metrics_snapshot(),
            "discord_bot": {
                "running": bool(
                    self._discord_bot_proc is not None and (
                        self._discord_bot_proc.poll() is None
                        if hasattr(self._discord_bot_proc, "poll")
                        else self._discord_bot_proc.is_running()
                    )
                ),
                "pid": getattr(self._discord_bot_proc, "pid", None),
            },
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
        if not self.lookup_service and self.redis_client and self.rts:
            self.lookup_service = LookupService(self.redis_client, self.rts)

    def start_service(self, name: str) -> None:
        """Start a specific service by name if it is enabled via settings."""
        name = name.lower()
        self._ensure_event_loop()
        self._ensure_redis_clients()
        if name == "tastytrade_auth":
            if self.tastytrade_auth:
                return
            if os.getenv("TASTYTRADE_AUTH_KEEPER_ENABLED", "false").lower() != "true":
                LOGGER.info(
                    "TastyTrade auth keeper disabled "
                    "(set TASTYTRADE_AUTH_KEEPER_ENABLED=true to enable)"
                )
                return
            if not settings.tastytrade_client_secret or not settings.tastytrade_refresh_token:
                LOGGER.warning("TastyTrade auth keeper disabled; credentials missing")
                return
            from src.services.automated_options_service import AutomatedOptionsService

            self.tastytrade_auth = get_tastytrade_auth_service()
            self.tastytrade_auth.start()
            self.automated_options_service = AutomatedOptionsService()
            self.automated_options_service.start_pending_auth_replay_worker()
            LOGGER.info("TastyTrade auth keeper started")
            return
        if name == "tastytrade" and settings.tastytrade_stream_enabled:
            if self.tastytrade and self.tastytrade.is_running:
                return
            if (
                not self.tastytrade_auth
                and os.getenv("TASTYTRADE_AUTH_KEEPER_ENABLED", "false").lower()
                == "true"
            ):
                self.start_service("tastytrade_auth")
            self.tastytrade = TastyTradeStreamer(
                StreamerSettings(
                    client_id=settings.tastytrade_client_id or "",
                    client_secret=settings.tastytrade_client_secret or "",
                    refresh_token=settings.tastytrade_refresh_token or "",
                    symbols=settings.tastytrade_symbol_list,
                    depth_levels=settings.tastytrade_depth_cap,
                    enable_depth=getattr(settings, "tastytrade_enable_depth", False),
                    greeks_symbols=settings.tastytrade_greeks_symbol_list,
                    enable_greeks=settings.tastytrade_enable_greeks,
                ),
                on_trade=self._handle_trade_event,
                on_depth=self._handle_depth_event,
                on_greeks=self._handle_greeks_event,
                auth_service=self.tastytrade_auth,
            )
            if not self.trin_history:
                self.trin_history = TrinHistoryRecorder(
                    TrinHistoryRecorderSettings(
                        db_path=Path(settings.tastytrade_trin_history_db_path),
                        parquet_dir=Path(settings.tastytrade_trin_history_parquet_dir),
                        flush_interval_seconds=settings.tastytrade_trin_history_flush_seconds,
                    )
                )
                self.trin_history.start()
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
            # Seed the main poller with the configured symbol list so startup is
            # resilient even when the live discovery endpoint is unavailable.
            # The poller will still refresh the supported set from GEXBot and
            # apply exclusions, but it no longer depends on that call to avoid
            # silently running with zero symbols.
            base_symbols = settings.gex_symbol_list
            self.gex_poller = GEXBotPoller(
                GEXBotPollerSettings(
                    api_key=settings.gexbot_api_key,
                    symbols=base_symbols,
                    interval_seconds=settings.gex_poll_interval_seconds,
                    aggregation_period=settings.gex_poll_aggregation,
                    # Main poller should poll at 5s during RTH regardless of .env
                    rth_interval_seconds=5.0,
                    off_hours_interval_seconds=settings.gex_poll_off_hours_interval_seconds,
                    dynamic_schedule=settings.gex_poll_dynamic_schedule,
                    # Keep NQ_NDX and VIX on the dedicated fast poller, but do
                    # not exclude SPX from the main poller. SPX is a shared
                    # reference symbol and should still be cached even if the
                    # NQ poller is disabled.
                    exclude_symbols=["NQ_NDX", "VIX"],
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
        elif name == "social_feed" and settings.social_feed_enabled:
            if self.social_feed:
                return
            feeds = []
            priority_handles = {
                h.strip().lower()
                for h in settings.social_feed_priority_handles.split(",")
                if h.strip()
            }
            for url in settings.social_feed_urls.split("|"):
                url = url.strip()
                if not url:
                    continue
                # Determine source type and author from URL
                if url.startswith("ts:"):
                    # Truth Social account: "ts:username:Display Name"
                    parts = url.split(":", 2)
                    username = parts[1] if len(parts) > 1 else ""
                    author = parts[2] if len(parts) > 2 else username
                    is_priority = username.lower() in priority_handles
                    if username:
                        feeds.append(FeedConfig(username, SocialSource.TRUTH_SOCIAL, author, priority=is_priority))
                elif "nitter" in url:
                    # Nitter RSS: extract @handle from path
                    from urllib.parse import urlparse
                    path = urlparse(url).path.strip("/")
                    handle = path.split("/")[0] if "/" in path else path.replace("/rss", "")
                    is_priority = handle.lower() in priority_handles
                    feeds.append(FeedConfig(url, SocialSource.TWITTER, f"@{handle}", priority=is_priority))
                else:
                    # Derive a readable author name from the domain
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc.replace("www.", "").replace("feeds.", "").replace("search.", "")
                    author_map = {
                        "marketwatch.com": "MarketWatch",
                        "cnbc.com": "CNBC",
                        "finance.yahoo.com": "Yahoo Finance",
                        "benzinga.com": "Benzinga",
                        "bloomberg.com": "Bloomberg",
                        "feedburner.com": "ZeroHedge",
                        "news.google.com": "Google News",
                        "seekingalpha.com": "Seeking Alpha",
                        "investing.com": "Investing.com",
                    }
                    author = author_map.get(domain, domain)
                    is_priority = domain.split(".")[0].lower() in priority_handles
                    feeds.append(FeedConfig(url, SocialSource.NEWS_RSS, author, priority=is_priority))
            if feeds:
                self.social_feed = SocialFeedService(
                    redis_client=self.redis_client,
                    feeds=feeds,
                    rth_interval_seconds=settings.social_feed_rth_interval_seconds,
                    off_hours_interval_seconds=settings.social_feed_off_hours_interval_seconds,
                    priority_interval_seconds=settings.social_feed_priority_interval_seconds,
                    min_score_threshold=settings.social_min_score_threshold,
                    dedup_ttl_seconds=settings.social_dedup_ttl_seconds,
                )
                self.social_feed.start()
                LOGGER.info("Social feed service started with %d feeds", len(feeds))
            else:
                LOGGER.warning("Social feed enabled but no URLs configured")
        elif name == "correlation" and settings.correlation_enabled:
            if self.correlation_engine:
                return
            self.correlation_engine = CorrelationEngine(
                redis_client=self.redis_client,
                window_seconds=settings.correlation_window_seconds,
                volume_spike_multiplier=settings.correlation_volume_spike_multiplier,
                gex_shift_abs=settings.correlation_gex_shift_abs,
                price_move_pct=settings.correlation_price_move_pct,
                cooldown_seconds=settings.correlation_cooldown_seconds,
            )
            self.correlation_engine.start()
            LOGGER.info("Correlation engine started")
        elif name == "calendar":
            if self.calendar_service:
                return
            self.calendar_service = EconomicCalendarService(redis_client=self.redis_client)
            self.calendar_service.start()
            LOGGER.info("Economic calendar service started")
        elif name == "uw_rest" and settings.uw_rest_poller_enabled and settings.uw_api_key:
            if self.uw_rest_poller:
                return
            self.uw_rest_poller = UWRestPoller(
                UWRestPollerSettings(
                    api_key=settings.uw_api_key,
                    sweep_interval_rth=settings.uw_sweep_interval_rth,
                    alert_interval_rth=settings.uw_alert_interval_rth,
                    tide_interval_rth=settings.uw_tide_interval_rth,
                    darkpool_interval_rth=settings.uw_darkpool_interval_rth,
                    sector_interval_rth=settings.uw_sector_interval_rth,
                    off_hours_multiplier=settings.uw_off_hours_multiplier,
                    min_sweep_premium=settings.uw_min_sweep_premium,
                ),
                redis_client=self.redis_client,
            )
            self.uw_rest_poller.start()
            LOGGER.info("UW REST poller started")
        elif name == "ivr":
            if self._ivr_task and not self._ivr_task.done():
                return
            self._ensure_redis_clients()
            self.ivr_service = IVRService(
                config=IVRServiceSettings(
                    db_path=settings.data_path / "ivr_data.db",
                    option_trades_db=settings.data_path / "uw_messages.db",
                ),
                redis_client=self.redis_client,
            )
            self._ivr_task = asyncio.create_task(
                self._run_ivr_loop(), name="ivr-service"
            )
            LOGGER.info("IVR service started")
        elif name == "discord_bot":
            import subprocess
            import sys
            proc = self._discord_bot_proc
            if proc is not None:
                try:
                    alive = proc.poll() is None if hasattr(proc, "poll") else proc.is_running()
                except Exception:
                    alive = False
                if alive:
                    return  # already running
            bot_script = Path(__file__).parent / "discord-bot" / "run_discord_bot.py"
            self._discord_bot_proc = subprocess.Popen(
                [sys.executable, str(bot_script)],
                cwd=str(Path(__file__).parent),
            )
            LOGGER.info("Discord bot started (pid=%d)", self._discord_bot_proc.pid)

    async def _run_ivr_loop(self) -> None:
        """Periodically aggregate IV history and compute IVR per symbol."""
        while True:
            try:
                if self.ivr_service:
                    rows = self.ivr_service.aggregate_daily_iv()
                    if rows:
                        results = self.ivr_service.compute_ivr_batch()
                        LOGGER.info("IVR computed for %d symbols", len(results))
            except Exception:
                LOGGER.exception("Error in IVR computation loop")
            await asyncio.sleep(900)  # 15 minutes

    async def stop_service(self, name: str) -> None:
        """Stop a running service and clean up the local reference."""
        name = name.lower()
        if name == "tastytrade" and self.tastytrade:
            await self.tastytrade.stop()
            self.tastytrade = None
            if self.trin_history:
                await self.trin_history.stop()
                self.trin_history = None
            LOGGER.info("TastyTrade streamer stopped")
        elif name == "tastytrade_auth":
            if self.automated_options_service:
                await self.automated_options_service.stop_pending_auth_replay_worker()
                self.automated_options_service = None
            if self.tastytrade_auth:
                await self.tastytrade_auth.stop()
                self.tastytrade_auth = None
                LOGGER.info("TastyTrade auth keeper stopped")
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
        elif name == "social_feed" and self.social_feed:
            await self.social_feed.stop()
            self.social_feed = None
            LOGGER.info("Social feed service stopped")
        elif name == "correlation" and self.correlation_engine:
            await self.correlation_engine.stop()
            self.correlation_engine = None
            LOGGER.info("Correlation engine stopped")
        elif name == "calendar" and self.calendar_service:
            await self.calendar_service.stop()
            self.calendar_service = None
            LOGGER.info("Economic calendar service stopped")
        elif name == "uw_rest" and self.uw_rest_poller:
            await self.uw_rest_poller.stop()
            self.uw_rest_poller = None
            LOGGER.info("UW REST poller stopped")
        elif name == "ivr" and self._ivr_task:
            self._ivr_task.cancel()
            try:
                await self._ivr_task
            except (asyncio.CancelledError, Exception):
                pass
            self._ivr_task = None
            self.ivr_service = None
            LOGGER.info("IVR service stopped")
        elif name == "discord_bot":
            proc = self._discord_bot_proc
            if proc is not None:
                try:
                    alive = proc.poll() is None if hasattr(proc, "poll") else proc.is_running()
                    if alive:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10) if hasattr(proc, "wait") else proc.wait(timeout=10)
                        except Exception:
                            proc.kill()
                        LOGGER.info("Discord bot stopped (pid=%d)", proc.pid)
                except Exception as exc:
                    LOGGER.warning("Error stopping discord bot: %s", exc)
            self._discord_bot_proc = None

    async def restart_service(self, name: str) -> None:
        """Convenience helper for the ``/control`` endpoint."""
        await self.stop_service(name)
        self.start_service(name)

    async def poll_service_now(
        self, name: str, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """Force a one-off poll for a managed service without relying on its loop."""
        self._ensure_event_loop()
        self._ensure_redis_clients()
        name = name.lower()
        if name != "gex_nq_poller":
            raise HTTPException(
                status_code=400,
                detail=f"poll-now is only supported for gex_nq_poller, not {name}",
            )

        target_symbol = (symbol or "NQ_NDX").upper().strip() or "NQ_NDX"
        poller = self.gex_nq_poller
        if poller is None:
            symbols = settings.gex_nq_poll_symbol_list
            if not symbols:
                raise HTTPException(
                    status_code=400,
                    detail="No symbols configured for NQ poller",
                )
            poller = GEXBotPoller(
                GEXBotPollerSettings(
                    api_key=settings.gexbot_api_key or "",
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

        snapshot = await poller.fetch_symbol_now(target_symbol)
        return {
            "service": name,
            "symbol": target_symbol,
            "fetched": snapshot is not None,
            "snapshot": snapshot,
        }

    def _silence_streamer_logs(self) -> None:
        for logger_name in NOISY_STREAM_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    async def _handle_trade_event(
        self, payload: Dict[str, Any], source: str = "tastytrade"
    ) -> None:
        """Persist trade ticks to RedisTimeSeries in a thread so I/O stays async friendly."""
        symbol = str(payload.get("symbol", "")).upper()
        if source.lower() == "tastytrade" and symbol.startswith("$TRIN"):
            if self.trin_history:
                self.trin_history.record_trade({**payload, "source": source})
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

    async def _handle_greeks_event(
        self, payload: Dict[str, float]
    ) -> None:
        """Persist real-time Greeks to Redis timeseries."""
        if not self.rts:
            return
        await asyncio.to_thread(self._write_greeks_timeseries, payload)

    def _write_greeks_timeseries(self, payload: Dict[str, Any]) -> None:
        """Write greeks samples to Redis timeseries per field."""
        symbol = payload.get("symbol", "").upper() or "UNKNOWN"
        timestamp_ms = _timestamp_ms(payload.get("timestamp"))
        samples = []
        for field in (
            "delta", "gamma", "theta", "vega", "rho",
            "implied_volatility", "underlying_price", "option_price",
        ):
            value = payload.get(field)
            if value is None:
                continue
            samples.append((
                f"ts:greeks:{field}:{symbol}",
                timestamp_ms,
                float(value),
                {"symbol": symbol, "type": "greeks", "field": field},
            ))
        if self.rts and samples:
            self.rts.multi_add(samples)

    def _write_trade_timeseries(self, payload: Dict[str, Any], source: str) -> None:
        """Write trade price/size samples and keep per-source counters."""
        symbol = payload.get("symbol", "").upper() or "UNKNOWN"
        timestamp_ms = _timestamp_ms(payload.get("timestamp"))
        price = float(payload.get("price") or payload.get("last") or 0.0)
        size = float(payload.get("size") or payload.get("volume") or 0.0)
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
        self.last_trade_ts = payload.get("timestamp") or datetime.now(timezone.utc).isoformat()
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
                payload.get("timestamp") or datetime.now(timezone.utc).isoformat()
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
            "timestamp": payload.get("timestamp") or datetime.now(timezone.utc).isoformat(),
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
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

    def _status_metrics_snapshot(self) -> Dict[str, Any]:
        """Return market metrics for /status without temporarily hidden providers."""
        snapshot = self.metrics_snapshot()
        for key in (
            "trades_by_source",
            "level2_by_source",
            "last_trade_timestamps",
            "last_depth_timestamps",
        ):
            values = snapshot.get(key)
            if isinstance(values, dict):
                values.pop("schwab", None)
        for key in ("trades_by_symbol", "level2_by_symbol"):
            values = snapshot.get(key)
            if isinstance(values, dict):
                for per_symbol in values.values():
                    if isinstance(per_symbol, dict):
                        per_symbol.pop("schwab", None)
        return snapshot

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

# Redis channel for TastyTrade market data
TASTYTRADE_TRADE_CHANNEL = "market_data:tastytrade:trades"

# ML Trade bot constants (kept for potential future use)
ML_TRADE_LATEST_KEY = "trade:ml-bot:latest"
ML_TRADE_HISTORY_KEY = "trade:ml-bot"
ML_TRADE_STREAM_CHANNEL = "trade:ml-bot:stream"
ML_TRADE_HISTORY_LIMIT = 500


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure ServiceManager starts before handling requests and shuts down cleanly."""
    LOGGER.info("Starting services during FastAPI lifespan")
    service_manager.start()

    try:
        yield
    finally:
        LOGGER.info("Stopping services during FastAPI lifespan")
        try:
            await asyncio.wait_for(service_manager.stop(), timeout=5.0)
        except Exception:
            LOGGER.warning("Error during graceful service shutdown", exc_info=True)


app = FastAPI(title="Data Pipeline", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://unusualwhales.com", "http://192.168.168.151:8877"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
    allow_credentials=True,
)

# Mount trading panel API and static page
app.include_router(trading_router, prefix="/api/v1", tags=["trading"])
app.include_router(datastores_router, prefix="/api", tags=["datastores"])


@app.get("/order-panel")
async def order_panel():
    from fastapi.responses import FileResponse
    panel_path = PROJECT_ROOT / "frontend" / "src" / "order_panel.html"
    return FileResponse(str(panel_path))


@app.get("/api/calendar")
async def get_calendar() -> Dict[str, Any]:
    """Return today's high-impact economic events (cached from Forex Factory feed)."""
    redis_conn = _get_redis_client()
    try:
        raw = redis_conn.get(CALENDAR_REDIS_KEY)
        if raw:
            events = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
            return {"events": events, "count": len(events)}
    except Exception:
        pass
    return {"events": [], "count": 0}


@app.get("/gex-monitor")
async def gex_monitor():
    from fastapi.responses import FileResponse
    monitor_path = PROJECT_ROOT / "frontend" / "src" / "gex_monitor.html"
    return FileResponse(
        str(monitor_path),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/gex-monitor/popout")
async def gex_monitor_popout(w: int = 360, h: int = 1200):
    """Tiny launcher that opens /gex-monitor as a sized popup window."""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(f"""<!DOCTYPE html><html><head><title>GEX Popout</title></head>
<body style="background:#0d1117;color:#e6edf3;font-family:system-ui;display:flex;align-items:center;justify-content:center;height:100vh">
<script>
var popup = window.open('/gex-monitor','gex_monitor','width={w},height={h},resizable=yes,scrollbars=yes,toolbar=no,menubar=no,location=no,status=no');
if (popup) {{ document.body.innerHTML = '<p style="font-size:13px">GEX Monitor opened — you can close this tab.</p>'; }}
else {{ document.body.innerHTML = '<p style="font-size:13px">Popup blocked. <a href="/gex-monitor" target="_blank" style="color:#58a6ff">Open manually</a></p>'; }}
</script></body></html>""")


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
    # Log status access via dedicated status logger at DEBUG level
    logging.getLogger("data_pipeline.status").debug(
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


@app.post("/control/{service_name}/poll-now")
async def control_poll_now(
    service_name: str, symbol: Optional[str] = None
) -> Dict[str, Any]:
    """Force a single poll for supported services (no auth)."""
    try:
        return await service_manager.poll_service_now(service_name, symbol=symbol)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/control/{service_name}/status")
async def control_status(service_name: str) -> Dict[str, Any]:
    """Return a service-specific status snapshot (no auth)."""
    _service_attr_map = {
        "schwab": "schwab_service",
    }
    try:
        attr = _service_attr_map.get(service_name, service_name)
        svc = getattr(service_manager, attr, None)
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
    .service-controls { margin: 1.25rem 0; display: flex; flex-wrap: wrap; gap: 0.75rem; }
    .svc-group { background: #1e232b; border-radius: 8px; padding: 0.6rem 0.9rem; display: flex; align-items: center; gap: 0.4rem; }
    .svc-label { font-size: 0.85rem; color: #aaa; margin-right: 0.3rem; min-width: 6rem; }
    .svc-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 0.35rem; background: #555; }
    button { cursor: pointer; border: none; border-radius: 4px; padding: 0.25rem 0.6rem; font-size: 0.8rem; color: #fff; }
    .btn-start  { background: #1a7a2e; }
    .btn-stop   { background: #8a1a1a; }
    .btn-restart{ background: #555; }
    button:hover { opacity: 0.85; }
  </style>
</head>
<body>
  <h1>Data Pipeline Status</h1>
  <p class=\"warning\">Dashboard auto-refreshes every 1 second.</p>
  <div id=\"controls\" class=\"service-controls\"></div>
  <pre id=\"status\">Loading...</pre>
  <script>
    const SERVICES = [
      'schwab',
      'tastytrade',
      'gex_poller',
      'gex_nq_poller',
      'redis_flush',
      'discord_bot',
    ];

    // Keys in /status JSON that map to each service's running state
    const STATUS_KEYS = {
      schwab:       d => d.schwab_streamer && d.schwab_streamer.running,
      tastytrade:   d => d.tastytrade_streamer && d.tastytrade_streamer.running,
      gex_poller:   d => d.gex_poller && d.gex_poller.running,
      gex_nq_poller:d => d.gex_nq_poller && d.gex_nq_poller.running,
      redis_flush:  d => d.redis_flush_worker && d.redis_flush_worker.running,
      discord_bot:  d => d.discord_bot && d.discord_bot.running,
    };

    async function callControl(service, action) {
      try {
        const res = await fetch(`/control/${service}/${action}`, { method: action === 'status' ? 'GET' : 'POST' });
        const data = await res.json();
        if (!res.ok) { alert(`${action} ${service} failed: ${JSON.stringify(data)}`); return; }
        if (action !== 'status') alert(`${action} ${service}: ${JSON.stringify(data)}`);
      } catch (err) { alert(`Error calling ${action} on ${service}: ${err}`); }
    }

    let lastStatus = {};

    function updateDots(data) {
      SERVICES.forEach(s => {
        const dot = document.getElementById(`dot-${s}`);
        if (!dot) return;
        const running = STATUS_KEYS[s] ? STATUS_KEYS[s](data) : null;
        dot.style.background = running === true ? '#2ecc71' : running === false ? '#e74c3c' : '#555';
        dot.title = running === true ? 'running' : running === false ? 'stopped' : 'unknown';
      });
    }

    function renderControls() {
      const container = document.getElementById('controls');
      SERVICES.forEach(s => {
        const grp = document.createElement('div');
        grp.className = 'svc-group';
        const dot = document.createElement('span');
        dot.className = 'svc-dot';
        dot.id = `dot-${s}`;
        const lbl = document.createElement('span');
        lbl.className = 'svc-label';
        lbl.textContent = s;
        const btnStart   = document.createElement('button');
        btnStart.className = 'btn-start';
        btnStart.textContent = 'Start';
        btnStart.onclick = () => callControl(s, 'start');
        const btnStop    = document.createElement('button');
        btnStop.className = 'btn-stop';
        btnStop.textContent = 'Stop';
        btnStop.onclick = () => callControl(s, 'stop');
        const btnRestart = document.createElement('button');
        btnRestart.className = 'btn-restart';
        btnRestart.textContent = 'Restart';
        btnRestart.onclick = () => callControl(s, 'restart');
        grp.appendChild(dot);
        grp.appendChild(lbl);
        grp.appendChild(btnStart);
        grp.appendChild(btnStop);
        grp.appendChild(btnRestart);
        container.appendChild(grp);
      });
    }

    async function refresh() {
      try {
        const res = await fetch('/status');
        const data = await res.json();
        document.getElementById('status').textContent = JSON.stringify(data, null, 2);
        updateDots(data);
      } catch (err) {
        document.getElementById('status').textContent = 'Error: ' + err;
      }
    }

    renderControls();
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


@app.get("/status.html", response_class=HTMLResponse)
async def status_page(request: Request) -> str:
    """Serve a lightweight HTML dashboard for ops users."""
    logging.getLogger("data_pipeline.status").debug(
        "Status page requested from %s", getattr(request.client, "host", "unknown")
    )
    return STATUS_PAGE


# ---------------------------------------------------------------------------
# Market Movers API
# ---------------------------------------------------------------------------

class MarketMoversRequest(BaseModel):
    """Parameters for the market-mover analysis endpoint."""
    lookback_days:         int   = 21      # default 3 weeks
    min_realized_impact:   float = 0.0    # 0 = return everything (noise flagged)
    top_n:                 int   = 100
    noise_floor:           float = 5.0    # events below this are flagged as noise
    tickers:               list[str] = []  # [] = use defaults (ES_SPX, SPY, QQQ)


class MarketMoversResponse(BaseModel):
    lookback_days:         int
    total_events:          int
    mover_count:           int
    noise_count:           int
    results:               list[MarketMoverResult]


@app.post("/api/market-movers", response_model=MarketMoversResponse)
async def market_movers_endpoint(req: MarketMoversRequest) -> MarketMoversResponse:
    """Rank historical social/news events by their *realized* market impact.

    Returns events ordered by how much actual price movement, GEX shift, and
    volume followed the post.  Low-impact events are flagged ``is_noise=true``
    rather than silently dropped so callers can audit the full picture.
    """
    from src.services.market_mover_analyzer import DEFAULT_TICKERS

    tickers = None
    if req.tickers:
        # Build sub-dict from requested tickers, fall back to defaults for unknowns
        tickers = {
            t: DEFAULT_TICKERS.get(t, t)
            for t in req.tickers
            if t in DEFAULT_TICKERS
        } or None

    analyzer = MarketMoverAnalyzer(noise_floor=req.noise_floor, tickers=tickers)
    try:
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: analyzer.analyze(
                lookback_days=req.lookback_days,
                min_realized_impact=req.min_realized_impact,
                top_n=req.top_n,
            ),
        )
    except Exception as exc:
        LOGGER.exception("market-movers analysis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    movers = [r for r in results if not r.is_noise]
    noise  = [r for r in results if r.is_noise]
    return MarketMoversResponse(
        lookback_days=req.lookback_days,
        total_events=len(results),
        mover_count=len(movers),
        noise_count=len(noise),
        results=results,
    )


@app.get("/api/market-movers/realtime", response_model=MarketMoversResponse)
async def market_movers_realtime_endpoint(
    hours: int = 24,
    top_n: int = 50,
    noise_floor: float = 5.0,
) -> MarketMoversResponse:
    """Convenience GET endpoint: last *hours* of events only."""
    analyzer = MarketMoverAnalyzer(noise_floor=noise_floor)
    try:
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: analyzer.analyze_realtime(lookback_hours=hours, top_n=top_n),
        )
    except Exception as exc:
        LOGGER.exception("market-movers realtime failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    movers = [r for r in results if not r.is_noise]
    noise  = [r for r in results if r.is_noise]
    return MarketMoversResponse(
        lookback_days=max(1, (hours + 23) // 24),
        total_events=len(results),
        mover_count=len(movers),
        noise_count=len(noise),
        results=results,
    )


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


@app.api_route("/uw", methods=["POST", "OPTIONS"])
async def universal_webhook(request: Request):
    """Accept UW Phoenix websocket messages in array format.
    
    Expected format: [joinRef, ref, topic, eventType, payload]
    - topic: "market_agg_socket" or "option_trades_super_algo" or "option_trades_super_algo:SPX"
    - eventType: event type from Phoenix (e.g., "update", "new_data")
    - payload: message data object
    """
    raw_body = await request.body()

    try:
        raw_payload = json.loads(raw_body) if raw_body else []
    except json.JSONDecodeError as e:
        LOGGER.error("Failed to parse JSON body: %s", e)
        return {"status": "error", "reason": "invalid_json"}

    # Validate Phoenix websocket array format: [joinRef, ref, topic, eventType, payload]
    if not isinstance(raw_payload, list) or len(raw_payload) != 5:
        LOGGER.warning(
            "/uw received invalid format, expected 5-element array, got: %s (len=%s)",
            type(raw_payload).__name__,
            len(raw_payload) if isinstance(raw_payload, list) else "N/A"
        )
        return {"status": "error", "reason": "invalid_format"}

    topic = raw_payload[2] if len(raw_payload) > 2 else None
    LOGGER.info("/uw received Phoenix message, topic=%s", topic)

    try:
        from src.services.uw_message_service import UWMessageService
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
        LOGGER.exception("Failed to process UW Phoenix message: %s", e)
        return {"status": "error", "reason": str(e)}


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


@app.get("/lookup/trin_history")
async def lookup_trin_history(
    symbol: str,
    limit: int = 100,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> Dict[str, Any]:
    """Return persisted TRIN history from DuckDB/Parquet-backed storage."""
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    lookup = service_manager.lookup_service
    if not lookup:
        raise HTTPException(status_code=503, detail="Lookup service unavailable")
    capped_limit = max(1, min(limit, 1000))
    history = lookup.trin_history(
        symbol.upper(), limit=capped_limit, start_time=start_time, end_time=end_time
    )
    return {
        "symbol": symbol.upper(),
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



def _sanitize_floats(obj):
    """Recursively replace NaN/Inf floats with None for JSON compliance."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_floats(v) for v in obj]
    return obj

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
    snapshot = _sanitize_floats(snapshot)
    return {
        "symbol": symbol.upper(),
        "sum_gex_vol": snapshot.get("sum_gex_vol")
        if isinstance(snapshot, dict)
        else None,
        "snapshot": snapshot,
    }


def _loads_json_payload(raw: Any) -> Any:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _market_agg_regime(ratio: Optional[float]) -> str:
    if ratio is None:
        return "unknown"
    if ratio < 0.80:
        return "long"
    if ratio > 1.0:
        return "short"
    return "neutral"


def _market_agg_snapshot_from_cache(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    ratio = _safe_float(data.get("put_call_ratio"))
    return {
        "type": "market_agg",
        "put_call_ratio": ratio,
        "regime": _market_agg_regime(ratio),
        "call_premium": _safe_float(data.get("call_premium")),
        "put_premium": _safe_float(data.get("put_premium")),
        "call_volume": data.get("call_volume"),
        "put_volume": data.get("put_volume"),
        "net_call_premium": _safe_float(data.get("net_call_premium")),
        "net_put_premium": _safe_float(data.get("net_put_premium")),
        "net_volume": data.get("net_volume"),
        "bar_bias": data.get("bar_bias"),
        "overall_bias": data.get("overall_bias"),
        "timestamp": data.get("timestamp"),
        "date": data.get("date"),
        "received_at": payload.get("received_at"),
        "snapshot": payload,
    }


@app.get("/lookup/market_agg")
async def lookup_market_agg() -> Dict[str, Any]:
    """Return the latest cached market aggregation snapshot from Redis."""
    redis_conn = _get_redis_client()
    raw = redis_conn.get("uw:market_agg:latest")
    if not raw:
        raise HTTPException(
            status_code=404, detail="Market aggregation snapshot not available"
        )
    try:
        payload = _loads_json_payload(raw)
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail="Market aggregation payload malformed"
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=502, detail="Market aggregation payload malformed"
        )
    return _market_agg_snapshot_from_cache(payload)


@app.get("/market-agg/history")
async def market_agg_history(limit: int = 100) -> Dict[str, Any]:
    """Return persisted market aggregation rows from DuckDB."""
    capped = min(max(limit, 1), 1000)
    db_path = settings.data_path / "uw_messages.db"
    if not db_path.exists():
        raise HTTPException(
            status_code=404, detail="Market aggregation history database not available"
        )
    try:
        with duckdb.connect(str(db_path), read_only=True) as conn:
            rows = conn.execute(
                """
                SELECT
                    received_at,
                    date,
                    call_premium,
                    put_premium,
                    call_premium_otm_only,
                    put_premium_otm_only,
                    delta,
                    gamma,
                    theta,
                    vega
                FROM market_agg_state
                ORDER BY received_at DESC
                LIMIT ?
                """,
                [capped],
            ).fetchall()
    except duckdb.CatalogException as exc:
        raise HTTPException(
            status_code=404, detail="market_agg_state table not available"
        ) from exc
    except Exception as exc:
        LOGGER.exception("Failed to read market_agg_state history")
        raise HTTPException(
            status_code=500, detail="Failed to read market aggregation history"
        ) from exc

    keys = [
        "received_at",
        "date",
        "call_premium",
        "put_premium",
        "call_premium_otm_only",
        "put_premium_otm_only",
        "delta",
        "gamma",
        "theta",
        "vega",
    ]
    return {
        "limit": capped,
        "rows": [dict(zip(keys, row)) for row in rows],
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


@app.websocket("/ws/gex")
async def gex_monitor_websocket(websocket: WebSocket, symbol: str = "NQ_NDX") -> None:
    """Stream full GEX snapshots via Redis pubsub for the GEX monitor page."""
    await websocket.accept()
    normalized = (symbol or "NQ_NDX").upper()
    redis_conn = _get_redis_client()
    pubsub = redis_conn.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(SNAPSHOT_PUBSUB_CHANNEL)

    GEX_FIELDS = (
        "symbol", "timestamp", "spot", "zero_gamma",
        "net_gex", "net_gex_oi", "sum_gex_vol", "sum_gex_oi",
        "major_pos_vol", "major_neg_vol", "major_pos_oi", "major_neg_oi",
        "delta_risk_reversal", "maxchange", "max_priors",
    )

    # Map primary symbol to a secondary symbol whose net_gex we display
    CROSS_GEX_MAP = {"NQ_NDX": "SPX", "SPX": "NQ_NDX"}

    def _summarize_wall(major_strike, strikes, prefer_positive):
        """Compute wall ladder: major + up to 2 next candidates with %."""
        filtered = []
        for s, g in strikes:
            if prefer_positive and g <= 0:
                continue
            if not prefer_positive and g >= 0:
                continue
            filtered.append((s, g))
        if not filtered:
            return None
        filtered.sort(key=lambda p: abs(p[1]), reverse=True)
        major = None
        if isinstance(major_strike, (int, float)):
            for s, g in filtered:
                if abs(s - major_strike) <= 0.51:
                    major = (s, g)
                    break
        if major is None:
            major = filtered[0]
        entries = []
        for s, g in filtered:
            if abs(s - major[0]) <= 0.51:
                continue
            pct = (abs(g) / abs(major[1]) * 100) if major[1] else None
            entries.append({"strike": s, "pct": pct})
            if len(entries) >= 2:
                break
        return {"major": major[0], "next": entries}

    def _flatten_wall_fields(out: Dict[str, Any], side: str, ladder: Dict[str, Any]) -> None:
        prefix = f"{side}_wall"
        out[f"{prefix}_major_strike"] = ladder.get("major")
        entries = ladder.get("next") or []
        for idx in range(2):
            entry = entries[idx] if idx < len(entries) and isinstance(entries[idx], dict) else {}
            n = idx + 1
            out[f"{prefix}_candidate{n}_strike"] = entry.get("strike")
            out[f"{prefix}_candidate{n}_pct"] = entry.get("pct")

    def _parse_strikes(raw):
        """Normalize strikes list into [(strike, gamma), ...]."""
        if not isinstance(raw, list):
            return []
        result = []
        for entry in raw:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                try:
                    gamma = float(entry[1])
                    if gamma == 0.0 and len(entry) >= 3:
                        gamma = float(entry[2])
                    result.append((float(entry[0]), gamma))
                except (TypeError, ValueError):
                    pass
        return result

    def _extract(payload: Dict[str, Any]) -> Dict[str, Any]:
        out = {k: payload.get(k) for k in GEX_FIELDS}
        out["symbol"] = normalized
        # Wall ladders from strikes data
        strikes = _parse_strikes(payload.get("strikes"))
        if strikes:
            call_ladder = _summarize_wall(payload.get("major_pos_vol"), strikes, True)
            put_ladder = _summarize_wall(payload.get("major_neg_vol"), strikes, False)
            if call_ladder:
                out["call_wall_ladder"] = call_ladder
                _flatten_wall_fields(out, "call", call_ladder)
            if put_ladder:
                out["put_wall_ladder"] = put_ladder
                _flatten_wall_fields(out, "put", put_ladder)
            flat = {k: v for k, v in out.items() if k.startswith(("call_wall_", "put_wall_"))}
            if flat:
                try:
                    key = f"{SNAPSHOT_KEY_PREFIX}{normalized}"
                    enriched = dict(payload)
                    enriched.update(flat)
                    redis_conn.setex(key, 300, json.dumps(enriched, default=str))
                except Exception:
                    pass
        # Cross-symbol net GEX (e.g. ES_SPX net_gex when viewing NQ_NDX)
        cross_sym = CROSS_GEX_MAP.get(normalized)
        if cross_sym:
            try:
                cross_key = f"{SNAPSHOT_KEY_PREFIX}{cross_sym}"
                cross_raw = redis_conn.get(cross_key)
                if cross_raw:
                    cross_snap = json.loads(cross_raw if isinstance(cross_raw, str) else cross_raw.decode("utf-8"))
                    out["cross_symbol"] = cross_sym
                    out["cross_net_gex"] = cross_snap.get("net_gex")
            except Exception:
                pass
        return out

    # Send the latest cached snapshot immediately
    try:
        key = f"{SNAPSHOT_KEY_PREFIX}{normalized}"
        raw = redis_conn.get(key)
        if raw:
            try:
                snapshot = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
            except Exception:
                snapshot = None
            if isinstance(snapshot, dict):
                await websocket.send_json(_extract(snapshot))
    except Exception:
        LOGGER.debug("Failed to send initial /ws/gex snapshot", exc_info=True)

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
                continue
            if (payload.get("symbol") or "").upper() != normalized:
                continue
            await websocket.send_json(_extract(payload))
    except WebSocketDisconnect:
        return
    finally:
        try:
            pubsub.close()
        except Exception:
            pass


@app.websocket("/ws/social")
async def social_events_websocket(websocket: WebSocket) -> None:
    """Stream social media events (Truth Social, Twitter, News) to the GEX monitor."""
    await websocket.accept()
    redis_conn = _get_redis_client()

    # Backfill recent history so the feed isn't empty on connect
    try:
        history = redis_conn.lrange(SOCIAL_HISTORY_KEY, 0, 199)
        for item in reversed(history or []):
            try:
                if isinstance(item, (bytes, bytearray)):
                    payload = json.loads(item.decode("utf-8"))
                else:
                    payload = json.loads(item)
                await websocket.send_json(payload)
            except Exception:
                continue
    except Exception:
        pass

    pubsub = redis_conn.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(SOCIAL_ALL_EVENTS_CHANNEL)

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
                continue
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        return
    finally:
        try:
            pubsub.close()
        except Exception:
            pass


CORRELATION_ALERT_CHANNEL = "correlation:alerts:stream"
CORRELATION_HISTORY_KEY = "correlation:alerts:history"


@app.websocket("/ws/correlation")
async def correlation_alerts_websocket(websocket: WebSocket) -> None:
    """Stream correlation alerts (social event + market signal coincidences) to the GEX monitor."""
    await websocket.accept()
    redis_conn = _get_redis_client()

    # Backfill today's alerts so the banner repopulates on reconnect
    try:
        history = redis_conn.lrange(CORRELATION_HISTORY_KEY, 0, 49)
        for item in reversed(history or []):
            try:
                payload = json.loads(item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else item)
                await websocket.send_json(payload)
            except Exception:
                continue
    except Exception:
        pass

    pubsub = redis_conn.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(CORRELATION_ALERT_CHANNEL)

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
                continue
            # Persist to history (keep last 50, today only via TTL)
            try:
                serialized = json.dumps(payload, default=str)
                redis_conn.lpush(CORRELATION_HISTORY_KEY, serialized)
                redis_conn.ltrim(CORRELATION_HISTORY_KEY, 0, 49)
                redis_conn.expire(CORRELATION_HISTORY_KEY, 86400)  # expire at end of day
            except Exception:
                pass
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        return
    finally:
        try:
            pubsub.close()
        except Exception:
            pass


@app.websocket("/ws/market-sentiment")
async def market_sentiment_websocket(websocket: WebSocket) -> None:
    """Stream market sentiment data (put/call ratio, regime, premiums) to the GEX monitor."""
    await websocket.accept()
    redis_conn = _get_redis_client()

    # Send latest cached market agg data immediately on connect
    try:
        raw = redis_conn.get("uw:market_agg:latest")
        if raw:
            latest = _loads_json_payload(raw)
            await websocket.send_json(_market_agg_snapshot_from_cache(latest))
    except Exception:
        LOGGER.debug("Failed to send initial market sentiment snapshot", exc_info=True)

    # Subscribe to both market agg updates and alert channel
    pubsub = redis_conn.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe("uw:market_agg:stream", "market_agg:alerts")

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
                continue

            channel = message.get("channel")
            if isinstance(channel, bytes):
                channel = channel.decode("utf-8")

            if channel == "uw:market_agg:stream":
                await websocket.send_json(_market_agg_snapshot_from_cache(payload))
            elif channel == "market_agg:alerts":
                await websocket.send_json({
                    "type": "sentiment_alert",
                    "alert_type": payload.get("alert_type"),
                    "current_ratio": payload.get("current_ratio"),
                    "previous_ratio": payload.get("previous_ratio"),
                    "from_regime": payload.get("from_regime"),
                    "to_regime": payload.get("to_regime"),
                    "direction": payload.get("direction"),
                    "change_pct": payload.get("change_pct"),
                })
    except WebSocketDisconnect:
        return
    finally:
        try:
            pubsub.close()
        except Exception:
            pass


@app.websocket("/ws/sweep")
async def sweep_intelligence_websocket(websocket: WebSocket, symbol: str = "MNQ") -> None:
    """Stream sweep classifier alerts and position monitor events for the intelligence dashboard.

    Subscribes to:
        sweep:alert:{symbol}   — SweepAlert payloads from SweepClassifierService
        sweep:monitor:{symbol} — PositionMonitorService level updates + TP signals
        market:dom:{symbol}    — DOM snapshots (price/imbalance heat data)
        market:cvd:{symbol}    — CVD rolling windows

    Each message sent to the client has a ``type`` field:
        sweep_alert    — classification result (sweep/directional + confidence)
        monitor_event  — danger level escalation or TP wall proximity signal
        dom_snapshot   — DOM depth/imbalance update (throttled to 2 Hz)
        cvd_snapshot   — CVD rolling windows update (throttled to 1 Hz)
    """
    await websocket.accept()
    sym = (symbol or "MNQ").upper()
    redis_conn = _get_redis_client()
    pubsub = redis_conn.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(
        f"sweep:alert:{sym}",
        f"sweep:monitor:{sym}",
        f"market:dom:{sym}",
        f"market:cvd:{sym}",
    )

    # Channel → message type mapping
    _CHANNEL_TYPE = {
        f"sweep:alert:{sym}":   "sweep_alert",
        f"sweep:monitor:{sym}": "monitor_event",
        f"market:dom:{sym}":    "dom_snapshot",
        f"market:cvd:{sym}":    "cvd_snapshot",
    }

    # Throttle DOM/CVD to avoid overwhelming the browser
    _THROTTLE_INTERVAL = {
        "dom_snapshot": 0.5,   # 2 Hz
        "cvd_snapshot": 1.0,   # 1 Hz
    }
    _last_sent: dict[str, float] = {}

    # Send cached latest DOM/CVD on connect so the page isn't blank
    try:
        for ch in (f"market:dom:{sym}", f"market:cvd:{sym}"):
            raw = redis_conn.get(ch + ":latest")
            if raw:
                try:
                    payload = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
                    msg_type = _CHANNEL_TYPE.get(ch, "dom_snapshot")
                    await websocket.send_json({"type": msg_type, **payload})
                except Exception:
                    pass
    except Exception:
        pass

    try:
        while True:
            try:
                message = await asyncio.to_thread(pubsub.get_message, timeout=1.5)
            except Exception:
                message = None
            if not message or message.get("type") != "message":
                await asyncio.sleep(0.05)
                continue

            channel = message.get("channel", "")
            if isinstance(channel, bytes):
                channel = channel.decode("utf-8")

            msg_type = _CHANNEL_TYPE.get(channel)
            if not msg_type:
                continue

            # Throttle high-frequency streams
            throttle = _THROTTLE_INTERVAL.get(msg_type)
            if throttle:
                now = asyncio.get_event_loop().time()
                if now - _last_sent.get(msg_type, 0) < throttle:
                    continue
                _last_sent[msg_type] = now

            raw = message.get("data")
            try:
                if isinstance(raw, (bytes, bytearray)):
                    payload = json.loads(raw.decode("utf-8"))
                elif isinstance(raw, str):
                    payload = json.loads(raw)
                else:
                    continue
            except Exception:
                continue

            await websocket.send_json({"type": msg_type, **payload})
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


def _normalize_string(value: Optional[Any]) -> str:
    """Trim and stringify optionally missing payload values."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()



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
    return int(datetime.now(timezone.utc).timestamp() * 1000)


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

    # Ensure a console handler with timestamps is always present
    has_console = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, RotatingFileHandler)
        for h in root.handlers
    )
    if not has_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(fmt)
        root.addHandler(ch)
    else:
        # Apply formatter to existing console handlers missing one
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler):
                h.setFormatter(fmt)

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
        "access_log": False,  # Disable HTTP access logging
    }
    
    # Add SSL support if certificates provided
    if args.ssl_certfile and args.ssl_keyfile:
        uvicorn_config["ssl_certfile"] = args.ssl_certfile
        uvicorn_config["ssl_keyfile"] = args.ssl_keyfile
        LOGGER.info(
            "SSL enabled: cert=%s, key=%s", args.ssl_certfile, args.ssl_keyfile
        )
    
    try:
        uvicorn.run(**uvicorn_config)
    except (KeyboardInterrupt, RecursionError):
        # Python 3.13 can hit RecursionError when _cancel_all_tasks walks
        # deeply nested gather/child-task chains (e.g. tastytrade DXLinkStreamer).
        # The signal handler in lifespan already stopped services; just exit.
        pass


if __name__ == "__main__":
    main()
