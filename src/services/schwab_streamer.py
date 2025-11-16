"""Schwab streaming client that forwards tick + level2 data into the trading system."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

import httpx
from websocket import WebSocketApp
from schwab.client import Client as Schwab
from schwab import auth

from ..config import settings
from ..lib.logging import get_logger
from ..lib.redis_client import RedisClient
from ..models.market_data import Level2Event, Level2Quote, TickEvent
from .trading_publisher import TradingEventPublisher

import os
import json
import base64
from pathlib import Path
from threading import Lock

LOG = get_logger(__name__)


@dataclass
class SchwabToken:
    """Holds OAuth access token metadata."""

    access_token: str
    expires_at: datetime
    refresh_token: str  # Add refresh_token to update it

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() >= self.expires_at - timedelta(seconds=30)


class SchwabAuthClient:
    """Handles Schwab OAuth token management with auto-refresh using schwab-py."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        rest_url: str,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.rest_url = rest_url.rstrip("/")
        self._stop_refresh = threading.Event()
        self._refresh_thread: Optional[threading.Thread] = None
        self._lock = Lock()

        # token persistence path: project-root/.tokens/schwab_token.json
        project_root = Path(__file__).resolve().parents[2]
        self._tok_path = project_root / ".tokens" / "schwab_token.json"
        
        # Initialize schwab-py client
        self.schwab = auth.easy_client(
            api_key=self.client_id,
            app_secret=self.client_secret,
            callback_url='http://127.0.0.1:8182',  # Required for login flow
            token_path=str(self._tok_path)
        )
        
        # If we have a refresh token, set it in the schwab client
        if self.refresh_token:
            # Load or set the token
            try:
                tokens = self._load_persisted_tokens()
                if tokens:
                    self.schwab.tokens = tokens
                else:
                    # Set initial tokens
                    self.schwab.tokens = {
                        'access_token': '',
                        'refresh_token': self.refresh_token,
                        'token_type': 'Bearer',
                        'expires_in': 0
                    }
            except Exception:
                pass

    def start_auto_refresh(self) -> None:
        """Start background thread for automatic token refresh."""
        if self._refresh_thread and self._refresh_thread.is_alive():
            LOG.info("Auto-refresh already running")
            return
        self._stop_refresh.clear()
        self._refresh_thread = threading.Thread(target=self._auto_refresh_loop, daemon=True)
        self._refresh_thread.start()
        LOG.info("Started Schwab token auto-refresh")

    def stop_auto_refresh(self) -> None:
        """Stop the auto-refresh thread."""
        self._stop_refresh.set()
        if self._refresh_thread:
            self._refresh_thread.join(timeout=5)
        LOG.info("Stopped Schwab token auto-refresh")

    def refresh_tokens(self) -> Optional[dict]:
        """Manually refresh tokens using schwab-py."""
        LOG.info("Manually refreshing Schwab tokens...")
        try:
            with self._lock:
                self.schwab.refresh_token()
                tokens = self.schwab.tokens
                LOG.info("Tokens refreshed successfully")
                return tokens
        except Exception as e:
            LOG.error("Manual token refresh failed: %s", e)
            return None

    def get_token(self) -> SchwabToken:
        """Return valid access token, refreshing if needed."""
        with self._lock:
            # schwab-py handles refresh automatically when accessing access_token
            access_token = self.schwab.access_token
            tokens = self.schwab.tokens
            expires_in = tokens.get('expires_in', 1800)
            expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
            refresh_token = tokens.get('refresh_token', self.refresh_token)
            return SchwabToken(
                access_token=access_token,
                expires_at=expires_at,
                refresh_token=refresh_token
            )

    def _auto_refresh_loop(self) -> None:
        """Background loop to refresh tokens periodically."""
        ACCESS_REFRESH_INTERVAL = 29 * 60  # 29 minutes
        while not self._stop_refresh.is_set():
            try:
                with self._lock:
                    self.schwab.refresh_token()
            except Exception as e:
                LOG.error("Auto-refresh failed: %s", e)
            # Sleep in chunks to be responsive to stop
            slept = 0
            while slept < ACCESS_REFRESH_INTERVAL and not self._stop_refresh.is_set():
                chunk = min(10, ACCESS_REFRESH_INTERVAL - slept)
                time.sleep(chunk)
                slept += chunk


class SchwabMessageParser:
    """Convert Schwab payloads into internal models."""

    def parse(self, payload: dict) -> list[tuple[str, object]]:
        """Return list of (kind, event) tuples."""
        events: list[tuple[str, object]] = []
        msg_type = (payload.get("type") or payload.get("topic") or "").lower()
        if msg_type in {"tick", "trade"}:
            event = TickEvent(
                symbol=payload.get("symbol", payload.get("ticker", "")).upper(),
                timestamp=self._coerce_ts(payload.get("timestamp")),
                last=self._safe_float(payload.get("price") or payload.get("last")),
                bid=self._safe_float(payload.get("bid")),
                ask=self._safe_float(payload.get("ask")),
                volume=self._safe_float(payload.get("volume")),
            )
            events.append(("tick", event))
        elif msg_type in {"level2", "book"}:
            bids = [
                Level2Quote(price=self._safe_float(level.get("price", 0.0)), size=self._safe_float(level.get("size", 0.0)))
                for level in payload.get("bids", [])
            ]
            asks = [
                Level2Quote(price=self._safe_float(level.get("price", 0.0)), size=self._safe_float(level.get("size", 0.0)))
                for level in payload.get("asks", [])
            ]
            event = Level2Event(
                symbol=payload.get("symbol", payload.get("ticker", "")).upper(),
                timestamp=self._coerce_ts(payload.get("timestamp")),
                bids=bids,
                asks=asks,
            )
            events.append(("level2", event))
        return events

    @staticmethod
    def _coerce_ts(value) -> datetime:
        if isinstance(value, (int, float)):
            return datetime.utcfromtimestamp(float(value) / (1000 if value > 1e12 else 1))
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.utcnow()

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None


class SchwabStreamClient:
    """Maintains websocket connection to Schwab streaming API using schwab-py."""

    def __init__(
        self,
        auth_client: SchwabAuthClient,
        publisher: TradingEventPublisher,
        stream_url: str,
        symbols: list[str],
        heartbeat_seconds: int = 15,
        tick_handler: Optional[Callable[[TickEvent], None]] = None,
        level2_handler: Optional[Callable[[Level2Event], None]] = None,
    ):
        self.auth_client = auth_client
        self.publisher = publisher
        self.stream_url = stream_url
        self.symbols = symbols
        self.heartbeat_seconds = heartbeat_seconds
        self.parser = SchwabMessageParser()
        self._tick_handler = tick_handler
        self._level2_handler = level2_handler
        self.stream = self.auth_client.schwab.stream()

    def start(self) -> None:
        """Start streaming loop (blocking)."""
        LOG.info("Starting Schwab streamer for symbols: %s", ",".join(self.symbols))
        self.auth_client.start_auto_refresh()  # Start background token refresh
        self.stream.start(
            symbols=self.symbols,
            on_tick=self._on_tick,
            on_level2=self._on_level2
        )

    def stop(self) -> None:
        """Signal streaming loop to stop."""
        self.stream.stop()
        self.auth_client.stop_auto_refresh()  # Stop background token refresh

    def refresh_tokens(self) -> Optional[dict]:
        """Manually refresh tokens via the auth client."""
        return self.auth_client.refresh_tokens()

    def _on_tick(self, payload: dict) -> None:
        """Handle tick event from schwab-py stream."""
        for kind, event in self.parser.parse(payload):
            if kind == "tick":
                self.publisher.publish_tick(event)  # type: ignore[arg-type]
                if self._tick_handler:
                    try:
                        self._tick_handler(event)
                    except Exception as exc:  # pragma: no cover - defensive
                        LOG.warning("Tick handler error: %s", exc)

    def _on_level2(self, payload: dict) -> None:
        """Handle level2 event from schwab-py stream."""
        for kind, event in self.parser.parse(payload):
            if kind == "level2":
                self.publisher.publish_level2(event)  # type: ignore[arg-type]
                if self._level2_handler:
                    try:
                        self._level2_handler(event)
                    except Exception as exc:  # pragma: no cover - defensive
                        LOG.warning("Level2 handler error: %s", exc)


def build_streamer(
    redis_client: Optional[RedisClient] = None,
    publisher_factory: Optional[Callable[[RedisClient], TradingEventPublisher]] = None,
    tick_handler: Optional[Callable[[TickEvent], None]] = None,
    level2_handler: Optional[Callable[[Level2Event], None]] = None,
) -> SchwabStreamClient:
    """Factory to wire dependencies from settings."""
    if not settings.schwab_client_id or not settings.schwab_client_secret or not settings.schwab_refresh_token:
        raise RuntimeError("SCHWAB_* credentials must be configured to enable streaming")
    redis_client = redis_client or RedisClient()
    publisher_factory = publisher_factory or (lambda rc: TradingEventPublisher(
        redis_client=rc,
        tick_channel=settings.schwab_tick_channel,
        level2_channel=settings.schwab_level2_channel,
    ))
    publisher = publisher_factory(redis_client)
    auth_client = SchwabAuthClient(
        client_id=settings.schwab_client_id,
        client_secret=settings.schwab_client_secret,
        refresh_token=settings.schwab_refresh_token,
        rest_url=settings.schwab_rest_url,
    )
    return SchwabStreamClient(
        auth_client=auth_client,
        publisher=publisher,
        stream_url=settings.schwab_stream_url,
        symbols=settings.schwab_symbol_list,
        heartbeat_seconds=settings.schwab_heartbeat_seconds,
        tick_handler=tick_handler,
        level2_handler=level2_handler,
    )
