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

from ..config import settings
from ..lib.logging import get_logger
from ..lib.redis_client import RedisClient
from ..models.market_data import Level2Event, Level2Quote, TickEvent
from .trading_publisher import TradingEventPublisher

LOG = get_logger(__name__)


@dataclass
class SchwabToken:
    """Holds OAuth access token metadata."""

    access_token: str
    expires_at: datetime

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() >= self.expires_at - timedelta(seconds=30)


class SchwabAuthClient:
    """Handles Schwab OAuth token management."""

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
        self._token: Optional[SchwabToken] = None

    def get_token(self) -> SchwabToken:
        """Return valid access token, refreshing when required."""
        if self._token and not self._token.is_expired:
            return self._token
        self._token = self._fetch_token()
        return self._token

    def _fetch_token(self) -> SchwabToken:
        token_endpoint = f"{self.rest_url}/oauth/token"
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        LOG.info("Requesting Schwab OAuth token")
        resp = httpx.post(token_endpoint, data=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        expires_in = int(data.get("expires_in", 1800))
        token = SchwabToken(
            access_token=data["access_token"],
            expires_at=datetime.utcnow() + timedelta(seconds=expires_in),
        )
        LOG.info("Obtained Schwab token expiring at %s", token.expires_at.isoformat())
        return token


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
    """Maintains websocket connection to Schwab streaming API."""

    def __init__(
        self,
        auth_client: SchwabAuthClient,
        publisher: TradingEventPublisher,
        stream_url: str,
        symbols: list[str],
        heartbeat_seconds: int = 15,
    ):
        self.auth_client = auth_client
        self.publisher = publisher
        self.stream_url = stream_url
        self.symbols = symbols
        self.heartbeat_seconds = heartbeat_seconds
        self.parser = SchwabMessageParser()
        self._ws: Optional[WebSocketApp] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start streaming loop (blocking)."""
        LOG.info("Starting Schwab streamer for symbols: %s", ",".join(self.symbols))
        while not self._stop_event.is_set():
            token = self.auth_client.get_token()
            headers = {"Authorization": f"Bearer {token.access_token}"}
            self._run_socket(headers)
            if not self._stop_event.is_set():
                LOG.warning("Streaming connection closed; retrying in 5s")
                time.sleep(5)

    def stop(self) -> None:
        """Signal streaming loop to stop."""
        self._stop_event.set()
        if self._ws:
            self._ws.close()

    def _run_socket(self, headers: dict) -> None:
        self._ws = WebSocketApp(
            self.stream_url,
            header=[f"{k}: {v}" for k, v in headers.items()],
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        thread.start()
        heartbeat = self.heartbeat_seconds
        while thread.is_alive() and not self._stop_event.is_set():
            time.sleep(heartbeat)
            if self._ws:
                try:
                    self._ws.send(json.dumps({"action": "heartbeat"}))
                except Exception:
                    LOG.warning("Failed to send heartbeat; closing socket")
                    self._ws.close()
                    break

    def _on_open(self, ws: WebSocketApp) -> None:  # pragma: no cover - network I/O
        LOG.info("Connected to Schwab stream; subscribing to symbols")
        subscription = {
            "action": "subscribe",
            "symbols": self.symbols,
            "channels": ["tick", "level2"],
        }
        ws.send(json.dumps(subscription))

    def _on_message(self, ws: WebSocketApp, message: str) -> None:  # pragma: no cover
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            LOG.debug("Ignoring non-JSON message: %s", message)
            return
        for kind, event in self.parser.parse(payload):
            if kind == "tick":
                self.publisher.publish_tick(event)  # type: ignore[arg-type]
            elif kind == "level2":
                self.publisher.publish_level2(event)  # type: ignore[arg-type]

    def _on_error(self, ws: WebSocketApp, error: Exception) -> None:  # pragma: no cover
        LOG.error("Schwab stream error: %s", error)

    def _on_close(self, ws: WebSocketApp, *_args) -> None:  # pragma: no cover
        LOG.info("Schwab stream closed")


def build_streamer(
    redis_client: Optional[RedisClient] = None,
    publisher_factory: Optional[Callable[[RedisClient], TradingEventPublisher]] = None,
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
    )
