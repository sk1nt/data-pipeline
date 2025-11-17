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
from schwab import streaming as schwab_streaming
import asyncio

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
        interactive: bool = True,
        schwab_client: Optional[auth.easy_client] = None,
        access_refresh_interval_seconds: int = 29 * 60,
        refresh_token_rotate_interval_seconds: int = 6 * 24 * 60 * 60,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.rest_url = rest_url.rstrip("/")
        self._stop_refresh = threading.Event()
        self._refresh_thread: Optional[threading.Thread] = None
        self._lock = Lock()
        self._access_refresh_interval = access_refresh_interval_seconds
        self._refresh_token_rotate_interval = refresh_token_rotate_interval_seconds
        self._last_refresh_token_rotate = datetime.utcnow()

        # token persistence path: project-root/.tokens/schwab_token.json
        project_root = Path(__file__).resolve().parents[2]
        self._tok_path = project_root / ".tokens" / "schwab_token.json"
        
        # Initialize schwab-py client (or use provided one for testability)
        if schwab_client is not None:
            self.schwab = schwab_client
        else:
            # Prefer redirect URI configured in settings; default to https
            callback = settings.schwab_redirect_uri or "https://127.0.0.1:8182"
            try:
                self.schwab = auth.easy_client(
                    api_key=self.client_id,
                    app_secret=self.client_secret,
                    callback_url=callback,
                    token_path=str(self._tok_path),
                    interactive=interactive,
                )
            except ValueError as e:
                # Some older token files are stored in a legacy format and will
                # cause the schwab lib to raise ValueError. Attempt to rename
                # the legacy token file and retry creation without token_path
                # to allow a fresh login flow (this is a best-effort fix).
                LOG.warning("Schwab easy_client failed to load token file, attempting to rotate legacy token: %s", e)
                try:
                    if self._tok_path.exists():
                        legacy = self._tok_path.with_suffix(".old")
                        os.replace(self._tok_path, legacy)
                        LOG.info("Renamed legacy token file to %s", legacy)
                    # Some older installs use `schwab.token` in the token dir; rotate it too
                    alt_tok = self._tok_path.with_name("schwab.token")
                    if alt_tok.exists():
                        alt_legacy = alt_tok.with_suffix(".old")
                        os.replace(alt_tok, alt_legacy)
                        LOG.info("Renamed legacy token file to %s", alt_legacy)
                except Exception as rename_exc:  # pragma: no cover - defensive
                    LOG.debug("Failed to rename legacy token file: %s", rename_exc)
                # Retry without passing token_path (client will manage it)
                self.schwab = auth.easy_client(
                    api_key=self.client_id,
                    app_secret=self.client_secret,
                    callback_url=callback,
                    token_path=str(self._tok_path),
                    interactive=interactive,
                )
        
        # If we have a refresh token, set it in the schwab client
        if self.refresh_token:
            # Load or set the token
            try:
                tokens = self._load_persisted_tokens()
                if tokens:
                    self.schwab.tokens = tokens
                else:
                    # Attempt to load tokens from environment variables
                    env_tokens = self._load_tokens_from_env()
                    if env_tokens:
                        self.schwab.tokens = env_tokens
                    else:
                        # Set initial tokens if only refresh token is present.
                        # Do not persist an empty access token to avoid creating
                        # token files when none were provided.
                        self.schwab.tokens = {
                            'access_token': os.environ.get('SCHWAB_ACCESS_TOKEN', ''),
                            'refresh_token': self.refresh_token,
                            'token_type': 'Bearer',
                            'expires_in': int(os.environ.get('SCHWAB_EXPIRES_IN', '0')),
                        }
            except Exception:
                pass

    def _load_tokens_from_env(self) -> Optional[dict]:
        """Build token dict from environment variables if present.

        This supports CI-friendly runs and avoids requiring a browser when
        `SCHWAB_REFRESH_TOKEN` (or `SCHWAB_TOKENS_JSON`) is provided in env.
        """
        # First, allow a JSON-encoded token to be supplied via env
        json_tok = os.environ.get('SCHWAB_TOKENS_JSON')
        if json_tok:
            try:
                parsed = json.loads(json_tok)
                # If keys are nested under `tokens` (the exchange script format), unwrap
                if isinstance(parsed, dict) and 'tokens' in parsed:
                    parsed = parsed['tokens']
                if isinstance(parsed, dict) and parsed.get('refresh_token'):
                    # persist for future runs
                    try:
                        self._persist_tokens(parsed)
                    except Exception:
                        pass
                    return parsed
            except Exception:
                LOG.debug("SCHWAB_TOKENS_JSON present but failed to parse; ignoring")

        # Fallback: use access/refresh tokens as separate env variables
        rt = os.environ.get('SCHWAB_REFRESH_TOKEN') or os.environ.get('SCHWAB_RTOKEN') or None
        at = os.environ.get('SCHWAB_ACCESS_TOKEN') or os.environ.get('SCHWAB_ATOKEN') or None
        if rt or at:
            tokens = {
                'access_token': at or '',
                'refresh_token': rt or '',
                'token_type': 'Bearer',
                'expires_in': int(os.environ.get('SCHWAB_EXPIRES_IN', '0')),
            }
            try:
                self._persist_tokens(tokens)
            except Exception:
                pass
            return tokens

    def _load_persisted_tokens(self) -> Optional[dict]:
        """Load persisted tokens from persistent token path if present."""
        try:
            if self._tok_path.exists():
                with open(self._tok_path, "r") as fh:
                    return json.load(fh)
        except Exception as e:
            LOG.debug("Failed to read persisted tokens: %s", e)
        return None

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
                try:
                    if isinstance(tokens, dict):
                        self._persist_tokens(tokens)
                except Exception as e:  # pragma: no cover - defensive
                    LOG.debug("Failed to persist tokens after manual refresh: %s", e)
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
        """Background loop to refresh tokens periodically and rotate refresh token less frequently."""
        while not self._stop_refresh.is_set():
            now = datetime.utcnow()
            try:
                with self._lock:
                    # Refresh access token
                    self.schwab.refresh_token()
                    # Immediately persist tokens for safety
                    tokens = getattr(self.schwab, "tokens", None)
                    if isinstance(tokens, dict):
                        try:
                            self._persist_tokens(tokens)
                        except Exception as e:  # pragma: no cover - defensive
                            LOG.debug("Failed to persist tokens after auto refresh: %s", e)
                    # Rotate refresh token on longer schedule
                    if (now - self._last_refresh_token_rotate).total_seconds() >= self._refresh_token_rotate_interval:
                        LOG.info("Rotating refresh token (scheduled interval reached)")
                        self.schwab.refresh_token()
                        # Persist again after rotation
                        tokens = getattr(self.schwab, "tokens", None)
                        if isinstance(tokens, dict):
                            try:
                                self._persist_tokens(tokens)
                            except Exception as e:  # pragma: no cover - defensive
                                LOG.debug("Failed to persist tokens after rotation: %s", e)
                        self._last_refresh_token_rotate = now
            except Exception as e:
                LOG.error("Auto-refresh failed: %s", e)
            # Sleep in chunks to be responsive to stop
            slept = 0
            while slept < self._access_refresh_interval and not self._stop_refresh.is_set():
                chunk = min(10, self._access_refresh_interval - slept)
                time.sleep(chunk)
                slept += chunk

    def refresh_refresh_token(self) -> Optional[dict]:
        """Force a refresh_token rotation and update rotation timestamp."""
        LOG.info("Forcing rotation of Schwab refresh token")
        try:
            with self._lock:
                self.schwab.refresh_token()
                tokens = getattr(self.schwab, "tokens", None)
                if isinstance(tokens, dict):
                    try:
                        self._persist_tokens(tokens)
                    except Exception:  # pragma: no cover - defensive
                        pass
                self._last_refresh_token_rotate = datetime.utcnow()
                return self.schwab.tokens
        except Exception as e:  # pragma: no cover - defensive
            LOG.error("Forced refresh token rotation failed: %s", e)
            return None

    def _persist_tokens(self, tokens: dict) -> None:
        """Persist tokens to disk atomically at the configured token path."""
        try:
            parent = self._tok_path.parent
            parent.mkdir(parents=True, exist_ok=True)
            tmp = self._tok_path.with_suffix(".tmp")
            with open(tmp, "w") as fh:
                json.dump(tokens, fh)
            os.replace(tmp, self._tok_path)
            LOG.debug("Persisted Schwab tokens to %s", self._tok_path)
        except Exception as e:  # pragma: no cover - defensive
            LOG.error("Failed to persist Schwab tokens: %s", e)

    def stream(self):
        """Return a synchronous stream adapter that wraps schwab-py's StreamClient.
        The wrapper provides `start(symbols, on_tick, on_level2)` and `stop()`
        methods to match existing expectations used elsewhere in the codebase.
        """
        try:
            sc = schwab_streaming.StreamClient(self.schwab)
            return _SchwabStreamAdapter(sc)
        except Exception as e:
            LOG.error("Failed to construct StreamClient adapter: %s", e)
            raise


class _SchwabStreamAdapter:
    """Synchronous adapter over `schwab.streaming.StreamClient`.
    Implements `start(symbols, on_tick, on_level2)` and `stop()`.
    """
    def __init__(self, stream_client: schwab_streaming.StreamClient):
        self._sc = stream_client
        self._running = False
        self._symbols = []

    def start(self, symbols=None, on_tick=None, on_level2=None):
        symbols = symbols or []
        self._symbols = symbols
        # Register handlers that forward raw payload to provided handlers
        if on_tick:
            # Attach to both equity and futures level-one handlers as appropriate
            self._sc.add_level_one_futures_handler(lambda msg: on_tick(msg))
            self._sc.add_level_one_equity_handler(lambda msg: on_tick(msg))
        if on_level2:
            # Attach book handlers
            self._sc.add_nasdaq_book_handler(lambda msg: on_level2(msg))
            self._sc.add_nyse_book_handler(lambda msg: on_level2(msg))

        # Login and subscribe for symbols
        asyncio.run(self._sc.login())
        # Subscribe to symbol types
        # Map to futures vs equities
        futures_symbols = set([s for s in symbols if s.upper() in {"MES", "MNQ", "NQ"}])
        equity_symbols = [s for s in symbols if s.upper() not in futures_symbols]
        if futures_symbols:
            asyncio.run(self._sc.level_one_futures_add(list(futures_symbols)))
        if equity_symbols:
            asyncio.run(self._sc.level_one_equity_add(equity_symbols))
            asyncio.run(self._sc.nasdaq_book_add(equity_symbols))
            asyncio.run(self._sc.nyse_book_add(equity_symbols))
        self._running = True

    def stop(self):
        if self._running:
            try:
                asyncio.run(self._sc.logout())
            except Exception:
                pass
            self._running = False


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
        # Expect the auth_client to provide a `stream()` method returning a
        # streaming wrapper with `start()`/`stop()` methods. The `schwab-py`
        # Client itself doesn't provide a `stream()` member.
        self.stream = self.auth_client.stream()

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
    use_tastytrade_symbols: bool = False,
    interactive: bool = True,
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
        interactive=interactive,
    )
    symbols = settings.tastytrade_symbol_list if use_tastytrade_symbols else settings.schwab_symbol_list
    return SchwabStreamClient(
        auth_client=auth_client,
        publisher=publisher,
        stream_url=settings.schwab_stream_url,
        symbols=symbols,
        heartbeat_seconds=settings.schwab_heartbeat_seconds,
        tick_handler=tick_handler,
        level2_handler=level2_handler,
    )
