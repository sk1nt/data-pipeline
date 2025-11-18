"""Schwab streaming client that forwards tick + level2 data into the trading system."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Callable, Optional

from schwab import auth
from schwab import streaming as schwab_streaming

from ..config import settings
from ..lib.logging import get_logger
from ..lib.redis_client import RedisClient
from ..models.market_data import Level2Event, Level2Quote, TickEvent
from ..token_store import MissingBootstrapTokenError, TokenStore, TokenStoreError
from .trading_publisher import TradingEventPublisher

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

        self._token_store = TokenStore()
        self._tok_path = self._token_store.tokens_path

        bootstrap_available = self._token_store.has_bootstrap_token()
        if not bootstrap_available and not interactive and schwab_client is None:
            raise MissingBootstrapTokenError(
                f"No Schwab tokens found at {self._tok_path}. Run the one-time interactive bootstrap first."
            )
        # Upgrade legacy files if present so schwab-py can load them consistently.
        try:
            self._token_store.ensure_metadata_format()
        except TokenStoreError as exc:
            LOG.error("Failed to normalize Schwab token file: %s", exc)
            raise

        # Initialize schwab-py client (or use provided one for testability)
        if schwab_client is not None:
            self.schwab = schwab_client
            self._created_schwab_client = False
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
                self._created_schwab_client = True
            except Exception as e:
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

        # After we created/initialized the client, if tokens exist make sure they
        # are persisted; this covers interactive login flows where the client
        # wrote its own token storage but also ensures we are consistent.
        # If we created the Client instance ourselves (not provided via the
        # `schwab_client` argument), ensure tokens are persisted if present.
        # This avoids tests that pass a dummy client from unexpectedly writing
        # tokens during initialization.
        try:
            tokens = getattr(self.schwab, 'tokens', None)
            if getattr(self, '_created_schwab_client', False) and isinstance(tokens, dict) and tokens.get('refresh_token'):
                self._persist_tokens(tokens)
            # Always append refresh token into .env when requested by environment,
            # even when a dummy (in tests) or externally-provided schwab client was
            # used for initialization.
            if isinstance(tokens, dict) and tokens.get('refresh_token') and os.environ.get('SCHWAB_PERSIST_REFRESH_TO_ENV') in ("1", "true", "yes"):
                self._append_refresh_to_env(tokens.get('refresh_token'))
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
        data = self._token_store._load_tokens_file()
        if data is None:
            return None
        return self._token_store._extract_token_payload(data)

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
                tokens = self._refresh_tokens_internal()
                LOG.info("Tokens refreshed successfully")
                try:
                    if isinstance(tokens, dict):
                        self._persist_tokens(tokens)
                        if os.environ.get('SCHWAB_PERSIST_REFRESH_TO_ENV') in ("1", "true", "yes"):
                            try:
                                self._append_refresh_to_env(tokens.get('refresh_token'))
                            except Exception:
                                LOG.debug('Failed to append refresh token to .env during refresh', exc_info=True)
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
                    tokens = self._refresh_tokens_internal()
                    if isinstance(tokens, dict):
                        try:
                            self._persist_tokens(tokens)
                        except Exception as e:  # pragma: no cover - defensive
                            LOG.debug("Failed to persist tokens after auto refresh: %s", e)
                        if os.environ.get('SCHWAB_PERSIST_REFRESH_TO_ENV') in ("1", "true", "yes"):
                            try:
                                self._append_refresh_to_env(tokens.get('refresh_token'))
                            except Exception:
                                LOG.debug('Failed to append refresh token to .env during auto-refresh', exc_info=True)
                    # Rotate refresh token on longer schedule
                    if (now - self._last_refresh_token_rotate).total_seconds() >= self._refresh_token_rotate_interval:
                        LOG.info("Rotating refresh token (scheduled interval reached)")
                        tokens = self._refresh_tokens_internal()
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
                tokens = self._refresh_tokens_internal()
                if isinstance(tokens, dict):
                    try:
                        self._persist_tokens(tokens)
                    except Exception:  # pragma: no cover - defensive
                        pass
                if isinstance(tokens, dict) and os.environ.get('SCHWAB_PERSIST_REFRESH_TO_ENV') in ("1", "true", "yes"):
                    try:
                        self._append_refresh_to_env(tokens.get('refresh_token'))
                    except Exception:
                        LOG.debug('Failed to append refresh token to .env during forced rotation', exc_info=True)
                self._last_refresh_token_rotate = datetime.utcnow()
                return self.schwab.tokens
        except Exception as e:  # pragma: no cover - defensive
            LOG.error("Forced refresh token rotation failed: %s", e)
            return None

    def _persist_tokens(self, tokens: dict) -> None:
        """Persist tokens to disk atomically at the configured token path."""
        try:
            if not isinstance(tokens, dict) or not tokens:
                return
            self._token_store.write_token_payload(tokens)
            LOG.debug("Persisted Schwab tokens to %s via TokenStore", self._tok_path)
        except TokenStoreError as exc:  # pragma: no cover - defensive
            LOG.error("Failed to persist Schwab tokens via TokenStore: %s", exc)
        except Exception as e:  # pragma: no cover - defensive
            LOG.error("Failed to persist Schwab tokens: %s", e)

    def _refresh_tokens_internal(self) -> dict:
        """Refresh tokens using either schwab-py helper or underlying OAuth session."""
        refresh_fn = getattr(self.schwab, "refresh_token", None)
        if callable(refresh_fn):
            refresh_fn()
            tokens = getattr(self.schwab, "tokens", None)
            if isinstance(tokens, dict):
                return tokens
        session = getattr(self.schwab, "session", None)
        if session and hasattr(session, "refresh_token"):
            token_url = settings.schwab_token_url
            refreshed = session.refresh_token(token_url)
            tokens = getattr(session, "token", None) or refreshed
            if hasattr(self.schwab, "tokens"):
                self.schwab.tokens = tokens
            metadata = getattr(self.schwab, "token_metadata", None)
            if metadata is not None and hasattr(metadata, "token"):
                metadata.token = tokens
            return tokens
        raise AttributeError("Schwab client does not expose refresh_token interfaces")

    def _append_refresh_to_env(self, refresh_token: str) -> None:
        """Append or replace SCHWAB_REFRESH_TOKEN in .env atomically.

        This helps interactive developers and CI to persist the refresh token
        in environment files for later non-interactive runs.
        """
        try:
            env_path = Path(__file__).resolve().parents[2] / '.env'
            # If .env doesn't exist, avoid creating a new file and instead write
            # the refresh token into `.env.back` to be safe for local environments
            # and CI that rely on .env contents. We still fallback to writing
            # `.env.back` in case .env is missing.
            target_path = env_path if env_path.exists() else env_path.with_suffix('.back')

            # Read current raw lines so we can preserve formatting and comments
            original_lines = []
            if target_path.exists():
                with open(target_path, 'r') as fh:
                    original_lines = fh.readlines()

            # Find if the variable exists; maintain original line if present
            key = 'SCHWAB_REFRESH_TOKEN'
            found = False
            new_lines = []
            for ln in original_lines:
                if ln.strip().startswith(f'{key}=') or ln.strip().startswith(f'{key} ='):
                    # preserve any comment after the value
                    comment = ''
                    if '#' in ln:
                        # Keep inline comments if present
                        parts = ln.split('#', 1)
                        comment = '#' + parts[1].rstrip('\n')
                    new_lines.append(f'{key}="{refresh_token}" {comment}\n')
                    found = True
                else:
                    new_lines.append(ln)

            if not found:
                # If the file had a trailing newline and not empty, preserve format
                if new_lines and not new_lines[-1].endswith('\n'):
                    new_lines[-1] = new_lines[-1] + '\n'
                new_lines.append(f'{key}="{refresh_token}"\n')

            # Make a timestamped backup of target path before replacing
            try:
                if target_path.exists():
                    bak = target_path.with_suffix('.bak')
                    tmp_bak = target_path.with_suffix('.bak.tmp')
                    with open(tmp_bak, 'w') as fh:
                        fh.writelines(original_lines)
                    os.replace(tmp_bak, bak)
            except Exception:
                LOG.debug('Failed to create .env backup; proceeding with caution', exc_info=True)

            # Write new content atomically
            tmp = target_path.with_suffix('.tmp')
            with open(tmp, 'w') as fh:
                fh.writelines(new_lines)
            os.replace(tmp, target_path)
            LOG.info('Updated %s with SCHWAB_REFRESH_TOKEN (preserving existing content) at %s', target_path.name, target_path)
        except Exception as e:  # pragma: no cover - defensive
            LOG.debug('Failed to append refresh token to .env: %s', e, exc_info=True)

    def stream(self):
        """Return a synchronous stream adapter that wraps schwab-py's StreamClient.
        The wrapper provides `start(symbols, on_tick, on_level2)` and `stop()`
        methods to match existing expectations used elsewhere in the codebase.
        """
        try:
            sc = schwab_streaming.StreamClient(self.schwab)
            return _SchwabStreamAdapter(sc, auth_client=self)
        except Exception as e:
            LOG.error("Failed to construct StreamClient adapter: %s", e)
            raise


class _SchwabStreamAdapter:
    """Synchronous adapter over `schwab.streaming.StreamClient`.
    Implements `start(symbols, on_tick, on_level2)` and `stop()`.
    """
    def __init__(self, stream_client: schwab_streaming.StreamClient, auth_client: Optional[SchwabAuthClient] = None):
        self._sc = stream_client
        self._running = False
        self._symbols = []
        self._auth_client = auth_client

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
        # After a successful login (which may have rotated tokens), persist
        # the current tokens from the parent auth client if available
        try:
            if self._auth_client and getattr(self._auth_client, 'schwab', None):
                tokens = getattr(self._auth_client.schwab, 'tokens', None)
                if isinstance(tokens, dict):
                    self._auth_client._persist_tokens(tokens)
                    if os.environ.get('SCHWAB_PERSIST_REFRESH_TO_ENV') in ("1", "true", "yes"):
                        self._auth_client._append_refresh_to_env(tokens.get('refresh_token'))
        except Exception:
            pass
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
    # Caller (scripts/start_schwab_streamer.py or calling code) should ensure
    # the required SCHWAB_* settings are present. We avoid enforcing them
    # here so tests can monkeypatch SchwabAuthClient.
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
