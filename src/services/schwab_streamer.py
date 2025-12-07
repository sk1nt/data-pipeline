"""Schwab streaming client that forwards tick + level2 data into the trading system."""

from __future__ import annotations

import asyncio
import calendar
import json
import os
import threading
import time
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Callable, Optional

from schwab import auth
from schwab import streaming as schwab_streaming

from ..config import settings
from lib.logging import get_logger
from lib.redis_client import RedisClient
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
                LOG.warning(
                    "Schwab easy_client failed to load token file, attempting to rotate legacy token: %s",
                    e,
                )
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
                            "access_token": os.environ.get("SCHWAB_ACCESS_TOKEN", ""),
                            "refresh_token": self.refresh_token,
                            "token_type": "Bearer",
                            "expires_in": int(os.environ.get("SCHWAB_EXPIRES_IN", "0")),
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
            tokens = getattr(self.schwab, "tokens", None)
            if (
                getattr(self, "_created_schwab_client", False)
                and isinstance(tokens, dict)
                and tokens.get("refresh_token")
            ):
                self._persist_tokens(tokens)
            # Always append refresh token into .env when requested by environment,
            # even when a dummy (in tests) or externally-provided schwab client was
            # used for initialization.
            if (
                isinstance(tokens, dict)
                and tokens.get("refresh_token")
                and os.environ.get("SCHWAB_PERSIST_REFRESH_TO_ENV")
                in ("1", "true", "yes")
            ):
                self._append_refresh_to_env(tokens.get("refresh_token"))
        except Exception:
            pass

    def _load_tokens_from_env(self) -> Optional[dict]:
        """Build token dict from environment variables if present.

        This supports CI-friendly runs and avoids requiring a browser when
        `SCHWAB_REFRESH_TOKEN` (or `SCHWAB_TOKENS_JSON`) is provided in env.
        """
        # First, allow a JSON-encoded token to be supplied via env
        json_tok = os.environ.get("SCHWAB_TOKENS_JSON")
        if json_tok:
            try:
                parsed = json.loads(json_tok)
                # If keys are nested under `tokens` (the exchange script format), unwrap
                if isinstance(parsed, dict) and "tokens" in parsed:
                    parsed = parsed["tokens"]
                if isinstance(parsed, dict) and parsed.get("refresh_token"):
                    # persist for future runs
                    try:
                        self._persist_tokens(parsed)
                    except Exception:
                        pass
                    return parsed
            except Exception:
                LOG.debug("SCHWAB_TOKENS_JSON present but failed to parse; ignoring")

        # Fallback: use access/refresh tokens as separate env variables
        rt = (
            os.environ.get("SCHWAB_REFRESH_TOKEN")
            or os.environ.get("SCHWAB_RTOKEN")
            or None
        )
        at = (
            os.environ.get("SCHWAB_ACCESS_TOKEN")
            or os.environ.get("SCHWAB_ATOKEN")
            or None
        )
        if rt or at:
            tokens = {
                "access_token": at or "",
                "refresh_token": rt or "",
                "token_type": "Bearer",
                "expires_in": int(os.environ.get("SCHWAB_EXPIRES_IN", "0")),
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
        self._refresh_thread = threading.Thread(
            target=self._auto_refresh_loop, daemon=True
        )
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
                        if os.environ.get("SCHWAB_PERSIST_REFRESH_TO_ENV") in (
                            "1",
                            "true",
                            "yes",
                        ):
                            try:
                                self._append_refresh_to_env(tokens.get("refresh_token"))
                            except Exception:
                                LOG.debug(
                                    "Failed to append refresh token to .env during refresh",
                                    exc_info=True,
                                )
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
            expires_in = tokens.get("expires_in", 1800)
            expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
            refresh_token = tokens.get("refresh_token", self.refresh_token)
            return SchwabToken(
                access_token=access_token,
                expires_at=expires_at,
                refresh_token=refresh_token,
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
                            LOG.debug(
                                "Failed to persist tokens after auto refresh: %s", e
                            )
                        if os.environ.get("SCHWAB_PERSIST_REFRESH_TO_ENV") in (
                            "1",
                            "true",
                            "yes",
                        ):
                            try:
                                self._append_refresh_to_env(tokens.get("refresh_token"))
                            except Exception:
                                LOG.debug(
                                    "Failed to append refresh token to .env during auto-refresh",
                                    exc_info=True,
                                )
                    # Rotate refresh token on longer schedule
                    if (
                        now - self._last_refresh_token_rotate
                    ).total_seconds() >= self._refresh_token_rotate_interval:
                        LOG.info("Rotating refresh token (scheduled interval reached)")
                        tokens = self._refresh_tokens_internal()
                        if isinstance(tokens, dict):
                            try:
                                self._persist_tokens(tokens)
                            except Exception as e:  # pragma: no cover - defensive
                                LOG.debug(
                                    "Failed to persist tokens after rotation: %s", e
                                )
                        self._last_refresh_token_rotate = now
            except Exception as e:
                LOG.error("Auto-refresh failed: %s", e)
            # Sleep in chunks to be responsive to stop
            slept = 0
            while (
                slept < self._access_refresh_interval
                and not self._stop_refresh.is_set()
            ):
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
                if isinstance(tokens, dict) and os.environ.get(
                    "SCHWAB_PERSIST_REFRESH_TO_ENV"
                ) in ("1", "true", "yes"):
                    try:
                        self._append_refresh_to_env(tokens.get("refresh_token"))
                    except Exception:
                        LOG.debug(
                            "Failed to append refresh token to .env during forced rotation",
                            exc_info=True,
                        )
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
            env_path = Path(__file__).resolve().parents[2] / ".env"
            # Never write to `.env` to avoid overwriting developer configurations.
            # Always write to `.env.back` (or create it) so appended refresh tokens
            # do not risk leaking or overwriting .env values.
            target_path = env_path.with_suffix(".back")

            # Read current raw lines so we can preserve formatting and comments
            original_lines = []
            if target_path.exists():
                with open(target_path, "r") as fh:
                    original_lines = fh.readlines()

            # Find if the variable exists; maintain original line if present
            key = "SCHWAB_REFRESH_TOKEN"
            found = False
            new_lines = []
            for ln in original_lines:
                if ln.strip().startswith(f"{key}=") or ln.strip().startswith(
                    f"{key} ="
                ):
                    # preserve any comment after the value
                    comment = ""
                    if "#" in ln:
                        # Keep inline comments if present
                        parts = ln.split("#", 1)
                        comment = "#" + parts[1].rstrip("\n")
                    new_lines.append(f'{key}="{refresh_token}" {comment}\n')
                    found = True
                else:
                    new_lines.append(ln)

            if not found:
                # If the file had a trailing newline and not empty, preserve format
                if new_lines and not new_lines[-1].endswith("\n"):
                    new_lines[-1] = new_lines[-1] + "\n"
                new_lines.append(f'{key}="{refresh_token}"\n')

            # Make a timestamped backup of target path before replacing
            try:
                if target_path.exists():
                    bak = target_path.with_suffix(".bak")
                    tmp_bak = target_path.with_suffix(".bak.tmp")
                    with open(tmp_bak, "w") as fh:
                        fh.writelines(original_lines)
                    os.replace(tmp_bak, bak)
            except Exception:
                LOG.debug(
                    "Failed to create .env backup; proceeding with caution",
                    exc_info=True,
                )

            # Write new content atomically
            tmp = target_path.with_suffix(".tmp")
            with open(tmp, "w") as fh:
                fh.writelines(new_lines)
            os.replace(tmp, target_path)
            LOG.info(
                "Updated %s with SCHWAB_REFRESH_TOKEN (preserving existing content) at %s",
                target_path.name,
                target_path,
            )
        except Exception as e:  # pragma: no cover - defensive
            LOG.debug("Failed to append refresh token to .env: %s", e, exc_info=True)

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

    def __init__(
        self,
        stream_client: schwab_streaming.StreamClient,
        auth_client: Optional[SchwabAuthClient] = None,
    ):
        self._sc = stream_client
        self._running = False
        self._symbols = []
        self._auth_client = auth_client
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._message_future: Optional[Future] = None

    def start(self, symbols=None, on_tick=None, on_level2=None):
        symbols = symbols or []
        self._symbols = symbols
        self._ensure_loop()
        LOG.debug("Stream adapter start invoked for %d symbols", len(symbols))
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
        LOG.debug("Logging into Schwab stream...")
        self._run_coro(self._sc.login())
        LOG.debug("Login succeeded; subscribing for symbols")
        # After a successful login (which may have rotated tokens), persist
        # the current tokens from the parent auth client if available
        try:
            if self._auth_client and getattr(self._auth_client, "schwab", None):
                tokens = getattr(self._auth_client.schwab, "tokens", None)
                if isinstance(tokens, dict):
                    self._auth_client._persist_tokens(tokens)
                    if os.environ.get("SCHWAB_PERSIST_REFRESH_TO_ENV") in (
                        "1",
                        "true",
                        "yes",
                    ):
                        self._auth_client._append_refresh_to_env(
                            tokens.get("refresh_token")
                        )
        except Exception:
            pass
        # Subscribe to symbol types
        # Map to futures vs equities
        futures_symbols = {s for s in symbols if _is_futures_contract(s)}
        equity_symbols = [s for s in symbols if not _is_futures_contract(s)]
        if futures_symbols:
            LOG.debug("Subscribing futures symbols: %s", futures_symbols)
            futures_list = list(futures_symbols)
            self._run_coro(self._sc.level_one_futures_add(futures_list))
            self._subscribe_futures_level2(futures_list)
        if equity_symbols:
            LOG.debug("Subscribing equity symbols: %s", equity_symbols)
            self._run_coro(self._sc.level_one_equity_add(equity_symbols))
        self._running = True
        LOG.debug("Starting Schwab message pump")
        self._message_future = asyncio.run_coroutine_threadsafe(
            self._consume_messages(), self._loop
        )

    def stop(self):
        if self._running:
            try:
                self._run_coro(self._sc.logout())
                LOG.debug("Logged out of Schwab stream")
            except Exception:
                pass
            self._running = False
        if self._message_future:
            self._message_future.cancel()
            try:
                self._message_future.result(timeout=5)
            except Exception:
                pass
            self._message_future = None
        if self._loop:
            loop = self._loop
            loop.call_soon_threadsafe(loop.stop)
            if self._loop_thread:
                self._loop_thread.join(timeout=5)
            loop.close()
            self._loop = None
            self._loop_thread = None

    def _subscribe_futures_level2(self, futures_symbols: list[str]) -> None:
        """Subscribe to futures depth book if supported."""
        if not futures_symbols:
            return
        try:
            LOG.debug("Subscribing futures symbols to level2 feed: %s", futures_symbols)
            self._run_coro(self._sc.nasdaq_book_add(futures_symbols))
        except AttributeError:
            LOG.warning(
                "Schwab client does not expose NASDAQ level2 subscription; skipping"
            )
        except Exception as exc:
            LOG.warning("Failed to subscribe futures level2 feed: %s", exc)

    @property
    def is_running(self) -> bool:
        return self._running and self._loop is not None and self._loop.is_running()

    def _ensure_loop(self) -> None:
        if self._loop and self._loop.is_running():
            return
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run_loop() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(
            target=_run_loop, name="schwab-stream-loop", daemon=True
        )
        thread.start()
        ready.wait(timeout=2)
        self._loop = loop
        self._loop_thread = thread

    def _run_coro(self, coro, timeout: int = 30):
        if not self._loop:
            raise RuntimeError("Event loop not initialized")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            result = future.result(timeout=timeout)
            LOG.debug("Coroutine %s completed", getattr(coro, "__name__", repr(coro)))
            return result
        except TimeoutError:
            future.cancel()
            LOG.error(
                "Coroutine %s timed out after %ss",
                getattr(coro, "__name__", repr(coro)),
                timeout,
            )
            raise

    async def _consume_messages(self) -> None:
        LOG.debug("Message pump loop started")
        while self._running and self._loop and self._loop.is_running():
            try:
                await self._sc.handle_message()
                LOG.debug("Consumed message from Schwab stream")
            except asyncio.CancelledError:
                LOG.debug("Message pump cancelled")
                break
            except Exception as exc:
                LOG.warning("Schwab stream handler error: %s", exc)
                await asyncio.sleep(1)


class SchwabMessageParser:
    """Convert Schwab payloads into internal models."""

    LEVEL_ONE_SERVICES = {"LEVELONE_FUTURES", "LEVELONE_EQUITY"}
    BOOK_SERVICES = {"NASDAQ_BOOK", "NYSE_BOOK"}

    def parse(self, payload: dict) -> list[tuple[str, object]]:
        """Return list of (kind, event) tuples."""
        events: list[tuple[str, object]] = []
        msg_type = (payload.get("type") or payload.get("topic") or "").lower()
        if msg_type in {"tick", "trade"}:
            events.extend(self._parse_legacy_tick(payload))
        elif msg_type in {"level2", "book"}:
            events.extend(self._parse_legacy_book(payload))
        else:
            entries = []
            if "data" in payload:
                entries = payload.get("data", []) or []
            elif payload.get("service"):
                entries = [payload]
            for entry in entries:
                service = (entry.get("service") or "").upper()
                if service in self.LEVEL_ONE_SERVICES:
                    events.extend(self._parse_level_one_entry(entry))
                elif service in self.BOOK_SERVICES:
                    events.extend(self._parse_book_entry(entry))
        return events

    def _parse_legacy_tick(self, payload: dict) -> list[tuple[str, object]]:
        event = TickEvent(
            symbol=payload.get("symbol", payload.get("ticker", "")).upper(),
            timestamp=self._coerce_ts(payload.get("timestamp")),
            last=self._safe_float(payload.get("price") or payload.get("last")),
            bid=self._safe_float(payload.get("bid")),
            ask=self._safe_float(payload.get("ask")),
            volume=self._safe_float(payload.get("volume")),
        )
        return [("tick", event)]

    def _parse_legacy_book(self, payload: dict) -> list[tuple[str, object]]:
        bids = [
            Level2Quote(
                price=self._safe_float(level.get("price", 0.0)),
                size=self._safe_float(level.get("size", 0.0)) or 0.0,
            )
            for level in payload.get("bids", [])
        ]
        asks = [
            Level2Quote(
                price=self._safe_float(level.get("price", 0.0)),
                size=self._safe_float(level.get("size", 0.0)) or 0.0,
            )
            for level in payload.get("asks", [])
        ]
        event = Level2Event(
            symbol=payload.get("symbol", payload.get("ticker", "")).upper(),
            timestamp=self._coerce_ts(payload.get("timestamp")),
            bids=bids,
            asks=asks,
        )
        return [("level2", event)]

    def _parse_level_one_entry(self, entry: dict) -> list[tuple[str, object]]:
        events: list[tuple[str, object]] = []
        timestamp = entry.get("timestamp")
        for row in entry.get("content", []) or []:
            symbol = (
                row.get("symbol")
                or row.get("SYMBOL")
                or row.get("key")
                or entry.get("symbol")
                or ""
            )
            symbol = symbol.strip().upper()
            if not symbol:
                continue
            quote_ts = self._lookup_field(
                row, "QUOTE_TIME_MILLIS", "TRADE_TIME_MILLIS", "10", "11", timestamp
            )
            bid_price = self._safe_float(
                self._lookup_field(row, "BID_PRICE", "BID", "1")
            )
            ask_price = self._safe_float(
                self._lookup_field(row, "ASK_PRICE", "ASK", "2")
            )
            bid_size = self._safe_float(
                self._lookup_field(row, "BID_SIZE", "BIDSIZE", "BID_VOLUME", "4")
            )
            ask_size = self._safe_float(
                self._lookup_field(row, "ASK_SIZE", "ASKSIZE", "ASK_VOLUME", "5")
            )
            last_price = self._safe_float(
                self._lookup_field(row, "LAST_PRICE", "LAST", "3")
            )
            volume = self._safe_float(
                self._lookup_field(row, "TOTAL_VOLUME", "VOLUME", "LAST_SIZE", "8", "9")
            )
            tick = TickEvent(
                symbol=symbol,
                timestamp=self._coerce_ts(quote_ts),
                last=last_price,
                bid=bid_price,
                ask=ask_price,
                volume=volume,
            )
            events.append(("tick", tick))
            # Level-one feeds still expose best bid/ask which we treat as level2 depth level 1.
            bids = []
            asks = []
            if bid_price is not None:
                bids.append(Level2Quote(price=bid_price, size=bid_size or 0.0))
            if ask_price is not None:
                asks.append(Level2Quote(price=ask_price, size=ask_size or 0.0))
            if bids or asks:
                events.append(
                    (
                        "level2",
                        Level2Event(
                            symbol=symbol,
                            timestamp=tick.timestamp,
                            bids=bids,
                            asks=asks,
                        ),
                    )
                )
        return events

    def _parse_book_entry(self, entry: dict) -> list[tuple[str, object]]:
        events: list[tuple[str, object]] = []
        timestamp = entry.get("timestamp")
        for row in entry.get("content", []) or []:
            symbol = (
                row.get("symbol")
                or row.get("SYMBOL")
                or row.get("key")
                or entry.get("symbol")
                or ""
            )
            symbol = symbol.strip().upper()
            if not symbol:
                continue
            bids = self._parse_book_side(row.get("BIDS", []) or [], side="bid")
            asks = self._parse_book_side(row.get("ASKS", []) or [], side="ask")
            if not bids and not asks:
                continue
            event = Level2Event(
                symbol=symbol,
                timestamp=self._coerce_ts(row.get("BOOK_TIME") or timestamp),
                bids=bids,
                asks=asks,
            )
            events.append(("level2", event))
        return events

    def _parse_book_side(self, ladder: list, *, side: str) -> list[Level2Quote]:
        quotes: list[Level2Quote] = []
        for level in ladder:
            price = self._extract_price(level, side)
            if price is None:
                continue
            size = self._extract_size(level, side)
            quotes.append(Level2Quote(price=price, size=size or 0.0))
        return quotes

    @staticmethod
    def _lookup_field(row: dict, *candidates):
        for key in candidates:
            if key is None:
                continue
            if key in row and row[key] not in (None, ""):
                return row[key]
        return None

    def _extract_price(self, level: dict, side: str) -> Optional[float]:
        price_keys = ["PRICE", "price"]
        if side == "bid":
            price_keys.insert(0, "BID_PRICE")
        else:
            price_keys.insert(0, "ASK_PRICE")
        for key in price_keys:
            if key in level and level[key] is not None:
                return self._safe_float(level[key])
        return None

    def _extract_size(self, level: dict, side: str) -> Optional[float]:
        size_keys = ["SIZE", "size", "VOLUME", "TOTAL_VOLUME", "TOTAL_SIZE"]
        if side == "bid":
            size_keys.insert(0, "BID_VOLUME")
        else:
            size_keys.insert(0, "ASK_VOLUME")
        for key in size_keys:
            if key in level and level[key] is not None:
                return self._safe_float(level[key])
        nested_key = "BIDS" if side == "bid" else "ASKS"
        nested = level.get(nested_key)
        if isinstance(nested, list):
            total = 0.0
            for entry in nested:
                vol_key = "BID_VOLUME" if side == "bid" else "ASK_VOLUME"
                vol = self._safe_float(entry.get(vol_key))
                if vol:
                    total += vol
            if total > 0:
                return total
        return None

    @staticmethod
    def _coerce_ts(value) -> datetime:
        if isinstance(value, (int, float)):
            return datetime.utcfromtimestamp(
                float(value) / (1000 if value > 1e12 else 1)
            )
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
        tick_channel = getattr(
            self.publisher, "tick_channel", settings.schwab_tick_channel
        )
        LOG.debug(
            "Heartbeat every %ss; Redis tick channel %s",
            self.heartbeat_seconds,
            tick_channel,
        )
        self.auth_client.start_auto_refresh()  # Start background token refresh
        self.stream.start(
            symbols=self.symbols, on_tick=self._on_tick, on_level2=self._on_level2
        )
        LOG.debug("Stream start returned control to client")

    def stop(self) -> None:
        """Signal streaming loop to stop."""
        self.stream.stop()
        self.auth_client.stop_auto_refresh()  # Stop background token refresh

    @property
    def is_running(self) -> bool:
        state = getattr(self.stream, "is_running", None)
        if callable(state):  # pragma: no cover - defensive
            try:
                return bool(state())
            except Exception:
                return False
        if state is None:
            return False
        return bool(state)

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
    publisher_factory = publisher_factory or (
        lambda rc: TradingEventPublisher(
            redis_client=rc,
            tick_channel=settings.schwab_tick_channel,
            level2_channel=settings.schwab_level2_channel,
        )
    )
    publisher = publisher_factory(redis_client)
    auth_client = SchwabAuthClient(
        client_id=settings.schwab_client_id,
        client_secret=settings.schwab_client_secret,
        refresh_token=settings.schwab_refresh_token,
        rest_url=settings.schwab_rest_url,
        interactive=interactive,
    )
    base_symbols = (
        settings.tastytrade_symbol_list
        if use_tastytrade_symbols
        else settings.schwab_symbol_list
    )
    symbols = normalize_stream_symbols(base_symbols)
    return SchwabStreamClient(
        auth_client=auth_client,
        publisher=publisher,
        stream_url=settings.schwab_stream_url,
        symbols=symbols,
        heartbeat_seconds=settings.schwab_heartbeat_seconds,
        tick_handler=tick_handler,
        level2_handler=level2_handler,
    )


FUTURES_ROOTS = {"MNQ", "MES", "ES", "NQ"}
FUTURES_MONTH_CODES = {
    1: "F",
    2: "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z",
}
QUARTER_MONTHS = (3, 6, 9, 12)


def normalize_stream_symbols(symbols: list[str]) -> list[str]:
    """Expand configured symbol list into Schwab-ready identifiers.

    Futures roots (e.g. MNQ) are converted to the current CME contract code
    with a leading slash: `/MNQZ25`.
    """

    normalized: list[str] = []
    for sym in symbols:
        clean = sym.strip().upper()
        if not clean:
            continue
        if clean.startswith("/"):
            normalized.append(clean)
            continue
        root = clean.lstrip("/")
        if root in FUTURES_ROOTS:
            normalized.append(resolve_cme_contract(root))
        else:
            normalized.append(clean)
    return normalized


def resolve_cme_contract(root: str, *, as_of: Optional[datetime] = None) -> str:
    """Return current front-month CME contract for the given root (e.g. MNQ)."""

    now = as_of or datetime.utcnow()
    year = now.year
    candidates = []
    for offset in range(2):  # current year and next year
        yr = year + offset
        for month in QUARTER_MONTHS:
            candidates.append((yr, month))
    for yr, month in candidates:
        if yr == now.year and month < now.month:
            continue
        expiry = _third_friday(yr, month)
        if now <= expiry:
            code = FUTURES_MONTH_CODES[month]
            return f"/{root}{code}{yr % 100:02d}"
    # Fallback to March next year if loop above didn't return
    next_year = now.year + 1
    return f"/{root}{FUTURES_MONTH_CODES[3]}{next_year % 100:02d}"


def _third_friday(year: int, month: int) -> datetime:
    cal = calendar.monthcalendar(year, month)
    fridays = [week[calendar.FRIDAY] for week in cal if week[calendar.FRIDAY] != 0]
    day = fridays[2]  # third Friday
    return datetime(year, month, day, 23, 59, 59)


def _is_futures_contract(symbol: str) -> bool:
    if not symbol:
        return False
    clean = symbol.strip().upper()
    if clean.startswith("/"):
        return True
    return clean in FUTURES_ROOTS
