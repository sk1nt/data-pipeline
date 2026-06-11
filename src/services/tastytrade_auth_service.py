"""Shared Redis-first TastyTrade session keeper.

The TastyTrade SDK session object is process-local, but all processes can share
non-secret auth health metadata through Redis and fall back to a JSON status file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:  # pragma: no cover - optional dependency in some test environments
    from tastytrade import Session
except Exception:  # pragma: no cover
    Session = None  # type: ignore

try:  # Keep direct `services.*` imports working when src/ is on sys.path.
    from src.lib.redis_client import RedisClient, get_redis_client
except Exception:  # pragma: no cover
    from lib.redis_client import RedisClient, get_redis_client  # type: ignore

LOGGER = logging.getLogger(__name__)

AUTH_STATE_KEY = "tastytrade:auth:state"
AUTH_REFRESH_LOCK_KEY = "tastytrade:auth:refresh_lock"
DEFAULT_STATE_PATH = Path("data/tastytrade_auth_state.json")
AUTH_ERROR_TEXT = (
    "TastyTrade authentication failed (refresh token invalid or revoked). "
    "Update the configured refresh token."
)


class TastytradeAuthError(RuntimeError):
    """Raised when TastyTrade cannot create or refresh an authorized session."""


class TastytradeTransientAuthError(RuntimeError):
    """Raised when auth is temporarily unavailable but may recover."""


@dataclass
class TastytradeAuthSettings:
    client_secret: str
    refresh_token: str
    use_sandbox: bool = False
    refresh_buffer_seconds: int = 300
    fallback_state_path: Path = DEFAULT_STATE_PATH
    state_ttl_seconds: int = 1800
    refresh_lock_ttl_seconds: int = 60
    loop_interval_seconds: int = 30


def is_hard_auth_error(exc: Exception | str | None) -> bool:
    msg = str(exc or "").lower()
    return any(
        token in msg
        for token in (
            "invalid_grant",
            "invalid_token",
            "grant revoked",
            "refresh token invalid",
            "refresh token revoked",
        )
    )


class TastytradeAuthService:
    """Own a warm TastyTrade SDK session and publish shared auth metadata."""

    def __init__(
        self,
        settings: TastytradeAuthSettings,
        *,
        redis_client: Optional[RedisClient] = None,
        session_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.settings = settings
        self.redis_client = redis_client
        self.session_factory = session_factory or Session
        self._session: Any = None
        self._session_expiration: Optional[datetime] = None
        self._last_refresh_at: Optional[datetime] = None
        self._last_error: Optional[str] = None
        self._needs_reauth = False
        self._source = "memory"
        self._lock = threading.RLock()
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event: Optional[asyncio.Event] = None

    @property
    def session_expiration(self) -> Optional[datetime]:
        return self._session_expiration

    def ensure_authorized(self) -> bool:
        self.get_session()
        return True

    def get_session(self) -> Any:
        with self._lock:
            if self._needs_reauth:
                raise TastytradeAuthError(AUTH_ERROR_TEXT)
            if self._session is None:
                return self._create_session()
            if self._should_refresh():
                self.refresh_session()
            return self._session

    def refresh_session(self, *, force: bool = False) -> Any:
        with self._lock:
            if self._needs_reauth:
                raise TastytradeAuthError(AUTH_ERROR_TEXT)
            if self._session is None:
                return self._create_session()
            if not force and not self._should_refresh():
                return self._session

            lock_token = self._acquire_refresh_lock()
            if lock_token is None:
                state = self.read_shared_state()
                if self._state_is_healthy(state):
                    return self._session
            try:
                self._session.refresh()
                self._session_expiration = self._derive_expiration(self._session)
                self._last_refresh_at = datetime.now(timezone.utc)
                self._last_error = None
                self._needs_reauth = False
                self._write_state("redis")
                return self._session
            except Exception as exc:
                self._handle_auth_exception(exc)
                raise
            finally:
                if lock_token is not None:
                    self._release_refresh_lock(lock_token)

    def set_refresh_token(self, refresh_token: str) -> None:
        token = (refresh_token or "").strip()
        if not token:
            raise ValueError("refresh_token must be provided")
        with self._lock:
            self.settings.refresh_token = token
            self._session = None
            self._session_expiration = None
            self._needs_reauth = False
            self._last_error = None
            self._create_session()

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="tastytrade-auth-keeper")

    async def stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        LOGGER.info("TastyTrade auth keeper started")
        hard_auth_logged = False
        while self._stop_event and not self._stop_event.is_set():
            try:
                await asyncio.to_thread(self.refresh_session)
                hard_auth_logged = False
            except TastytradeAuthError:
                if not hard_auth_logged:
                    LOGGER.error(
                        "TastyTrade auth keeper paused: refresh token is invalid "
                        "or revoked for the configured environment"
                    )
                    hard_auth_logged = True
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=300.0)
                except asyncio.TimeoutError:
                    continue
            except Exception as exc:
                LOGGER.warning("TastyTrade auth keeper refresh failed: %s", exc)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=max(5, self.settings.loop_interval_seconds),
                )
            except asyncio.TimeoutError:
                continue
        LOGGER.info("TastyTrade auth keeper stopped")

    def status(self) -> Dict[str, Any]:
        state = self.read_shared_state()
        with self._lock:
            expires_at = self._iso(self._session_expiration)
            session_valid = bool(
                self._session
                and self._session_expiration
                and datetime.now(timezone.utc) < self._session_expiration
                and not self._needs_reauth
            )
            return {
                "running": bool(self._task and not self._task.done()),
                "session_valid": session_valid,
                "expires_at": expires_at,
                "last_refresh_at": self._iso(self._last_refresh_at),
                "last_error": self._last_error,
                "source": self._source,
                "needs_reauth": self._needs_reauth,
                "shared_state": state,
            }

    def read_shared_state(self) -> Dict[str, Any]:
        redis_state = self._read_redis_state()
        if redis_state:
            self._source = "redis"
            return redis_state
        json_state = self._read_json_state()
        if json_state:
            self._source = "json"
            return json_state
        self._source = "memory"
        return {}

    def _create_session(self) -> Any:
        if self.session_factory is None:
            raise RuntimeError("tastytrade SDK is not installed")
        try:
            kwargs = {
                "provider_secret": self.settings.client_secret,
                "refresh_token": self.settings.refresh_token,
            }
            if self.settings.use_sandbox:
                kwargs["is_test"] = True
            try:
                self._session = self.session_factory(**kwargs)
            except TypeError:
                kwargs.pop("is_test", None)
                self._session = self.session_factory(**kwargs)
            self._session_expiration = self._derive_expiration(self._session)
            self._last_refresh_at = datetime.now(timezone.utc)
            self._last_error = None
            self._needs_reauth = False
            self._write_state("redis")
            return self._session
        except Exception as exc:
            self._handle_auth_exception(exc)
            raise

    def _derive_expiration(self, session: Any) -> datetime:
        exp = getattr(session, "session_expiration", None)
        if isinstance(exp, datetime):
            if exp.tzinfo is None:
                return exp.replace(tzinfo=timezone.utc)
            return exp.astimezone(timezone.utc)
        return datetime.now(timezone.utc) + timedelta(minutes=20)

    def _should_refresh(self) -> bool:
        if self._session_expiration is None:
            return True
        refresh_at = self._session_expiration - timedelta(
            seconds=max(0, self.settings.refresh_buffer_seconds)
        )
        return datetime.now(timezone.utc) >= refresh_at

    def _session_is_still_usable(self) -> bool:
        return bool(
            self._session
            and self._session_expiration
            and datetime.now(timezone.utc) < self._session_expiration
        )

    def _handle_auth_exception(self, exc: Exception) -> None:
        self._last_error = str(exc)
        if is_hard_auth_error(exc):
            self._needs_reauth = True
            if not self._session_is_still_usable():
                self._session = None
                self._session_expiration = None
            self._write_state("redis")
            raise TastytradeAuthError(AUTH_ERROR_TEXT) from exc
        self._write_state("redis")
        raise TastytradeTransientAuthError(str(exc)) from exc

    def _write_state(self, preferred_source: str) -> None:
        payload = {
            "session_valid": bool(self._session and not self._needs_reauth),
            "expires_at": self._iso(self._session_expiration),
            "last_refresh_at": self._iso(self._last_refresh_at),
            "last_error": self._last_error,
            "needs_reauth": self._needs_reauth,
            "updated_at": self._iso(datetime.now(timezone.utc)),
        }
        wrote_redis = False
        if preferred_source in {"redis", "json"}:
            wrote_redis = self._write_redis_state(payload)
        self._write_json_state(payload)
        self._source = "redis" if wrote_redis else "json"

    def _redis(self) -> Any:
        if self.redis_client is None:
            self.redis_client = get_redis_client()
        return self.redis_client.client

    def _read_redis_state(self) -> Dict[str, Any]:
        try:
            raw = self._redis().get(AUTH_STATE_KEY)
            return json.loads(raw) if raw else {}
        except Exception:
            return {}

    def _write_redis_state(self, payload: Dict[str, Any]) -> bool:
        try:
            self._redis().setex(
                AUTH_STATE_KEY,
                self.settings.state_ttl_seconds,
                json.dumps(payload, default=str),
            )
            return True
        except Exception:
            return False

    def _read_json_state(self) -> Dict[str, Any]:
        try:
            path = self.settings.fallback_state_path
            if not path.exists():
                return {}
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_json_state(self, payload: Dict[str, Any]) -> None:
        try:
            path = self.settings.fallback_state_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except Exception:
            LOGGER.debug("Failed to write TastyTrade auth JSON state", exc_info=True)

    def _state_is_healthy(self, state: Dict[str, Any]) -> bool:
        if not state or state.get("needs_reauth"):
            return False
        expires_at = self._parse_dt(state.get("expires_at"))
        return bool(expires_at and expires_at > datetime.now(timezone.utc))

    def _acquire_refresh_lock(self) -> Optional[str]:
        token = f"{os.getpid()}:{threading.get_ident()}:{time.monotonic_ns()}"
        try:
            ok = self._redis().set(
                AUTH_REFRESH_LOCK_KEY,
                token,
                nx=True,
                ex=self.settings.refresh_lock_ttl_seconds,
            )
            return token if ok else None
        except Exception:
            return token

    def _release_refresh_lock(self, token: str) -> None:
        try:
            current = self._redis().get(AUTH_REFRESH_LOCK_KEY)
            if current == token:
                self._redis().delete(AUTH_REFRESH_LOCK_KEY)
        except Exception:
            pass

    @staticmethod
    def _parse_dt(value: Any) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    @staticmethod
    def _iso(value: Optional[datetime]) -> Optional[str]:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()


_default_auth_service: Optional[TastytradeAuthService] = None
_default_lock = threading.Lock()


def build_auth_settings_from_config() -> TastytradeAuthSettings:
    from config.settings import config

    return TastytradeAuthSettings(
        client_secret=config.effective_tastytrade_client_secret,
        refresh_token=config.effective_tastytrade_refresh_token,
        use_sandbox=bool(getattr(config, "tastytrade_use_sandbox", False)),
        refresh_buffer_seconds=int(
            os.getenv("TASTYTRADE_AUTH_REFRESH_BUFFER_SECONDS", "300")
        ),
        fallback_state_path=Path(
            os.getenv("TASTYTRADE_AUTH_STATE_PATH", str(DEFAULT_STATE_PATH))
        ),
    )


def get_tastytrade_auth_service() -> TastytradeAuthService:
    global _default_auth_service
    with _default_lock:
        if _default_auth_service is None:
            _default_auth_service = TastytradeAuthService(
                build_auth_settings_from_config()
            )
        return _default_auth_service


def reset_tastytrade_auth_service(service: Optional[TastytradeAuthService] = None) -> None:
    global _default_auth_service
    with _default_lock:
        _default_auth_service = service
