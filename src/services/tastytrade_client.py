from __future__ import annotations

from typing import Any

from config.settings import config
from services.tastytrade_auth_service import (
    TastytradeAuthError,
    TastytradeTransientAuthError,
    get_tastytrade_auth_service,
)


class TastyTradeClient:
    def __init__(self):
        self._auth_service = get_tastytrade_auth_service()

    @property
    def _session(self) -> Any:
        return getattr(self._auth_service, "_session", None)

    @property
    def _session_expiration(self):
        return self._auth_service.session_expiration

    def set_refresh_token(self, refresh_token: str) -> None:
        """Update the refresh token used for new sessions without touching .env."""
        token = (refresh_token or "").strip()
        if not token:
            raise ValueError("refresh_token must be provided")
        if config.tastytrade_use_sandbox:
            config.tastytrade_refresh_token = token
        else:
            config.tastytrade_prod_refresh_token = token
        self._auth_service.set_refresh_token(token)

    def _ensure_session(self):
        try:
            return self._auth_service.get_session()
        except TastytradeAuthError:
            raise
        except TastytradeTransientAuthError as exc:
            raise TastytradeAuthError(
                f"TastyTrade authorization temporarily unavailable: {exc}"
            ) from exc
        except Exception as exc:
            raise TastytradeAuthError(f"TastyTrade authorization error: {exc}") from exc

    def get_session(self):
        """Get a warm authorized session."""
        return self._ensure_session()

    def refresh_session(self):
        """Force refresh the shared session."""
        try:
            self._auth_service.refresh_session(force=True)
        except TastytradeAuthError:
            raise
        except TastytradeTransientAuthError as exc:
            raise TastytradeAuthError(
                f"TastyTrade authorization temporarily unavailable: {exc}"
            ) from exc

    def ensure_authorized(self) -> bool:
        """Attempt to ensure session is valid; raise TastytradeAuthError if not."""
        self._ensure_session()
        return True

    def status(self):
        return self._auth_service.status()


def post_with_retry(
    session, url: str, data: str, max_retries: int = 3, initial_backoff: float = 0.5
):
    """Post with retry on transient errors using retry_with_backoff."""
    from src.lib.retries import retry_with_backoff

    @retry_with_backoff(max_retries=max_retries, initial_backoff=initial_backoff)
    def _post():
        return session._post(url, data=data)

    return _post()


def get_with_retry(
    session, url: str, max_retries: int = 3, initial_backoff: float = 0.5
):
    from src.lib.retries import retry_with_backoff

    @retry_with_backoff(max_retries=max_retries, initial_backoff=initial_backoff)
    def _get():
        return session._get(url)

    return _get()


tastytrade_client = TastyTradeClient()
