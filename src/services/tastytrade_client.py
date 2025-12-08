from typing import Optional
from datetime import datetime, timezone
from tastytrade import Session
from tastytrade.utils import TastytradeError
from config.settings import config


class TastytradeAuthError(RuntimeError):
    pass


class TastyTradeClient:
    def __init__(self):
        self._session: Optional[Session] = None
        self._session_expiration: Optional[datetime] = None

    def set_refresh_token(self, refresh_token: str) -> None:
        """Update the refresh token used for new sessions and clear any cached session."""
        token = (refresh_token or "").strip()
        if not token:
            raise ValueError("refresh_token must be provided")
        # Persist onto config so future calls use the updated token
        if config.tastytrade_use_sandbox:
            config.tastytrade_refresh_token = token
        else:
            config.tastytrade_prod_refresh_token = token
        # Clear cached session so the next call reinitializes with the new token
        self._session = None
        self._session_expiration = None

    def _ensure_session(self) -> Session:
        """Ensure we have a valid session, refresh if needed."""
        if self._session is None or (
            self._session_expiration
            and datetime.now(timezone.utc) >= self._session_expiration
        ):
            try:
                self._session = Session(
                    provider_secret=config.effective_tastytrade_client_secret,
                    refresh_token=config.effective_tastytrade_refresh_token,
                )
            except Exception as exc:
                # Map SDK auth failures to TastytradeAuthError for callers
                raise TastytradeAuthError(
                    f"TastyTrade authentication failed: {exc}"
                ) from exc
            # Assume session has expiration, or set a default
            self._session_expiration = datetime.now(
                timezone.utc
            )  # TODO: get from session

        return self._session

    def get_session(self) -> Session:
        """Get the current session."""
        return self._ensure_session()

    def refresh_session(self):
        """Force refresh the session."""
        if self._session:
            try:
                self._session.refresh()
            except Exception:
                # ignore refresh failures; next ensure will reauth
                self._session = None
                self._session_expiration = None
            else:
                self._session_expiration = datetime.now(timezone.utc)

    def ensure_authorized(self) -> bool:
        """Attempt to ensure session is valid; raise TastytradeAuthError if not."""
        try:
            self._ensure_session()
            return True
        except TastytradeAuthError:
            raise
        except Exception as exc:
            # Wrap other errors as auth errors for callers
            raise TastytradeAuthError(f"TastyTrade authorization error: {exc}") from exc


def post_with_retry(session, url: str, data: str, max_retries: int = 3, initial_backoff: float = 0.5):
    """Post with retry on transient errors using retry_with_backoff."""
    from src.lib.retries import retry_with_backoff

    @retry_with_backoff(max_retries=max_retries, initial_backoff=initial_backoff)
    def _post():
        return session._post(url, data=data)

    return _post()


def get_with_retry(session, url: str, max_retries: int = 3, initial_backoff: float = 0.5):
    from src.lib.retries import retry_with_backoff

    @retry_with_backoff(max_retries=max_retries, initial_backoff=initial_backoff)
    def _get():
        return session._get(url)

    return _get()
    


# Global client instance
tastytrade_client = TastyTradeClient()
