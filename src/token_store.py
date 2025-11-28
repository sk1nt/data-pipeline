"""Utilities for persisting Schwab tokens in `.tokens` and refreshing them headlessly."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from schwab import auth

LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENS_DIR = PROJECT_ROOT / ".tokens"
SCHWAB_TOKEN_FILENAME = "schwab_token.json"
ACCESS_SNAPSHOT_FILENAME = "access_token.json"
REFRESH_SNAPSHOT_FILENAME = "refresh_token.json"


@dataclass
class TokenRefreshResult:
    """Return payload when a refresh occurs."""

    access_token_path: Path
    refresh_token_path: Path
    tokens_path: Path
    expires_at: datetime
    issued_at: datetime


class TokenStoreError(RuntimeError):
    """Base error for token-store issues."""


class MissingBootstrapTokenError(TokenStoreError):
    """Raised when no persisted refresh token file exists."""


class TokenRefreshError(TokenStoreError):
    """Raised when auto-refresh fails."""


class TokenStore:
    """Handles Schwab token persistence in `.tokens`."""

    def __init__(self, tokens_dir: Path | None = None):
        self.tokens_dir = tokens_dir or DEFAULT_TOKENS_DIR
        self.tokens_dir.mkdir(parents=True, exist_ok=True)
        self.tokens_path = self.tokens_dir / SCHWAB_TOKEN_FILENAME
        self.access_snapshot_path = self.tokens_dir / ACCESS_SNAPSHOT_FILENAME
        self.refresh_snapshot_path = self.tokens_dir / REFRESH_SNAPSHOT_FILENAME

    # Public helpers -----------------------------------------------------------------
    def has_bootstrap_token(self) -> bool:
        """Return True if we have a persisted refresh-capable token file."""
        data = self._load_tokens_file()
        token = self._extract_token_payload(data)
        return bool(token and token.get("refresh_token"))

    def ensure_metadata_format(self) -> Optional[dict]:
        """Ensure the Schwab token file matches the metadata format Schwab expects."""
        data = self._load_tokens_file()
        if data is None:
            return None
        if "creation_timestamp" in data and "token" in data:
            return data
        token_payload = self._extract_token_payload(data)
        if not token_payload:
            return None
        creation_ts = int(
            data.get("creation_timestamp") or data.get("updated_at") or time.time()
        )
        upgraded = {
            "creation_timestamp": creation_ts,
            "token": token_payload,
        }
        self._atomic_write(self.tokens_path, upgraded)
        LOG.info("Upgraded Schwab token file to metadata format expected by schwab-py.")
        return upgraded

    def persist_snapshots(
        self, token_payload: Dict[str, Any], issued_at: datetime | None = None
    ) -> None:
        """Write convenience JSON snapshots for access/refresh tokens."""
        token_payload = self._extract_token_payload(token_payload) or token_payload
        issued_at = issued_at or datetime.now(timezone.utc)
        expires_at = issued_at + timedelta(
            seconds=int(token_payload.get("expires_in") or 0)
        )
        access_snapshot = {
            "token": token_payload.get("access_token"),
            "token_type": token_payload.get("token_type", "Bearer"),
            "scope": token_payload.get("scope"),
            "issued_at": issued_at.isoformat(),
            "expires_at": expires_at.isoformat(),
        }
        refresh_snapshot = {
            "token": token_payload.get("refresh_token"),
            "rotated_at": issued_at.isoformat(),
        }
        self._atomic_write(self.access_snapshot_path, access_snapshot)
        self._atomic_write(self.refresh_snapshot_path, refresh_snapshot)
        LOG.debug(
            "Wrote snapshots: %s, %s",
            self.access_snapshot_path,
            self.refresh_snapshot_path,
        )

    def write_token_payload(
        self, token_payload: Dict[str, Any], issued_at: datetime | None = None
    ) -> dict:
        """Persist the canonical Schwab token file in metadata format."""
        token_payload = self._extract_token_payload(token_payload) or token_payload
        issued_at = issued_at or datetime.now(timezone.utc)
        metadata = self._load_tokens_file() or {}
        creation_ts = int(
            metadata.get("creation_timestamp")
            or metadata.get("updated_at")
            or time.time()
        )
        wrapped = {
            "creation_timestamp": creation_ts,
            "token": token_payload,
        }
        self._atomic_write(self.tokens_path, wrapped)
        self.persist_snapshots(token_payload, issued_at=issued_at)
        return wrapped

    # Internal helpers ----------------------------------------------------------------
    def _load_tokens_file(self) -> Optional[dict]:
        """Load the raw Schwab token JSON."""
        if not self.tokens_path.exists():
            return None
        try:
            with open(self.tokens_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:  # pragma: no cover - defensive
            raise TokenStoreError(f"Unable to load Schwab token file: {exc}") from exc

    def _extract_token_payload(self, data: Optional[dict]) -> Optional[dict]:
        """Return the innermost token dict containing refresh_token/scope data."""
        current = data
        depth = 0
        while isinstance(current, dict) and depth < 10:
            if current.get("refresh_token"):
                return current
            if "token" in current and isinstance(current["token"], dict):
                current = current["token"]
                depth += 1
                continue
            if "tokens" in current and isinstance(current["tokens"], dict):
                current = current["tokens"]
                depth += 1
                continue
            break
        return current if isinstance(current, dict) else None

    def _atomic_write(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        tmp.replace(path)


def refresh_and_cache_tokens(
    *,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    token_url: str,
    scope: Optional[str] = None,
    tokens_dir: Path | None = None,
) -> TokenRefreshResult:
    """Refresh Schwab tokens headlessly and update `.tokens` snapshots."""
    store = TokenStore(tokens_dir)
    if not store.tokens_path.exists():
        raise MissingBootstrapTokenError(
            f"No Schwab token file found at {store.tokens_path}. Run the interactive bootstrap once."
        )

    metadata = store.ensure_metadata_format()
    if metadata is None or not store.has_bootstrap_token():
        raise MissingBootstrapTokenError(
            f"Schwab token file {store.tokens_path} is missing a refresh token. Re-run the interactive bootstrap."
        )

    try:
        schwab_client = auth.easy_client(
            api_key=client_id,
            app_secret=client_secret,
            callback_url=redirect_uri,
            token_path=str(store.tokens_path),
            interactive=False,
            max_token_age=None,
        )
    except ValueError as exc:
        raise MissingBootstrapTokenError(
            "Existing Schwab token file is invalid or legacy formatted. "
            "Delete it and run the interactive bootstrap again."
        ) from exc
    except Exception as exc:  # pragma: no cover - network/lib failure
        raise TokenRefreshError(f"Unable to initialize Schwab client: {exc}") from exc

    issued_at = datetime.now(timezone.utc)
    try:
        refresh_kwargs = {"scope": scope} if scope else {}
        token_payload = schwab_client.session.refresh_token(token_url, **refresh_kwargs)
    except Exception as exc:  # pragma: no cover - network/lib failure
        raise TokenRefreshError(f"Schwab refresh_token call failed: {exc}") from exc

    store.write_token_payload(token_payload, issued_at=issued_at)
    expires_at = issued_at + timedelta(
        seconds=int(token_payload.get("expires_in") or 0)
    )

    return TokenRefreshResult(
        access_token_path=store.access_snapshot_path,
        refresh_token_path=store.refresh_snapshot_path,
        tokens_path=store.tokens_path,
        expires_at=expires_at,
        issued_at=issued_at,
    )


__all__ = [
    "TokenStore",
    "TokenStoreError",
    "MissingBootstrapTokenError",
    "TokenRefreshError",
    "TokenRefreshResult",
    "refresh_and_cache_tokens",
]
