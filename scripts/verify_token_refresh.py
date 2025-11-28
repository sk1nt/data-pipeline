#!/usr/bin/env python3
"""Headless verifier for Schwab token refresh flow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config import settings  # noqa: E402
from src.token_store import (  # noqa: E402
    MissingBootstrapTokenError,
    TokenRefreshError,
    refresh_and_cache_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headlessly refresh Schwab tokens once."
    )
    parser.add_argument(
        "--tokens-dir",
        default=PROJECT_ROOT / ".tokens",
        type=Path,
        help="Directory containing schwab_token.json (default: %(default)s)",
    )
    parser.add_argument(
        "--scope",
        default="api",
        help="Optional scope for the refresh request (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required = {
        "SCHWAB_CLIENT_ID": settings.schwab_client_id,
        "SCHWAB_CLIENT_SECRET": settings.schwab_client_secret,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        print(f"[ERROR] Missing required Schwab credentials: {', '.join(missing)}")
        return 2

    redirect_uri = settings.schwab_redirect_uri or "https://127.0.0.1:8182"
    token_url = settings.schwab_token_url

    try:
        result = refresh_and_cache_tokens(
            client_id=settings.schwab_client_id,
            client_secret=settings.schwab_client_secret,
            redirect_uri=redirect_uri,
            token_url=token_url,
            scope=args.scope,
            tokens_dir=args.tokens_dir,
        )
    except MissingBootstrapTokenError as exc:
        print(f"[ERROR] {exc}")
        print(
            "Run the one-time interactive login (e.g. scripts/schwab_token_manager.py exchange-url) first."
        )
        return 3
    except TokenRefreshError as exc:
        print(f"[ERROR] Token refresh failed: {exc}")
        return 4
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[ERROR] Unexpected failure: {exc}")
        return 5

    print("[OK] Schwab tokens refreshed headlessly.")
    print(f"Access token snapshot: {result.access_token_path}")
    print(f"Refresh token snapshot: {result.refresh_token_path}")
    print(f"Expires at: {result.expires_at.isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
