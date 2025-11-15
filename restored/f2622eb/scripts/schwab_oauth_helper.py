#!/usr/bin/env python
"""
Command-line helper to complete Schwab's OAuth authorization_code flow.

Steps:
1. Reads Schwab client credentials + redirect URI from the shared settings.
2. Prints (and optionally opens) the Schwab authorization URL.
3. Prompts for the ?code value returned after completing consent.
4. Exchanges the code for access + refresh tokens and persists them to .env.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import textwrap
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
from dotenv import set_key

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.lib.logging import get_logger

LOG = get_logger(__name__)


def build_authorize_url(
    client_id: str,
    redirect_uri: str,
    scope: str,
    auth_url: str,
    state: str,
) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
    }
    return f"{auth_url}?{urlencode(params)}"


def exchange_code_for_tokens(
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    token_url: str,
    code_verifier: str | None = None,
) -> dict[str, Any]:
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }
    if code_verifier:
        data["code_verifier"] = code_verifier

    auth = httpx.BasicAuth(client_id, client_secret)
    resp = httpx.post(
        token_url,
        data=data,
        auth=auth,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def persist_to_env(env_path: Path, values: dict[str, str]) -> None:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    for key, value in values.items():
        if value:
            set_key(str(env_path), key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Schwab OAuth helper (authorization_code flow)",
    )
    parser.add_argument(
        "--code",
        help="Authorization code returned by Schwab (if omitted, the script prompts interactively).",
    )
    parser.add_argument(
        "--code-verifier",
        help="Optional PKCE code_verifier if your Schwab app enforces PKCE.",
    )
    parser.add_argument(
        "--scope",
        help="Override OAuth scope passed to Schwab (defaults to SCHWAB_SCOPE env).",
    )
    parser.add_argument(
        "--redirect-uri",
        help="Override redirect URI (defaults to SCHWAB_REDIRECT_URI env).",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the env file to update with the refresh token (default: %(default)s).",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path to dump the full token payload as JSON for auditing.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not attempt to open the authorization URL in the default browser.",
    )
    parser.add_argument(
        "--no-write-env",
        action="store_true",
        help="Skip writing SCHWAB_REFRESH_TOKEN to the env file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the authorization URL; skip prompting for codes or exchanging tokens.",
    )
    return parser.parse_args()


def ensure_required_settings() -> None:
    missing = []
    if not settings.schwab_client_id:
        missing.append("SCHWAB_CLIENT_ID")
    if not settings.schwab_client_secret:
        missing.append("SCHWAB_CLIENT_SECRET")
    if not settings.schwab_redirect_uri:
        missing.append("SCHWAB_REDIRECT_URI")
    if missing:
        raise SystemExit(
            "Missing required env vars: "
            + ", ".join(missing)
            + ". Update .env and retry.",
        )


def main() -> None:
    ensure_required_settings()
    args = parse_args()

    scope = args.scope or settings.schwab_scope
    redirect_uri = args.redirect_uri or settings.schwab_redirect_uri
    env_path = Path(args.env_file).expanduser().resolve()

    state = secrets.token_urlsafe(16)
    authorize_url = build_authorize_url(
        client_id=settings.schwab_client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        auth_url=settings.schwab_auth_url,
        state=state,
    )

    print(
        textwrap.dedent(
            f"""
            1. Open the Schwab consent URL below in a browser.
            2. Log in, complete MFA, and approve access for this client.
            3. After Schwab redirects to {redirect_uri}, copy the `code` query parameter.
            4. Paste that code back into this script.

            State token (for verification): {state}
            Authorization URL:
            {authorize_url}
            """,
        ).strip(),
    )
    if not args.no_browser:
        try:
            webbrowser.open(authorize_url, new=2)
        except Exception as exc:  # pragma: no cover - best-effort hint
            LOG.warning("Unable to open browser automatically: %s", exc)

    if args.dry_run:
        print("\nDry run complete; no token exchange performed.")
        return

    code = (args.code or input("\nPaste Schwab authorization code: ")).strip()
    if not code:
        raise SystemExit("Authorization code is required to continue.")

    try:
        token_payload = exchange_code_for_tokens(
            code=code,
            client_id=settings.schwab_client_id,
            client_secret=settings.schwab_client_secret,
            redirect_uri=redirect_uri,
            token_url=settings.schwab_token_url,
            code_verifier=args.code_verifier,
        )
    except httpx.HTTPError as exc:
        raise SystemExit(f"Token exchange failed: {exc}") from exc

    refresh_token = token_payload.get("refresh_token")
    access_token = token_payload.get("access_token")
    expires_in = token_payload.get("expires_in")

    if not refresh_token:
        print(json.dumps(token_payload, indent=2))
        raise SystemExit("Schwab response did not include a refresh_token.")

    if not args.no_write_env:
        persist_to_env(
            env_path,
            {
                "SCHWAB_REFRESH_TOKEN": refresh_token,
                "SCHWAB_ACCESS_TOKEN": access_token or "",
            },
        )
        print(f"\nUpdated {env_path} with SCHWAB_REFRESH_TOKEN.")

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(token_payload, indent=2))
        print(f"Wrote full token payload to {out_path}")

    print(
        "\nToken exchange complete.",
        f"Access token expires in ~{expires_in} seconds." if expires_in else "",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
