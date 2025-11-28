#!/usr/bin/env python3
"""Start the Schwab streamer process.

This script encapsulates a minimal flow: ensure tokens exist (or prompt/exit),
then start the streamer with the chosen symbol set.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config import settings  # noqa: E402
from src.lib.logging import setup_logging  # noqa: E402
from src.services.schwab_streamer import build_streamer  # noqa: E402
from src.token_store import MissingBootstrapTokenError, TokenStoreError  # noqa: E402


def main(
    interactive: bool = False,
    tasty_symbols: bool = True,
    ensure_env: bool = False,
    persist_refresh_to_env: bool = False,
):
    setup_logging(level=settings.log_level, log_file=settings.log_file)
    env_path = PROJECT_ROOT / ".env"

    if not settings.schwab_client_id or not settings.schwab_client_secret:
        print(
            "SCHWAB_CLIENT_ID and SCHWAB_CLIENT_SECRET are required; set as env vars (export) or .env at:",
            env_path,
        )
        return 2
    if not settings.schwab_refresh_token:
        if interactive:
            print("Starting interactive OAuth flow...")
            # delegate to existing script that will persist token
            # The interactive flow with `build_streamer` will open the browser
        else:
            print(
                "No refresh token configured; run the manager to persist tokens or set SCHWAB_REFRESH_TOKEN in CI"
            )
            return 1

    try:
        streamer = build_streamer(
            use_tastytrade_symbols=tasty_symbols, interactive=interactive
        )
    except MissingBootstrapTokenError as exc:
        print(f"Missing Schwab bootstrap token: {exc}")
        print(
            "Run `python scripts/verify_token_refresh.py` or the exchange-url flow once, then retry."
        )
        return 1
    except TokenStoreError as exc:
        print(f"Failed to read/write Schwab tokens: {exc}")
        return 1
    print("Starting Schwab streamer for symbols:", ",".join(streamer.symbols))
    try:
        streamer.start()
        # Keep running until Ctrl-C
        import signal

        signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()
        print("Stopped streamer")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Run interactive login if needed",
    )
    ap.add_argument(
        "--no-tasty-symbols",
        dest="tasty_symbols",
        action="store_false",
        help="Do not use tastytrade symbol list",
    )
    args = ap.parse_args()
    raise SystemExit(
        main(interactive=args.interactive, tasty_symbols=args.tasty_symbols)
    )
