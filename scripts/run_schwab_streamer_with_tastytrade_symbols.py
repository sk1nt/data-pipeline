#!/usr/bin/env python3
"""Start the Schwab streamer using the same symbols configured for TastyTrade."""
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config import settings
from src.services.schwab_streamer import build_streamer


def main(interactive: bool = True) -> None:
    streamer = build_streamer(use_tastytrade_symbols=True, interactive=interactive)
    print("Starting Schwab streamer with tastytrade symbols:", streamer.symbols)
    streamer.start()
    try:
        import time
        time.sleep(30)
    finally:
        streamer.stop()
        print("Stopped")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--interactive', action='store_true', default=False, help='Open browser for OAuth flow (default false)')
    args = ap.parse_args()
    main(interactive=args.interactive)
