#!/usr/bin/env python3
"""CLI entrypoint for the Schwab streaming service."""

from __future__ import annotations

import argparse
import signal
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.services.schwab_streamer import build_streamer
from src.lib.logging import get_logger

LOG = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Schwab → trading system streamer")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build streamer without connecting (useful for validation)",
    )
    args = parser.parse_args()

    if not settings.schwab_enabled:
        raise SystemExit("Schwab streaming disabled (set SCHWAB_ENABLED=true in env)")

    streamer = build_streamer()

    if args.dry_run:
        LOG.info("Schwab streamer dry-run succeeded (no connection attempted)")
        return

    def _stop(signum, _frame):
        LOG.info("Received signal %s; shutting down Schwab streamer", signum)
        streamer.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    LOG.info("Starting Schwab streamer service")
    streamer.start()


if __name__ == "__main__":
    main()
