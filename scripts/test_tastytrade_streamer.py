#!/usr/bin/env python3
"""Smoke-test the TastyTrade DXLink streamer without running the full pipeline."""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_tastytrade_dependencies():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    src_path = PROJECT_ROOT / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from src.config import settings as cfg
    from src.services.tastytrade_streamer import StreamerSettings, TastyTradeStreamer

    return cfg, StreamerSettings, TastyTradeStreamer


settings, StreamerSettings, TastyTradeStreamer = _load_tastytrade_dependencies()


async def main() -> None:
    streamer = TastyTradeStreamer(
        StreamerSettings(
            client_id=settings.tastytrade_client_id or "",
            client_secret=settings.tastytrade_client_secret or "",
            refresh_token=settings.tastytrade_refresh_token or "",
            symbols=settings.tastytrade_symbol_list,
            depth_levels=settings.tastytrade_depth_cap,
        )
    )
    streamer.start()
    print("Streamer started; running for 30 seconds...")
    try:
        await asyncio.sleep(30)
    finally:
        await streamer.stop()
        print("Streamer stopped")


if __name__ == "__main__":
    asyncio.run(main())
