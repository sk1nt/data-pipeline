#!/usr/bin/env python3
"""Smoke-test the TastyTrade DXLink streamer without running the full pipeline."""
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config import settings
from src.services.tastytrade_streamer import StreamerSettings, TastyTradeStreamer

async def main() -> None:
    streamer = TastyTradeStreamer(
        StreamerSettings(
            client_id=settings.tastytrade_client_id or "",
            client_secret=settings.tastytrade_client_secret or "",
            refresh_token=settings.tastytrade_refresh_token or "",
            symbols=settings.tastytrade_symbol_list,
            depth_levels=settings.tastytrade_depth_levels,
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
