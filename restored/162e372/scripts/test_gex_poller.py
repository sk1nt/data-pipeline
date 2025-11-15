#!/usr/bin/env python3
"""Smoke-test the GEXBot poller independently."""
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config import settings
from src.lib.redis_client import RedisClient
from src.services.redis_timeseries import RedisTimeSeriesClient
from src.services.gexbot_poller import GEXBotPoller, GEXBotPollerSettings

async def main() -> None:
    redis_client = RedisClient(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
    )
    ts_client = RedisTimeSeriesClient(redis_client.client)
    poller = GEXBotPoller(
        GEXBotPollerSettings(
            api_key=settings.gexbot_api_key or "",
            symbols=settings.gex_symbol_list,
            interval_seconds=10,
        ),
        redis_client=redis_client,
        ts_client=ts_client,
    )
    poller.start()
    print("GEX poller running for 60 seconds...")
    try:
        await asyncio.sleep(60)
    finally:
        await poller.stop()
        print("GEX poller stopped")

if __name__ == "__main__":
    asyncio.run(main())
