#!/usr/bin/env python3
"""Force the default GEX poller to take snapshots for its current symbol list."""
# ruff: noqa: E402
import asyncio
import os
import sys
from pathlib import Path

import aiohttp

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lib.redis_client import RedisClient
from src.services.gexbot_poller import GEXBotPoller, GEXBotPollerSettings
from src.services.redis_timeseries import RedisTimeSeriesClient

async def main():
    # Use the same settings as the default gex_poller
    api_key = os.getenv("GEXBOT_API_KEY")
    if not api_key:
        print("GEXBOT_API_KEY not set")
        return

    settings = GEXBotPollerSettings(
        api_key=api_key,
        symbols=["NQ_NDX", "ES_SPX", "SPY", "QQQ", "SPX", "NDX"],
        interval_seconds=60.0,
        aggregation_period="zero",
        rth_interval_seconds=1.0,
        off_hours_interval_seconds=300.0,
        dynamic_schedule=True,
        exclude_symbols=[],
        sierra_chart_output_path=None,
    )

    redis_client = RedisClient()
    ts_client = RedisTimeSeriesClient(redis_client.client)

    poller = GEXBotPoller(
        settings,
        redis_client=redis_client,
        ts_client=ts_client,
    )

    # Get the current symbol list (this will refresh supported symbols)
    timeout = aiohttp.ClientTimeout(total=12)
    connector = aiohttp.TCPConnector(limit=8, force_close=True)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        await poller._refresh_supported_symbols(session)
        symbols = sorted(poller._supported_symbols or poller._base_symbols)
        print(f"Forcing snapshots for symbols: {symbols}")

        for symbol in symbols:
            print(f"Fetching {symbol}...")
            snapshot = await poller.fetch_symbol_now(symbol)
            if snapshot:
                print(f"  -> Snapshot taken for {symbol}")
            else:
                print(f"  -> Failed to fetch {symbol}")

if __name__ == "__main__":
    asyncio.run(main())
