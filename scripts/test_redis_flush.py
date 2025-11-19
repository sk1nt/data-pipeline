#!/usr/bin/env python3
"""Exercise the Redis flush worker with synthetic data."""
import asyncio
import random
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_flush_dependencies():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    src_path = PROJECT_ROOT / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from src.config import settings as cfg
    from src.lib.redis_client import RedisClient
    from src.services.redis_flush_worker import FlushWorkerSettings, RedisFlushWorker
    from src.services.redis_timeseries import RedisTimeSeriesClient

    return cfg, RedisClient, RedisTimeSeriesClient, FlushWorkerSettings, RedisFlushWorker


settings, RedisClient, RedisTimeSeriesClient, FlushWorkerSettings, RedisFlushWorker = _load_flush_dependencies()

async def main() -> None:
    redis_client = RedisClient(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password,
    )
    ts_client = RedisTimeSeriesClient(redis_client.client)

    # Seed some fake data
    now_ms = int(datetime.utcnow().timestamp() * 1000)
    samples = []
    for i in range(20):
        key = f"ts:test:{i % 3}"
        samples.append((key, now_ms + i * 1000, random.random() * 10, {"test": "true"}))
    ts_client.multi_add(samples)

    worker = RedisFlushWorker(redis_client, ts_client, FlushWorkerSettings(interval_seconds=5))
    worker.start()
    print("Flush worker running for 15 seconds...")
    await asyncio.sleep(15)
    await worker.stop()
    print("Summary:", worker.status())

if __name__ == "__main__":
    asyncio.run(main())
