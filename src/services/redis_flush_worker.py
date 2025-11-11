"""Background worker that flushes RedisTimeSeries data to DuckDB/Parquet."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import duckdb
import pandas as pd
import redis

from ..config import settings
from ..lib.redis_client import RedisClient
from .redis_timeseries import RedisTimeSeriesClient

LOGGER = logging.getLogger(__name__)


@dataclass
class FlushWorkerSettings:
    interval_seconds: int = settings.flush_interval_seconds
    key_pattern: str = "ts:*"
    last_hash: str = "ts:last_flushed"
    db_path: Path = Path(settings.timeseries_db_path)
    parquet_dir: Path = Path(settings.timeseries_parquet_dir)


class RedisFlushWorker:
    def __init__(
        self,
        redis_client: RedisClient,
        ts_client: RedisTimeSeriesClient,
        settings: FlushWorkerSettings = FlushWorkerSettings(),
    ) -> None:
        self.redis_client = redis_client
        self.ts_client = ts_client
        self.settings = settings
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self._last_summary: Dict[str, Any] = {}

    def start(self) -> None:
        if self._task and not self._task.done():
            LOGGER.warning("Redis flush worker already running")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="redis-flush-worker")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task:
            await self._task
            self._task = None

    async def _run(self) -> None:
        LOGGER.info("Redis flush worker started (interval=%ss)", self.settings.interval_seconds)
        while not self._stop_event.is_set():
            await self._flush_once()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.settings.interval_seconds)
            except asyncio.TimeoutError:
                continue
        LOGGER.info("Redis flush worker stopped")

    async def _flush_once(self) -> None:
        try:
            await asyncio.to_thread(self._flush_sync)
        except Exception:
            LOGGER.exception("Redis flush worker encountered an error")

    def _flush_sync(self) -> None:
        start_time = time.perf_counter()
        keys = list(self.redis_client.client.scan_iter(match=self.settings.key_pattern))
        if not keys:
            self._last_summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "samples": 0,
                "keys": 0,
                "duration": 0.0,
            }
            return
        last_hash = self.settings.last_hash
        new_records: List[Tuple[str, int, float]] = []
        last_updates = {}
        for key in keys:
            last_ts = self.redis_client.client.hget(last_hash, key)
            start = int(last_ts) + 1 if last_ts is not None else 0
            samples = self.ts_client.range(key, start, "+")
            if not samples:
                continue
            new_records.extend((key, ts, value) for ts, value in samples)
            last_updates[key] = samples[-1][0]
        if not new_records:
            duration = time.perf_counter() - start_time
            self._last_summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "samples": 0,
                "keys": 0,
                "duration": duration,
            }
            return
        df = pd.DataFrame(new_records, columns=["key", "ts", "value"])
        df["day"] = pd.to_datetime(df["ts"], unit="ms").dt.date
        self._write_to_duckdb(df)
        self._write_parquet(df)
        if last_updates:
            self.redis_client.client.hset(last_hash, mapping=last_updates)
        duration = time.perf_counter() - start_time
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "samples": int(len(df)),
            "keys": int(len(last_updates)),
            "duration": duration,
        }
        self._last_summary = summary
        LOGGER.info(
            "Redis flush worker: %s samples from %s keys in %.2fs",
            summary["samples"],
            summary["keys"],
            summary["duration"],
        )

    def _write_to_duckdb(self, df: pd.DataFrame) -> None:
        self.settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(self.settings.db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS redis_timeseries (
                key VARCHAR,
                ts BIGINT,
                value DOUBLE,
                day DATE
            )
            """
        )
        conn.register("flush_df", df)
        conn.execute("INSERT INTO redis_timeseries SELECT key, ts, value, day FROM flush_df")
        conn.close()

    def _write_parquet(self, df: pd.DataFrame) -> None:
        base_dir = self.settings.parquet_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        flush_ts = int(datetime.utcnow().timestamp())
        for day, group in df.groupby("day"):
            day_dir = base_dir / str(day)
            day_dir.mkdir(parents=True, exist_ok=True)
            filename = day_dir / f"flush_{flush_ts}.parquet"
            group.to_parquet(filename, index=False)

    def status(self) -> Dict[str, Any]:
        running = self._task is not None and not self._task.done()
        summary = dict(self._last_summary)
        summary.setdefault("running", running)
        summary.setdefault("samples", 0)
        summary.setdefault("keys", 0)
        summary.setdefault("duration", 0.0)
        summary.setdefault("timestamp", None)
        return summary
*** End Patch
PATCH
