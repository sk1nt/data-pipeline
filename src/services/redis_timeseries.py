"""Helpers for interacting with RedisTimeSeries."""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import redis

from ..config import settings

LOGGER = logging.getLogger(__name__)


class RedisTimeSeriesClient:
    """Wrapper around RedisTimeSeries commands with auto-create support."""

    def __init__(self, client: redis.Redis) -> None:
        self.client = client

    def add_sample(self, key: str, timestamp_ms: int, value: float, labels: Dict[str, str]) -> None:
        try:
            self.client.execute_command("TS.ADD", key, timestamp_ms, value)
        except redis.ResponseError as exc:
            if "TSDB: key does not exist" not in str(exc):
                raise
            self._create_series(key, labels)
            self.client.execute_command("TS.ADD", key, timestamp_ms, value)

    def multi_add(self, samples: Iterable[Tuple[str, int, float, Dict[str, str]]]) -> None:
        records = list(samples)
        if not records:
            return
        pipeline = self.client.pipeline()
        for key, ts, value, _ in records:
            pipeline.execute_command("TS.ADD", key, ts, value)
        try:
            pipeline.execute()
        except redis.ResponseError:
            LOGGER.warning("RedisTimeSeries pipeline failed; retrying with auto-create")
            for key, ts, value, labels in records:
                self.add_sample(key, ts, value, labels)

    def range(self, key: str, start_ms: int, end_ms: int) -> List[Tuple[int, float]]:
        result = self.client.execute_command("TS.RANGE", key, start_ms, end_ms)
        return [(int(ts), float(val)) for ts, val in result]

    def mrange(
        self,
        filters: List[str],
        *,
        start: str = "-",
        end: str = "+",
        count: Optional[int] = None,
        with_labels: bool = True,
    ):
        args = ["TS.MRANGE", start, end]
        if count is not None:
            args.extend(["COUNT", count])
        if with_labels:
            args.append("WITHLABELS")
        args.append("FILTER")
        args.extend(filters)
        return self.client.execute_command(*args)

    def _create_series(self, key: str, labels: Dict[str, str]) -> None:
        args = ["TS.CREATE", key, "RETENTION", settings.redis_retention_ms, "LABELS"]
        for k, v in labels.items():
            args.extend([k, v])
        self.client.execute_command(*args)
