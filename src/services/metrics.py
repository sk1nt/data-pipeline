from __future__ import annotations

import logging
from typing import Dict
from src.lib.redis_client import get_redis_client

logger = logging.getLogger(__name__)


class MetricsCollector:
    def __init__(self):
        self._counters: Dict[str, int] = {}

    def incr(self, key: str, amount: int = 1) -> None:
        try:
            rc = get_redis_client().client
            rc.incrby(f"metrics:{key}", amount)
        except Exception:
            # Fallback to local counters for tests if Redis not available
            self._counters[key] = self._counters.get(key, 0) + amount

    def get(self, key: str) -> int:
        return self._counters.get(key, 0)


metrics = MetricsCollector()
