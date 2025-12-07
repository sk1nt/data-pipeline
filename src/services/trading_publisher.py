"""Publish normalized market data events into the trading system."""

from __future__ import annotations

import json
from typing import Optional

from lib.logging import get_logger
from lib.redis_client import RedisClient
from ..models.market_data import TickEvent, Level2Event

LOG = get_logger(__name__)


class TradingEventPublisher:
    """Lightweight abstraction for streaming events into Redis-backed trading bus."""

    def __init__(
        self,
        redis_client: Optional[RedisClient] = None,
        tick_channel: str = "market_data:ticks",
        level2_channel: str = "market_data:level2",
    ):
        self.redis = redis_client or RedisClient()
        self.tick_channel = tick_channel
        self.level2_channel = level2_channel

    def publish_tick(self, event: TickEvent) -> bool:
        """Publish tick event."""
        try:
            payload = json.dumps(event.to_payload())
            self.redis.client.publish(self.tick_channel, payload)
            LOG.debug("Published tick %s ts=%s", event.symbol, event.timestamp)
            return True
        except Exception as exc:
            LOG.error("Failed to publish tick %s: %s", event.symbol, exc)
            return False

    def publish_level2(self, event: Level2Event) -> bool:
        """Publish level 2 snapshot."""
        try:
            payload = json.dumps(event.to_payload())
            self.redis.client.publish(self.level2_channel, payload)
            LOG.debug("Published level2 %s ts=%s", event.symbol, event.timestamp)
            return True
        except Exception as exc:
            LOG.error("Failed to publish level2 %s: %s", event.symbol, exc)
            return False
