"""Service for processing and storing Unusual Whales websocket messages."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.lib.redis_client import RedisClient
from src.models.uw_message import (
    MarketAggMessage,
    OptionTradeMessage,
    UWMessage,
    parse_uw_websocket_message,
)

LOGGER = logging.getLogger(__name__)

# Lazy import to avoid circular dependency
_alert_service = None


class UWMessageService:
    """Service for processing UW websocket messages, storing in Redis, and routing notifications."""

    # Redis key prefixes
    MARKET_AGG_LATEST_KEY = "uw:market_agg:latest"
    MARKET_AGG_HISTORY_KEY = "uw:market_agg:history"
    OPTION_TRADE_LATEST_KEY = "uw:option_trade:latest"
    OPTION_TRADE_HISTORY_KEY = "uw:option_trade:history"
    OPTION_TRADE_BY_SYMBOL_KEY = "uw:option_trade:symbol:{symbol}"

    # Redis pubsub channels
    MARKET_AGG_CHANNEL = "uw:market_agg:stream"
    OPTION_TRADE_CHANNEL = "uw:option_trade:stream"

    # TTL and limits
    CACHE_TTL_SECONDS = 86400  # 24 hours
    HISTORY_LIMIT = 1000  # Keep last 1000 messages per type

    def __init__(self, redis_client: RedisClient):
        """Initialize the service with a Redis client."""
        self.redis_client = redis_client

    def process_raw_message(self, raw: List[Any]) -> Optional[Dict[str, Any]]:
        """Process a raw UW websocket message.

        Args:
            raw: Raw websocket message as list

        Returns:
            Dictionary with processing result and metadata, or None if parsing fails
        """
        try:
            message = parse_uw_websocket_message(raw)
            return self._route_message(message)
        except Exception as e:
            LOGGER.error("Failed to parse UW message: %s", e, exc_info=True)
            return None

    def _route_message(self, message: UWMessage) -> Dict[str, Any]:
        """Route parsed message to appropriate handler."""
        if isinstance(message, MarketAggMessage):
            return self._handle_market_agg(message)
        elif isinstance(message, OptionTradeMessage):
            return self._handle_option_trade(message)
        else:
            LOGGER.warning("Unknown message type: %s", type(message))
            return {"status": "error", "reason": "unknown_message_type"}

    def _handle_market_agg(self, message: MarketAggMessage) -> Dict[str, Any]:
        """Handle market aggregation message."""
        now = datetime.now(timezone.utc)
        stamped = {
            "received_at": now.isoformat(),
            "message_type": message.message_type,
            "topic": message.topic,
            "data": message.data.model_dump(),
        }

        serialized = json.dumps(stamped, default=str)
        conn = self.redis_client.client

        # Store latest + history
        pipe = conn.pipeline()
        pipe.setex(self.MARKET_AGG_LATEST_KEY, self.CACHE_TTL_SECONDS, serialized)
        pipe.lpush(self.MARKET_AGG_HISTORY_KEY, serialized)
        pipe.ltrim(self.MARKET_AGG_HISTORY_KEY, 0, self.HISTORY_LIMIT - 1)
        pipe.execute()

        # Publish to stream for subscribers
        try:
            conn.publish(self.MARKET_AGG_CHANNEL, serialized)
        except Exception as e:
            LOGGER.exception("Failed to publish market_agg to Redis stream: %s", e)

        LOGGER.debug("Stored market_agg message for date %s", message.data.date)

        # Check for alerts on ratio shifts
        alert_triggered = False
        try:
            alert_service = self._get_alert_service()
            alert = alert_service.process_market_agg_update(message.data.model_dump())
            if alert:
                alert_triggered = True
                LOGGER.info("Market agg alert triggered: %s", alert.get("alert_type"))
        except Exception as e:
            LOGGER.exception("Failed to process market agg alerts: %s", e)

        return {
            "status": "success",
            "message_type": "market_agg",
            "date": message.data.date,
            "discord_notification": alert_triggered,
            "alert_triggered": alert_triggered,
        }
    
    def _get_alert_service(self):
        """Lazy load alert service to avoid circular imports."""
        global _alert_service
        if _alert_service is None:
            from src.services.market_agg_alert_service import MarketAggAlertService
            _alert_service = MarketAggAlertService(self.redis_client)
        return _alert_service

    def _handle_option_trade(self, message: OptionTradeMessage) -> Dict[str, Any]:
        """Handle option trade message."""
        now = datetime.now(timezone.utc)
        
        # Determine Discord notification routing
        discord_channel = self._get_discord_channel_for_trade(message)
        
        stamped = {
            "received_at": now.isoformat(),
            "message_type": message.message_type,
            "topic": message.topic,
            "topic_symbol": message.topic_symbol,
            "data": message.data.model_dump(),
            "discord_channel_id": discord_channel,  # Add channel routing
        }

        serialized = json.dumps(stamped, default=str)
        conn = self.redis_client.client

        # Store latest + history
        pipe = conn.pipeline()
        pipe.setex(self.OPTION_TRADE_LATEST_KEY, self.CACHE_TTL_SECONDS, serialized)
        pipe.lpush(self.OPTION_TRADE_HISTORY_KEY, serialized)
        pipe.ltrim(self.OPTION_TRADE_HISTORY_KEY, 0, self.HISTORY_LIMIT - 1)

        # Store by symbol for easy lookup
        if message.topic_symbol:
            symbol_key = self.OPTION_TRADE_BY_SYMBOL_KEY.format(
                symbol=message.topic_symbol.upper()
            )
            pipe.setex(symbol_key, self.CACHE_TTL_SECONDS, serialized)

        pipe.execute()

        # Publish to stream for subscribers
        try:
            conn.publish(self.OPTION_TRADE_CHANNEL, serialized)
        except Exception as e:
            LOGGER.exception("Failed to publish option_trade to Redis stream: %s", e)

        LOGGER.debug(
            "Stored option_trade message: %s %s @ %s",
            message.topic_symbol or "UNKNOWN",
            message.data.option_chain_id,
            message.data.price,
        )

        # Determine Discord notification routing
        discord_channel = self._get_discord_channel_for_trade(message)

        return {
            "status": "success",
            "message_type": "option_trade",
            "symbol": message.topic_symbol,
            "option_chain_id": message.data.option_chain_id,
            "is_index": message.data.is_index_option,
            "discord_notification": discord_channel is not None,
            "discord_channel_id": discord_channel,
        }

    def _get_discord_channel_for_trade(
        self, message: OptionTradeMessage
    ) -> Optional[int]:
        """Determine which Discord channel should receive this trade notification.

        Returns:
            Channel ID or None if no notification needed
        """
        # SPX trades go to dedicated channel
        if message.topic_symbol == "SPX":
            return 1429940127899324487

        # All other option trades go to general channel
        return 1425136266676146236

    def get_latest_market_agg(self) -> Optional[Dict[str, Any]]:
        """Retrieve the latest market aggregation data."""
        raw = self.redis_client.client.get(self.MARKET_AGG_LATEST_KEY)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception as e:
            LOGGER.error("Failed to parse market_agg from Redis: %s", e)
            return None

    def get_latest_option_trade(
        self, symbol: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve the latest option trade, optionally filtered by symbol."""
        if symbol:
            key = self.OPTION_TRADE_BY_SYMBOL_KEY.format(symbol=symbol.upper())
        else:
            key = self.OPTION_TRADE_LATEST_KEY

        raw = self.redis_client.client.get(key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception as e:
            LOGGER.error("Failed to parse option_trade from Redis: %s", e)
            return None

    def get_option_trade_history(
        self, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve recent option trade history."""
        capped = min(limit, self.HISTORY_LIMIT)
        raw_list = self.redis_client.client.lrange(
            self.OPTION_TRADE_HISTORY_KEY, 0, capped - 1
        )

        history = []
        for raw in raw_list:
            try:
                history.append(json.loads(raw))
            except Exception as e:
                LOGGER.warning("Failed to parse option_trade history item: %s", e)
                continue

        return history
