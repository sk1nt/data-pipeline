"""Service for monitoring market aggregation data and generating alerts on significant shifts."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Dict, Any

from src.lib.redis_client import RedisClient

LOGGER = logging.getLogger(__name__)


class MarketAggAlertService:
    """Monitor market_agg data and alert on significant put/call ratio shifts."""

    # Redis keys
    MARKET_AGG_STREAM_KEY = "uw:market_agg:stream"
    ALERT_CHANNEL = "market_agg:alerts"
    HISTORY_KEY = "market_agg:alert_history"
    LAST_RATIO_KEY = "market_agg:last_ratio"
    
    # Alert thresholds
    FAST_SHIFT_THRESHOLD = 0.15  # 15% change in ratio triggers alert
    TRANSITION_CALL_HEAVY = 0.85  # Ratio below this = call-heavy market
    TRANSITION_PUT_HEAVY = 1.15   # Ratio above this = put-heavy market
    
    # Discord channel IDs for alerts
    DISCORD_CHANNELS = [1425136266676146236, 1429940127899324487, 1440464526695731391]

    def __init__(self, redis_client: RedisClient):
        """Initialize the alert service with Redis client."""
        self.redis_client = redis_client
        self._last_ratio: Optional[Decimal] = None
        self._last_regime: Optional[str] = None  # 'call-heavy', 'neutral', 'put-heavy'

    def process_market_agg_update(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process market aggregation update and generate alerts if needed.
        
        Args:
            data: Market aggregation data with put_premium, call_premium, etc.
            
        Returns:
            Alert dict if alert triggered, None otherwise
        """
        try:
            # Extract premiums and calculate ratio
            put_premium = Decimal(data.get("put_premium", "0"))
            call_premium = Decimal(data.get("call_premium", "0"))
            
            if call_premium == 0:
                LOGGER.warning("Call premium is zero, cannot calculate ratio")
                return None
                
            current_ratio = put_premium / call_premium
            
            # Get last known ratio from Redis if we don't have it in memory
            if self._last_ratio is None:
                self._load_last_ratio()
            
            alert = None
            
            # Check for fast shift
            if self._last_ratio is not None:
                ratio_change = abs(current_ratio - self._last_ratio) / self._last_ratio
                
                if ratio_change >= self.FAST_SHIFT_THRESHOLD:
                    direction = "PUTS" if current_ratio > self._last_ratio else "CALLS"
                    alert = self._create_alert(
                        alert_type="FAST_SHIFT",
                        current_ratio=current_ratio,
                        previous_ratio=self._last_ratio,
                        change_pct=ratio_change * 100,
                        direction=direction,
                        data=data,
                    )
            
            # Check for regime transition
            current_regime = self._get_regime(current_ratio)
            if self._last_regime and current_regime != self._last_regime:
                alert = self._create_alert(
                    alert_type="REGIME_CHANGE",
                    current_ratio=current_ratio,
                    previous_ratio=self._last_ratio,
                    from_regime=self._last_regime,
                    to_regime=current_regime,
                    data=data,
                )
            
            # Update state
            self._last_ratio = current_ratio
            self._last_regime = current_regime
            self._save_last_ratio(current_ratio)
            
            # Publish alert if generated
            if alert:
                self._publish_alert(alert)
                return alert
                
            return None
            
        except Exception as exc:
            LOGGER.exception("Failed to process market agg update: %s", exc)
            return None

    def _get_regime(self, ratio: Decimal) -> str:
        """Determine market regime based on put/call ratio."""
        if ratio < self.TRANSITION_CALL_HEAVY:
            return "call-heavy"
        elif ratio > self.TRANSITION_PUT_HEAVY:
            return "put-heavy"
        else:
            return "neutral"

    def _create_alert(
        self,
        alert_type: str,
        current_ratio: Decimal,
        previous_ratio: Optional[Decimal],
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Create alert payload."""
        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alert_type": alert_type,
            "current_ratio": float(current_ratio),
            "previous_ratio": float(previous_ratio) if previous_ratio else None,
            "date": data.get("date"),
            "call_premium": data.get("call_premium"),
            "put_premium": data.get("put_premium"),
            "call_volume": data.get("call_volume"),
            "put_volume": data.get("put_volume"),
            "discord_channels": self.DISCORD_CHANNELS,
        }
        alert.update(kwargs)
        return alert

    def _publish_alert(self, alert: Dict[str, Any]) -> None:
        """Publish alert to Redis pubsub and store in history."""
        try:
            conn = self.redis_client.client
            serialized = json.dumps(alert, default=str)
            
            # Publish to channel for Discord bot
            conn.publish(self.ALERT_CHANNEL, serialized)
            
            # Store in alert history (keep last 100)
            conn.lpush(self.HISTORY_KEY, serialized)
            conn.ltrim(self.HISTORY_KEY, 0, 99)
            
            LOGGER.info(
                "Published market_agg alert: %s (ratio: %.3f -> %.3f)",
                alert["alert_type"],
                alert.get("previous_ratio", 0),
                alert["current_ratio"],
            )
        except Exception as exc:
            LOGGER.exception("Failed to publish alert: %s", exc)

    def _load_last_ratio(self) -> None:
        """Load last known ratio from Redis."""
        try:
            conn = self.redis_client.client
            value = conn.get(self.LAST_RATIO_KEY)
            if value:
                self._last_ratio = Decimal(value.decode("utf-8"))
                self._last_regime = self._get_regime(self._last_ratio)
        except Exception as exc:
            LOGGER.warning("Failed to load last ratio from Redis: %s", exc)

    def _save_last_ratio(self, ratio: Decimal) -> None:
        """Save current ratio to Redis for persistence."""
        try:
            conn = self.redis_client.client
            conn.setex(self.LAST_RATIO_KEY, 86400, str(ratio))  # 24h TTL
        except Exception as exc:
            LOGGER.warning("Failed to save last ratio to Redis: %s", exc)

    def format_alert_message(self, alert: Dict[str, Any]) -> str:
        """Format alert for Discord display."""
        alert_type = alert["alert_type"]
        current_ratio = alert["current_ratio"]
        previous_ratio = alert.get("previous_ratio")
        
        if alert_type == "FAST_SHIFT":
            direction = alert.get("direction", "UNKNOWN")
            change_pct = alert.get("change_pct", 0)
            emoji = "🔴" if direction == "PUTS" else "🟢"
            return (
                f"{emoji} **FAST SHIFT ALERT: Market Moving to {direction}**\n"
                f"Put/Call Ratio: `{previous_ratio:.3f}` → `{current_ratio:.3f}` "
                f"({change_pct:+.1f}%)\n"
                f"Call Premium: `${float(alert['call_premium']):,.0f}`\n"
                f"Put Premium: `${float(alert['put_premium']):,.0f}`\n"
                f"Date: `{alert['date']}`"
            )
        elif alert_type == "REGIME_CHANGE":
            from_regime = alert.get("from_regime", "unknown")
            to_regime = alert.get("to_regime", "unknown")
            emoji = "⚠️"
            return (
                f"{emoji} **REGIME CHANGE ALERT**\n"
                f"Market shifted from **{from_regime.upper()}** to **{to_regime.upper()}**\n"
                f"Put/Call Ratio: `{previous_ratio:.3f}` → `{current_ratio:.3f}`\n"
                f"Call Premium: `${float(alert['call_premium']):,.0f}`\n"
                f"Put Premium: `${float(alert['put_premium']):,.0f}`\n"
                f"Date: `{alert['date']}`"
            )
        else:
            return f"Unknown alert type: {alert_type}"
