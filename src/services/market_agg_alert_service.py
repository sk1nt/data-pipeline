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
    TRANSITION_LONG_BIAS = 0.80   # Ratio below this = long bias (call-heavy)
    TRANSITION_SHORT_BIAS = 1.00  # Ratio above this = short bias (put-heavy)
    
    # Discord channel IDs for alerts
    DISCORD_CHANNELS = [1425136266676146236, 1429940127899324487, 1440464526695731391]

    def __init__(self, redis_client: RedisClient):
        """Initialize the alert service with Redis client."""
        self.redis_client = redis_client
        self._last_ratio: Optional[Decimal] = None
        self._last_regime: Optional[str] = None  # 'long', 'neutral', 'short'

    def process_market_agg_update(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process market aggregation update and generate alerts if needed.
        
        Args:
            data: Market aggregation data with put_call_ratio, put_premium, call_premium, etc.
            
        Returns:
            Alert dict if alert triggered, None otherwise
        """
        try:
            # Use the put_call_ratio field from the message (volume-based ratio)
            # This is the authoritative ratio for bias detection:
            # - Below 0.8: Long bias
            # - 0.8 to 1.0: Neutral  
            # - Above 1.0: Short bias
            raw_ratio = data.get("put_call_ratio")
            if not raw_ratio:
                LOGGER.warning("Missing put_call_ratio in market_agg data")
                return None
            
            try:
                current_ratio = Decimal(str(raw_ratio))
            except Exception as e:
                LOGGER.warning("Invalid put_call_ratio value '%s': %s", raw_ratio, e)
                return None
            
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
            
            # Check for sentiment transition
            current_regime = self._get_regime(current_ratio)
            if self._last_regime and current_regime != self._last_regime:
                alert = self._create_alert(
                    alert_type="SENTIMENT_CHANGE",
                    current_ratio=current_ratio,
                    previous_ratio=self._last_ratio,
                    from_regime=self._last_regime,
                    to_regime=current_regime,
                    data=data,
                )
            
            # Log state for debugging
            LOGGER.debug(
                "Market agg update: ratio=%.3f regime=%s (prev_ratio=%s prev_regime=%s)",
                current_ratio,
                current_regime,
                self._last_ratio,
                self._last_regime,
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
        """Determine market regime based on put/call ratio.
        
        - Below 0.8: Long bias (calls dominating)
        - 0.8 to 1.0: Neutral
        - Above 1.0: Short bias (puts dominating)
        """
        if ratio < self.TRANSITION_LONG_BIAS:
            return "long"
        elif ratio > self.TRANSITION_SHORT_BIAS:
            return "short"
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

    def create_scheduled_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a scheduled Put/Call Ratio update (non-alert)."""
        try:
            raw_ratio = data.get("put_call_ratio")
            if not raw_ratio:
                return None
            
            current_ratio = Decimal(str(raw_ratio))
            current_regime = self._get_regime(current_ratio)
            
            update = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "alert_type": "SCHEDULED_UPDATE",
                "current_ratio": float(current_ratio),
                "regime": current_regime,
                "date": data.get("date"),
                "call_premium": data.get("call_premium"),
                "put_premium": data.get("put_premium"),
                "call_volume": data.get("call_volume"),
                "put_volume": data.get("put_volume"),
                "discord_channels": self.DISCORD_CHANNELS,
            }
            return update
        except Exception as exc:
            LOGGER.exception("Failed to create scheduled update: %s", exc)
            return None

    def format_alert_message(self, alert: Dict[str, Any]) -> str:
        """Format alert for Discord display."""
        alert_type = alert["alert_type"]
        current_ratio = alert["current_ratio"]
        previous_ratio = alert.get("previous_ratio")
        
        if alert_type == "SCHEDULED_UPDATE":
            # Format like GEX feed with ANSI color codes
            regime = alert.get("regime", "unknown")
            date = alert.get("date", "N/A")
            
            # ANSI color codes
            reset = "\u001b[0m"
            dim_white = "\u001b[2;37m"
            yellow = "\u001b[2;33m"
            green = "\u001b[2;32m"
            red = "\u001b[2;31m"
            
            # Choose color based on regime
            ratio_color = green if regime == "long" else (red if regime == "short" else yellow)
            
            call_prem = float(alert['call_premium'])
            put_prem = float(alert['put_premium'])
            call_vol = int(alert['call_volume'])
            put_vol = int(alert['put_volume'])
            
            return (
                f"```ansi\n"
                f"Put/Call Ratio: {dim_white}{date}{reset}\n"
                f"\n"
                f"Ratio               {ratio_color}{current_ratio:.3f}{reset}\n"
                f"Sentiment           {ratio_color}{regime.upper()}{reset}\n"
                f"\n"
                f"Call Premium        {green}${call_prem:,.0f}{reset}\n"
                f"Put Premium         {red}${put_prem:,.0f}{reset}\n"
                f"\n"
                f"Call Volume         {green}{call_vol:,}{reset}\n"
                f"Put Volume          {red}{put_vol:,}{reset}\n"
                f"```"
            )
        elif alert_type == "FAST_SHIFT":
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
        elif alert_type == "SENTIMENT_CHANGE":
            from_regime = alert.get("from_regime", "unknown")
            to_regime = alert.get("to_regime", "unknown")
            emoji = "⚠️"
            return (
                f"{emoji} **SENTIMENT CHANGE ALERT**\n"
                f"Market shifted from **{from_regime.upper()}** to **{to_regime.upper()}**\n"
                f"Put/Call Ratio: `{previous_ratio:.3f}` → `{current_ratio:.3f}`\n"
                f"Call Premium: `${float(alert['call_premium']):,.0f}`\n"
                f"Put Premium: `${float(alert['put_premium']):,.0f}`\n"
                f"Date: `{alert['date']}`"
            )
        else:
            return f"Unknown alert type: {alert_type}"
