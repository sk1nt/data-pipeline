import re
from typing import Optional
from enum import Enum


class AlertAction(str, Enum):
    BUY_TO_OPEN = "BTO"
    SELL_TO_CLOSE = "STC"
    # Add more as needed


class AlertParser:
    def __init__(self):
        # Pattern for alerts: "Alert: BTO UBER 78p 12/05 @ 0.75"
        self.pattern = re.compile(
            r"Alert:\s+(BTO|STC)\s+(\w+)\s+(\d+)(c|p)\s+(\d{1,2}/\d{1,2})\s+@\s+([\d.]+)",
            re.IGNORECASE,
        )

    def parse_alert(self, message: str, user_id: str) -> Optional[dict]:
        """Parse an alert message and return order parameters."""

        match = self.pattern.search(message)
        if not match:
            return None

        action, symbol, strike, option_type, expiry, price = match.groups()

        # Validate user
        from services.auth_service import AuthService

        if not AuthService.verify_user_for_alerts(user_id):
            return None

        return {
            "action": action.upper(),
            "symbol": symbol.upper(),
            "strike": float(strike),
            "option_type": option_type.lower(),  # 'c' or 'p'
            "expiry": expiry,
            "price": float(price),
            "user_id": user_id,
        }

    def is_buy_message(self, channel_id: str) -> bool:
        """Check if channel is for buy messages."""
        return channel_id == "1255265167113978008"
