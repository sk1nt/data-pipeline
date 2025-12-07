import re
from typing import Optional
from enum import Enum


class AlertAction(str, Enum):
    BUY_TO_OPEN = "BTO"
    SELL_TO_CLOSE = "STC"
    # Add more as needed


class AlertParser:
    def __init__(self):
        # Flexible pattern for alerts supporting formats like:
        # - "Alert: BTO UBER 78p 12/05 @ 0.75"
        # - "BTO UBER 78p 12/05 0.75"
        # - "BUY UBER 78p 12/05 0.75"
        # Price is optional and may be provided with or without '@'
        self.pattern = re.compile(
            r"^(?:Alert:)?\s*(BTO|STC|BUY|SELL)\s+(?P<symbol>[A-Z0-9_.-]+)\s+(?P<strike>\d+(?:\.\d+)?)\s*(?P<option_type>[cp])\s+(?P<expiry>\d{1,2}/\d{1,2}(?:/\d{2,4})?)(?:\s*(?:@|\s)\s*(?P<price>[\d.]+))?",
            re.IGNORECASE,
        )

    def parse_alert(self, message: str, user_id: str) -> Optional[dict]:
        """Parse an alert message and return order parameters."""

        match = self.pattern.search(message.strip())
        if not match:
            return None

        action = match.group(1)
        symbol = match.group("symbol")
        strike = match.group("strike")
        option_type = match.group("option_type")
        expiry = match.group("expiry")
        price = match.group("price")

        # Validate user
        from services.auth_service import AuthService

        # Validate user permit for alerts
        if not AuthService.verify_user_for_alerts(user_id):
            return None

        # Price may be missing; return None for price in that case
        parsed_price = float(price) if price is not None else None

        return {
            "action": action.upper(),
            "symbol": symbol.upper(),
            "strike": float(strike),
            "option_type": option_type.lower(),  # 'c' or 'p'
            "expiry": expiry,
            "price": parsed_price,
            "user_id": user_id,
        }

    def is_buy_message(self, channel_id: str) -> bool:
        """Check if channel is for buy messages."""
        return channel_id == "1255265167113978008"
