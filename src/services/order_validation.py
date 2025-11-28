from typing import List


class OrderValidationService:
    @staticmethod
    def validate_futures_order(
        action: str, symbol: str, tp_ticks: float, quantity: int, mode: str
    ) -> List[str]:
        """Validate futures order parameters."""
        errors = []

        if action not in ["buy", "sell", "flat"]:
            errors.append("Action must be 'buy', 'sell', or 'flat'")

        if not symbol.startswith("/"):
            errors.append("Symbol must start with '/' for futures")

        if tp_ticks <= 0:
            errors.append("tp_ticks must be positive")

        if quantity <= 0:
            errors.append("Quantity must be positive")

        if mode not in ["dry", "live"]:
            errors.append("Mode must be 'dry' or 'live'")

        return errors

    @staticmethod
    def validate_options_alert(message: str, user_id: str) -> List[str]:
        """Validate options alert message."""
        errors = []

        if not message.startswith("Alert:"):
            errors.append("Message must start with 'Alert:'")

        # Check for required pattern
        import re

        pattern = re.compile(
            r"Alert:\s+(BTO|STC)\s+(\w+)\s+(\d+)(c|p)\s+(\d{1,2}/\d{1,2})\s+@\s+([\d.]+)",
            re.IGNORECASE,
        )
        if not pattern.search(message):
            errors.append(
                "Invalid alert format. Expected: Alert: BTO/STC SYMBOL STRIKEc/p MM/DD @ PRICE"
            )

        # Check user permissions
        from ..services.auth_service import AuthService

        if not AuthService.verify_user_for_alerts(user_id):
            errors.append("User not authorized to send alerts")

        return errors

    @staticmethod
    def validate_order_cancellation(order_id: str, user_id: str) -> List[str]:
        """Validate order cancellation request."""
        errors = []

        if not order_id:
            errors.append("Order ID is required")

        # TODO: Check if user owns the order

        return errors
