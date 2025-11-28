from typing import Optional

# Import via absolute paths so the module works when `src` is on sys.path
from services.alert_parser import AlertParser
from services.options_fill_service import OptionsFillService
from services.tastytrade_client import tastytrade_client, TastytradeAuthError


class AutomatedOptionsService:
    def __init__(self):
        self.alert_parser = AlertParser()
        self.fill_service = OptionsFillService()

    async def process_alert(
        self, message: str, channel_id: str, user_id: str
    ) -> Optional[str]:
        """Process an alert message and place automated order."""

        # Parse alert
        alert_data = self.alert_parser.parse_alert(message, user_id)
        if not alert_data:
            return None

        # Calculate quantity based on allocation
        allocation_pct = 0.1  # TODO: Get from trader config
        buying_power = 10000  # TODO: Get from account
        quantity = (buying_power * allocation_pct) / alert_data["price"]

        # Adjust for buy message channel
        if self.alert_parser.is_buy_message(channel_id):
            quantity *= 0.5

        # Verify auth before attempting to place an order
        try:
            tastytrade_client.ensure_authorized()
        except TastytradeAuthError as exc:
            print(f"TastyTrade authentication invalid: {exc}")
            # Return an informative message so the caller (Discord bot) can send feedback
            return f'TastyTrade authentication invalid: {exc}. Please update the refresh token.'

        # Place order
        try:
            result = await self.fill_service.fill_options_order(
            symbol=alert_data["symbol"],
            strike=alert_data["strike"],
            option_type=alert_data["option_type"],
            expiry=alert_data["expiry"],
            quantity=int(quantity),
            action=alert_data["action"],
            user_id=user_id,
            channel_id=channel_id,
            )
            return result
        except TastytradeAuthError:
            print("TastyTrade auth invalid during order placement")
            return 'TastyTrade authentication invalid while placing order. Please update your refresh token.'
