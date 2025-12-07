import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

# Import via absolute paths so the module works when `src` is on sys.path
from services.alert_parser import AlertParser
from services.options_fill_service import OptionsFillService
from services.tastytrade_client import tastytrade_client, TastytradeAuthError
from lib.logging import get_logger
from lib.redis_client import get_redis_client

try:
    from tastytrade.account import Account  # type: ignore
except Exception:  # pragma: no cover - optional dependency path
    Account = None  # type: ignore

logger = get_logger(__name__)


class AutomatedOptionsService:
    def __init__(self, tastytrade_client=tastytrade_client):
        self.alert_parser = AlertParser()
        self.tastytrade_client = tastytrade_client
        self.fill_service = OptionsFillService(tastytrade_client=self.tastytrade_client)

    async def process_alert(
        self, message: str, channel_id: str, user_id: str
    ) -> Optional[dict]:
        """Process an alert message and place automated order."""

        # Parse alert
        alert_data = self.alert_parser.parse_alert(message, user_id)
        if not alert_data:
            return None

        # Verify allowed user & channel for automated trades
        from services.auth_service import AuthService

        if not AuthService.verify_user_and_channel_for_automated_trades(user_id, channel_id):
            # Not permitted to trigger automated trades
            logger.warning(
                "User %s or channel %s is not permitted for automated trades",
                user_id,
                channel_id,
            )
            return None

        # Verify auth before attempting to place an order
        try:
            self.tastytrade_client.ensure_authorized()
        except TastytradeAuthError as exc:
            msg = f"TastyTrade authentication invalid: {exc}. Please update the refresh token."
            print(msg)
            raise TastytradeAuthError(msg) from exc

        # Calculate quantity based on allocation and account balances
        quantity = self._compute_quantity(alert_data, channel_id)

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
                initial_price=Decimal(str(alert_data.get("price") or 0.0)),
            )
        except TastytradeAuthError as exc:
            msg = f"TastyTrade authentication invalid while placing order: {exc}. Please update your refresh token."
            print(msg)
            raise TastytradeAuthError(msg) from exc

        if not result:
            return None

        # Always return structured info: order_id, quantity, entry_price
        order_id = result.get("order_id") if isinstance(result, dict) else result
        entry_price = result.get("entry_price") if isinstance(result, dict) else None

        # Audit log to Redis for compliance and debugging
        try:
            redis_client = get_redis_client()
            redis_conn = redis_client.client
            audit_key = "audit:automated_alerts"
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "channel_id": channel_id,
                "alert_message": message,
                "parsed_alert": alert_data,
                "computed_quantity": int(quantity),
                "entry_price": float(entry_price) if entry_price is not None else None,
                "order_id": order_id,
            }
            redis_conn.lpush(audit_key, json.dumps(payload, default=str))
            logger.info(
                "Audit logged automated alert: order_id=%s qty=%s entry_price=%s",
                order_id,
                int(quantity),
                entry_price,
            )
        except Exception:
            # Non-fatal; we still return success, but log for the operator
            print("Failed to write audit log to Redis for automated alert")

        return {
            "order_id": order_id,
            "quantity": int(quantity),
            "entry_price": entry_price,
        }

    def _compute_quantity(self, alert_data: dict, channel_id: str) -> int:
        """Compute contract quantity using allocation, price, and account balances."""
        alloc_pct = 0.1  # TODO: configurable per trader
        price = float(alert_data.get("price") or 0.0)
        contract_price = max(price, 0.01) * 100  # options are 100x multiplier
        max_contracts = 10

        buying_power = 10000.0
        net_liq = 10000.0
        try:
            if Account is not None:
                session = self.tastytrade_client.get_session()
                accounts = Account.get(session)
                if accounts:
                    account = accounts[0]
                    balances = account.get_balances(session)
                    # Prefer available_trading_funds if present
                    buying_power = float(
                        getattr(balances, "available_trading_funds", None)
                        or getattr(balances, "derivative_buying_power", None)
                        or getattr(balances, "equity_buying_power", None)
                        or getattr(balances, "day_trading_buying_power", None)
                        or getattr(balances, "net_liquidating_value", 10000.0)
                        or 10000.0
                    )
                    net_liq = float(
                        getattr(balances, "net_liquidating_value", None) or buying_power
                    )
        except Exception:
            # Fallback to defaults if balances are unavailable
            pass

        alloc_dollars = max(0.0, buying_power * alloc_pct)
        qty = int(alloc_dollars // contract_price) if contract_price > 0 else 0

        # Adjust for buy alert channels (more conservative sizing)
        if self.alert_parser.is_buy_message(channel_id):
            qty = max(1, int(qty * 0.5))

        # Ensure at least 1 contract and cap to avoid oversizing
        if qty <= 0:
            qty = 1
        qty = min(qty, max_contracts)

        # Additional safety: avoid exceeding a loose multiple of net liq
        est_notional = qty * contract_price
        if net_liq > 0 and est_notional > net_liq * 10:
            # Scale down to 10x net liq as a guardrail
            qty = max(1, int((net_liq * 10) // contract_price))

        return qty
