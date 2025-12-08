from decimal import Decimal
from tastytrade import Account
from tastytrade.instruments import Future
from tastytrade.order import NewOrder, OrderAction, OrderTimeInForce, OrderType
from ..services.tastytrade_client import tastytrade_client
from ..services.futures_order_parser import FuturesOrderParams, FuturesAction
from ..models.order import Order, OrderType as ModelOrderType, OrderStatus, Environment
from ..config.settings import config
from datetime import datetime


def _select_account(accounts):
    if not accounts:
        return None
    target = getattr(config, "tastytrade_account", None)
    if target:
        for acc in accounts:
            acc_number = getattr(acc, "account_number", None) or getattr(acc, "number", None)
            if acc_number == target:
                return acc
    return accounts[0]


class FuturesOrderService:
    async def place_order(self, params: FuturesOrderParams, user_id: str) -> str:
        """Place a futures order via Tastytrade API."""

        session = tastytrade_client.get_session()

        # Get account
        accounts = Account.get(session)
        account = _select_account(accounts)
        if not account:
            raise ValueError("No Tastytrade accounts found")

        # Get future instrument
        future = Future.get(session, params.symbol)
        if not future:
            raise ValueError(f"Future symbol {params.symbol} not found")

        # Determine order action
        if params.action == FuturesAction.BUY:
            action = OrderAction.BUY_TO_OPEN
        elif params.action == FuturesAction.SELL:
            action = OrderAction.SELL_TO_CLOSE
        elif params.action == FuturesAction.FLAT:
            action = OrderAction.SELL_TO_CLOSE  # Assuming flat means close position

        # Create leg
        leg = future.build_leg(Decimal(params.quantity), action)

        # Create order
        order = NewOrder(
            time_in_force=OrderTimeInForce.DAY,
            order_type=OrderType.MARKET,  # Futures typically market orders
            legs=[leg],
        )

        # Dry run or live
        dry_run = params.mode == "dry"

        # Place order
        response = account.place_order(session, order, dry_run=dry_run)

        # Create order record
        order_record = Order(
            id=str(response.order.id)
            if hasattr(response, "order")
            else f"dry_{datetime.now().timestamp()}",
            symbol=params.symbol,
            quantity=float(params.quantity),
            order_type=ModelOrderType.MARKET,
            status=OrderStatus.PENDING,
            environment=Environment.SANDBOX
            if config.tastytrade_use_sandbox
            else Environment.PRODUCTION,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            channel_id="",  # Not applicable for chat commands
            user_id=user_id,
        )

        # TODO: Save to database

        return f"Order {order_record.id} placed successfully ({'dry run' if dry_run else 'live'})"
