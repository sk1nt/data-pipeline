from decimal import Decimal
from tastytrade import Account
from tastytrade.instruments import Future
from tastytrade.order import NewOrder, OrderAction, OrderTimeInForce, OrderType
from ..services.tastytrade_client import tastytrade_client, select_account
from ..services.futures_order_parser import FuturesOrderParams, FuturesAction
from ..models.order import Order, OrderType as ModelOrderType, OrderStatus, Environment
from ..config.settings import config
from datetime import datetime


class FuturesOrderService:
    async def place_order(self, params: FuturesOrderParams, user_id: str) -> dict:
        """Place a futures order via Tastytrade API.

        Returns dict with 'message' and 'order_id' keys.
        """
        tastytrade_client.ensure_authorized()
        session = tastytrade_client.get_session()

        accounts = Account.get(session)
        account = select_account(accounts)
        if not account:
            raise ValueError("No Tastytrade accounts found")

        future = Future.get(session, params.symbol)
        if not future:
            raise ValueError(f"Future symbol {params.symbol} not found")

        if params.action == FuturesAction.BUY:
            action = OrderAction.BUY_TO_OPEN
        elif params.action == FuturesAction.SELL:
            action = OrderAction.SELL_TO_CLOSE
        elif params.action == FuturesAction.FLAT:
            action = self._resolve_flatten_action(account, session, params.symbol)
        else:
            raise ValueError(f"Unsupported futures action: {params.action}")

        leg = future.build_leg(Decimal(params.quantity), action)
        dry_run = params.mode == "dry"

        order = NewOrder(
            time_in_force=OrderTimeInForce.DAY,
            order_type=OrderType.MARKET,
            legs=[leg],
        )

        response = account.place_order(session, order, dry_run=dry_run)
        order_id = str(response.order.id) if hasattr(response, "order") else None

        # If tp_ticks specified and not FLAT, place a TP limit order
        if params.tp_ticks > 0 and params.action != FuturesAction.FLAT and not dry_run:
            try:
                active_tick = float(getattr(future, "active_tick_size", 0.25))
                tp_distance = params.tp_ticks * active_tick

                fill_price = None
                if hasattr(response, "order") and hasattr(response.order, "price"):
                    fill_price = float(response.order.price)

                if fill_price:
                    if params.action == FuturesAction.BUY:
                        tp_price = fill_price + tp_distance
                        tp_action = OrderAction.SELL_TO_CLOSE
                    else:
                        tp_price = fill_price - tp_distance
                        tp_action = OrderAction.BUY_TO_CLOSE

                    tp_leg = future.build_leg(Decimal(params.quantity), tp_action)
                    tp_order = NewOrder(
                        time_in_force=OrderTimeInForce.DAY,
                        order_type=OrderType.LIMIT,
                        legs=[tp_leg],
                        price=round(tp_price, 2),
                    )
                    account.place_order(session, tp_order, dry_run=False)
            except Exception:
                pass  # TP placement is best-effort; entry already succeeded

        order_record = Order(
            id=order_id or f"dry_{datetime.now().timestamp()}",
            symbol=params.symbol,
            quantity=float(params.quantity),
            order_type=ModelOrderType.MARKET,
            status=OrderStatus.PENDING,
            environment=Environment.SANDBOX if config.tastytrade_use_sandbox else Environment.PRODUCTION,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            channel_id="",
            user_id=user_id,
        )

        return {
            "message": f"Order {order_record.id} placed successfully ({'dry run' if dry_run else 'live'})",
            "order_id": order_id,
        }

    @staticmethod
    def _resolve_flatten_action(account, session, symbol: str):
        """Resolve the proper close action from the live futures position."""
        search_symbol = (symbol or "").upper().lstrip("/")
        positions = account.get_positions(session)
        for pos in positions:
            pos_symbol = str(pos.get("symbol") or "").upper().lstrip("/")
            pos_underlying = str(pos.get("underlying_symbol") or "").upper().lstrip("/")
            if search_symbol not in pos_symbol and search_symbol not in pos_underlying:
                continue

            quantity = int(float(pos.get("quantity") or 0))
            direction = pos.get("quantity_direction") or pos.get("direction") or ""
            if hasattr(direction, "value"):
                direction = direction.value
            direction = str(direction).lower()

            if quantity < 0 or direction == "short":
                return OrderAction.BUY_TO_CLOSE
            return OrderAction.SELL_TO_CLOSE

        raise ValueError(f"No open futures position found for {symbol}")
