from datetime import datetime
from decimal import Decimal
from typing import Optional

from tastytrade import Account
from tastytrade.instruments import Future
from tastytrade.order import NewOrder, OrderAction, OrderTimeInForce, OrderType

from ..config.tastytrade_config import config
from ..models.order import Order, OrderType as ModelOrderType, OrderStatus, Environment
from ..services.complex_order_builder import build_bracket_complex_order
from ..services.futures_order_parser import FuturesAction, FuturesOrderParams
from ..services.tastytrade_client import select_account, tastytrade_client


class FuturesOrderService:
    async def place_order(self, params: FuturesOrderParams, user_id: str) -> dict:
        """Place a futures order via Tastytrade API.

        When a take-profit exit (and optional stop-loss) is requested the order
        is submitted atomically as a complex bracket order -- OTOCO when both TP
        and SL are present, OTO when only TP -- so the entry and exit(s) cannot
        desync.  ``FLAT`` and plain (no-TP) orders still use a raw market order.

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

        tp_ticks = float(getattr(params, "tp_ticks", 0) or 0)
        sl_ticks = float(getattr(params, "sl_ticks", 0) or 0)

        if tp_ticks > 0 and params.action != FuturesAction.FLAT:
            # Atomic OTOCO/OTO bracket -- shared builder with the Discord bot.
            order_id = self._place_bracket_order(
                account, session, future, params, action, tp_ticks, sl_ticks, dry_run
            )
        else:
            order = NewOrder(
                time_in_force=OrderTimeInForce.DAY,
                order_type=OrderType.MARKET,
                legs=[leg],
            )
            response = account.place_order(session, order, dry_run=dry_run)
            order_id = str(response.order.id) if hasattr(response, "order") else None

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

    def _place_bracket_order(
        self,
        account,
        session,
        future,
        params: FuturesOrderParams,
        action,
        tp_ticks: float,
        sl_ticks: float,
        dry_run: bool,
    ) -> Optional[str]:
        """Submit an atomic OTOCO/OTO bracket complex order.

        Falls back to a plain market order (no exit) when no reference price is
        available to compute the take-profit level.
        """
        ref_price = self._reference_price(future)
        qty = Decimal(params.quantity)

        if ref_price is None:
            order = NewOrder(
                time_in_force=OrderTimeInForce.DAY,
                order_type=OrderType.MARKET,
                legs=[future.build_leg(qty, action)],
            )
            response = account.place_order(session, order, dry_run=dry_run)
            return str(response.order.id) if hasattr(response, "order") else None

        active_tick = float(getattr(future, "active_tick_size", 0.25) or 0.25)
        is_long = params.action == FuturesAction.BUY
        tp_action = OrderAction.SELL_TO_CLOSE if is_long else OrderAction.BUY_TO_CLOSE

        if is_long:
            tp_price = ref_price + (tp_ticks * active_tick)
        else:
            tp_price = ref_price - (tp_ticks * active_tick)

        sl_price = None
        if sl_ticks > 0:
            if is_long:
                sl_price = ref_price - (sl_ticks * active_tick)
            else:
                sl_price = ref_price + (sl_ticks * active_tick)

        entry_leg = future.build_leg(qty, action)
        closing_leg = future.build_leg(qty, tp_action)

        bracket = build_bracket_complex_order(
            entry_leg=entry_leg,
            closing_leg=closing_leg,
            tp_price=tp_price,
            tp_action=tp_action,
            sl_price=sl_price,
        )

        response = account.place_complex_order(session, bracket, dry_run=dry_run)
        complex_order = getattr(response, "complex_order", None)
        return str(complex_order.id) if complex_order else None

    @staticmethod
    def _reference_price(future) -> Optional[float]:
        """Best-effort pre-trade reference price from a Future instrument."""
        for attr in ("mark_price", "last_price", "close_price", "settlement_price"):
            val = getattr(future, attr, None)
            if val is not None and float(val) > 0:
                return float(val)
        return None

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
