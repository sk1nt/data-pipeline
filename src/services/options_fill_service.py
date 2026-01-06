import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional
from datetime import datetime, timezone, date
from tastytrade import Account
from tastytrade.instruments import get_option_chain, Option, InstrumentType
from tastytrade.market_data import get_market_data
from tastytrade.order import (
    NewOrder,
    OrderAction,
    OrderTimeInForce,
    OrderType,
    PriceEffect,
)

# Import via absolute paths so the module works when `src` is on sys.path
from services.tastytrade_client import tastytrade_client, TastytradeAuthError


class InsufficientBuyingPowerError(Exception):
    """Raised when estimated notional exceeds available buying power."""
from src.lib.retries import retry_with_backoff, TransientError
from config.settings import config


class OptionsFillService:
    def __init__(
        self,
        max_retries: int = 3,
        timeout_seconds: int = 30,
        tick_increment: Decimal = Decimal("0.01"),
        tastytrade_client=tastytrade_client,
    ):
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.tick_increment = tick_increment
        self.tastytrade_client = tastytrade_client

    async def get_mid_price(self, option_symbol: str) -> Optional[Decimal]:
        """Fetch mid-price for an option symbol."""
        try:
            # Ensure we have a valid session/token before accessing market data
            self.tastytrade_client.ensure_authorized()
            session = self.tastytrade_client.get_session()
            data = get_market_data(
                session, option_symbol, instrument_type=InstrumentType.EQUITY_OPTION
            )
            if data.bid and data.ask:
                return (data.bid + data.ask) / 2
        except TastytradeAuthError as e:
            print(
                f"TastyTrade auth error while fetching market data for {option_symbol}: {e}"
            )
            raise
        except Exception as e:
            print(f"Error fetching market data for {option_symbol}: {e}")
        return None

    def _select_account(self, accounts):
        """Select configured account if present, else first available."""
        if not accounts:
            return None
        target = getattr(config, "tastytrade_account", None)
        if target:
            for acc in accounts:
                acc_number = getattr(acc, "account_number", None) or getattr(
                    acc, "number", None
                )
                if acc_number == target:
                    return acc
        return accounts[0]

    async def place_limit_order(
        self, option: Option, quantity: Decimal, action: OrderAction, price: Decimal
    ) -> Optional[str]:
        """Place a limit order for the option."""
        # Normalize action if a string alias was passed for compatibility
        if isinstance(action, str):
            alias = action.upper()
            if alias in ("BTO", "BUY"):
                action = OrderAction.BUY_TO_OPEN
            elif alias in ("STC", "SELL"):
                action = OrderAction.SELL_TO_CLOSE
            else:
                # leave as-is; underlying SDK will validate
                pass

        # Verify auth before attempting to place an order
        self.tastytrade_client.ensure_authorized()
        session = self.tastytrade_client.get_session()
        accounts = Account.get(session)
        account = self._select_account(accounts)
        if not account:
            return None

        # Preflight: verify trading status to avoid placing orders when options are closing-only
        try:
            trading_status = account.get_trading_status(session)
            # If options are closing only, we should not submit opening orders
            if getattr(trading_status, "is_options_closing_only", False):
                # opening orders are not allowed; abort
                print("Trading status indicates options are closing-only; aborting order")
                return None
        except Exception:
            # Non-fatal; continue if trading status is unavailable
            pass

        # Preflight: verify buying power is sufficient for the requested quantity at the target price
        try:
            balances = account.get_balances(session)
            buying_power = float(
                getattr(balances, "available_trading_funds", None)
                or getattr(balances, "derivative_buying_power", None)
                or getattr(balances, "equity_buying_power", None)
                or getattr(balances, "day_trading_buying_power", None)
                or getattr(balances, "net_liquidating_value", 0)
                or 0
            )
            contract_price = float(price) * 100
            est_notional = float(quantity) * contract_price
            if buying_power <= 0 or est_notional > buying_power:
                msg = (
                    f"Insufficient buying power for order: est_notional={est_notional} buying_power={buying_power}"
                )
                print(msg)
                raise InsufficientBuyingPowerError(msg)
        except InsufficientBuyingPowerError:
            raise
        except Exception:
            # If we can't determine balances, continue and allow order to proceed
            pass

        leg = option.build_leg(quantity, action)
        # Determine appropriate price effect for the order based on action
        desired_price_effect = (
            PriceEffect.DEBIT
            if action in (OrderAction.BUY_TO_OPEN, OrderAction.BUY_TO_CLOSE)
            else PriceEffect.CREDIT
        )
        order = NewOrder(
            time_in_force=OrderTimeInForce.DAY,
            order_type=OrderType.LIMIT,
            legs=[leg],
            price=price,
            price_effect=desired_price_effect,
        )
        # Ensure the model has our desired price effect; rebuild if the SDK normalized it away
        try:
            if order.model_dump().get('price_effect') != desired_price_effect:
                order = order.model_copy(update={"price_effect": desired_price_effect})
        except Exception:
            pass

        try:
            # Use retry wrapper for transient errors
            @retry_with_backoff(max_retries=3, initial_backoff=0.5)
            def attempt_place_order():
                # Log the final order JSON before sending for debugging
                try:
                    logger = __import__("logging").getLogger(__name__)
                    logger.debug("Placing order JSON: %s", order.model_dump())
                except Exception:
                    pass
                return account.place_order(
                    session, order, dry_run=config.tastytrade_use_sandbox
                )

            response = attempt_place_order()
            return str(response.order.id) if hasattr(response, "order") else None
        except TastytradeAuthError as e:
            print(f"TastyTrade auth error while placing order: {e}")
            raise
        except Exception as e:
            # If the error suggests we can't buy for a credit, it means price_effect is wrong
            # Options cannot use MARKET orders - must always be LIMIT
            msg = str(e)
            print(f"Error placing order: {msg}")
            if "cant_buy_for_credit" in msg.lower() or "cannot buy" in msg.lower():
                try:
                    logger = __import__("logging").getLogger(__name__)
                    logger.warning("Order failed; TastyTrade rejected price_effect. Not retrying to avoid duplicate rejections.")
                except Exception:
                    pass
                # Don't retry with market order - options don't support market orders on TastyTrade
                # The price_effect logic above should have been correct, so this is a broker constraint
                raise Exception(f"TastyTrade rejected order (price_effect issue): {msg}")
            return None

    async def check_order_filled(self, order_id: str) -> bool:
        """Check if an order is filled."""
        try:
            self.tastytrade_client.ensure_authorized()
            session = self.tastytrade_client.get_session()
            accounts = Account.get(session)
            account = self._select_account(accounts)
            if account:
                orders = account.get_live_orders(session)
                for order in orders:
                    if str(order.id) == order_id and order.status.value == "Filled":
                        return True
        except TastytradeAuthError as e:
            print(f"TastyTrade auth error while checking order {order_id}: {e}")
            raise
        except Exception as e:
            print(f"Error checking order status: {e}")
        return False

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            self.tastytrade_client.ensure_authorized()
            session = self.tastytrade_client.get_session()
            accounts = Account.get(session)
            account = self._select_account(accounts)
            if account:
                @retry_with_backoff(max_retries=3, initial_backoff=0.5)
                def attempt_delete():
                    return account.delete_order(session, order_id)

                attempt_delete()
                return True
        except TastytradeAuthError as e:
            print(f"TastyTrade auth error while canceling order {order_id}: {e}")
            raise
        except Exception as e:
            print(f"Error canceling order {order_id}: {e}")
            return False

    async def fill_options_order(
        self,
        symbol: str,
        strike: float,
        option_type: str,
        expiry: str,
        quantity: float,
        action: str,
        user_id: str,
        channel_id: str,
        initial_price: Optional[Decimal] = None,
    ) -> Optional[dict]:
        """Attempt to fill an options order with retries and price adjustments."""

        # Ensure auth is valid before initiating any trade-related calls
        self.tastytrade_client.ensure_authorized()
        # Get option
        session = self.tastytrade_client.get_session()
        chain = get_option_chain(session, symbol)
        target_date = self._normalize_expiry(expiry)
        chain_key = self._find_chain_expiry_key(chain, target_date)
        if chain_key is None:
            print(f"Expiry {target_date} not found in chain (original input: {expiry})")
            return None

        options = [
            opt
            for opt in chain[chain_key]
            if opt.strike_price == Decimal(str(strike))
            and opt.option_type.value.lower() == option_type
        ]
        if not options:
            print(f"Option not found: {symbol} {strike} {option_type} {expiry}")
            return None

        option = options[0]

        # Determine action
        if action == "BTO":
            order_action = OrderAction.BUY_TO_OPEN
        elif action == "STC":
            order_action = OrderAction.SELL_TO_CLOSE
        else:
            return None

        # Get initial mid-price
        mid_price = await self.get_mid_price(option.symbol)
        if not mid_price:
            # If there's no mid price and we have an initial price, allow proceeding
            if initial_price is None:
                return None

        # Use the provided initial_price (from alert) if present; otherwise default to mid price
        if initial_price is not None:
            try:
                if not isinstance(initial_price, Decimal):
                    current_price = Decimal(str(initial_price))
                else:
                    current_price = initial_price
            except Exception:
                # fallback to mid_price if parsing fails
                current_price = mid_price
        else:
            current_price = mid_price
        # Price discovery: start at initial price or mid market and iterate
        from src.services.price_discovery import discovery_attempts
        from src.services.metrics import metrics

        start_price = current_price
        attempts_iter = discovery_attempts(
            start_price,
            mid_price,
            self.tick_increment,
            max_increments=self.max_retries,
            wait_seconds=20,
            convert_to_market_if_remaining_ticks_leq=1,
        )
        attempt = 0
        for current_price, convert_to_market in attempts_iter:
            # Emit a metrics counter for discovery attempt
            try:
                metrics.incr("price_discovery_attempts")
            except Exception:
                pass
            print(f"Attempt {attempt + 1}: Placing order at {current_price} (aggressive={convert_to_market})")
            try:
                logger = __import__("logging").getLogger(__name__)
                logger.info(
                    "Attempt %s: placing order at %s (aggressive=%s)", attempt + 1, current_price, convert_to_market
                )
            except Exception:
                pass
            # Note: TastyTrade does not support market orders for options, always use limit
            order_id = await self.place_limit_order(
                option, Decimal(str(quantity)), order_action, current_price
            )
            if not order_id:
                return None

            # Wait for fill (use shorter timeout during price discovery to react quickly)
            await asyncio.sleep(min(self.timeout_seconds, 20))
            if await self.check_order_filled(order_id):
                print(f"Order filled on attempt {attempt + 1}")

                # Create profit-taking order if filled
                if order_action == OrderAction.BUY_TO_OPEN:
                    await self.create_profit_exit(
                        option, Decimal(str(quantity)), order_id
                    )

                # Return both the order id and price used for the filled entry
                return {"order_id": order_id, "entry_price": str(current_price)}
            if attempt < self.max_retries:
                await self.cancel_order(order_id)
                # If we were converting to market, we stop iterating; else attempts_iter will move forward
                attempt += 1

        # Leave as DAY order
        print("Order not filled, leaving as DAY order")
        return {"order_id": order_id, "entry_price": str(current_price)}

    def _normalize_expiry(self, expiry: str) -> date:
        """Normalize expiry like '12/05' to a date object using the nearest future year."""
        if not expiry:
            return None
        exp = expiry.strip()
        import re

        m = re.match(
            r"^(?P<month>\d{1,2})/(?P<day>\d{1,2})(?:/(?P<year>\d{2,4}))?$", exp
        )
        if not m:
            # Try ISO date
            try:
                return date.fromisoformat(exp)
            except Exception:
                return None
        month = int(m.group("month"))
        day = int(m.group("day"))
        year_part = m.group("year")
        if year_part:
            if len(year_part) == 2:
                year = 2000 + int(year_part)
            else:
                year = int(year_part)
        else:
            today = datetime.now(timezone.utc).date()
            tentative = date(today.year, month, day)
            if tentative < today:
                tentative = date(today.year + 1, month, day)
            return tentative
        try:
            return date(year, month, day)
        except Exception:
            return None

    def _find_chain_expiry_key(self, chain, target: Optional[date]):
        """Find the matching chain key for the given target date, tolerating different key formats."""
        if target is None:
            return None
        iso = target.isoformat()
        # direct date key
        if target in chain:
            return target
        if iso in chain:
            return iso
        for k in chain.keys():
            try:
                # If key is an ISO string, try parsing and compare by month/day/year
                if isinstance(k, str):
                    from datetime import date as _date

                    try:
                        parsed = _date.fromisoformat(k)
                        if parsed == target:
                            return k
                        # also allow matching by month/day ignoring year
                        if parsed.month == target.month and parsed.day == target.day:
                            return k
                    except Exception:
                        pass
                if isinstance(k, str) and k.startswith(iso):
                    return k
                if isinstance(k, datetime) and k.date() == target:
                    return k
                if isinstance(k, date) and k == target:
                    return k
            except Exception:
                continue
        return None

    async def create_profit_exit(
        self, option: Option, quantity: Decimal, entry_order_id: str
    ):
        """Create a limit exit order at 100% profit."""
        # Try to determine entry price from the existing order
        try:
            self.tastytrade_client.ensure_authorized()
            session = self.tastytrade_client.get_session()
            accounts = Account.get(session)
            account = self._select_account(accounts)
            if not account:
                return None
            orders = account.get_live_orders(session)
            entry_order = None
            for o in orders:
                if str(getattr(o, "id", "")) == str(entry_order_id):
                    entry_order = o
                    break
            entry_price = None
            if entry_order:
                # Prefer actual fill price from leg fills if present (most accurate)
                try:
                    legs = getattr(entry_order, "legs", []) or []
                    if legs and getattr(legs[0], "fills", None):
                        fills = getattr(legs[0], "fills") or []
                        if fills and getattr(fills[0], "fill_price", None) is not None:
                            entry_price = Decimal(str(fills[0].fill_price))
                except Exception:
                    entry_price = None
                # Fallback to order.price if no fill price available
                if entry_price is None:
                    try:
                        if getattr(entry_order, "price", None) is not None:
                            entry_price = Decimal(str(entry_order.price))
                    except Exception:
                        entry_price = None
                # Fallback: try to use leg price
                if entry_price is None:
                    try:
                        if legs and getattr(legs[0], "price", None) is not None:
                            entry_price = Decimal(str(legs[0].price))
                    except Exception:
                        entry_price = None
            if entry_price is None:
                # Could not find the entry price; abort gracefully
                return None
            # Compute exit at 100% profit (2x) and round to tick increment/0.01
            profit_multiplier = Decimal("2.0")
            exit_price = (entry_price * profit_multiplier).quantize(
                self.tick_increment, rounding=ROUND_HALF_UP
            )
            # Place an exit for 50% of the filled quantity (rounded down)
            try:
                qty_int = int(quantity)
            except Exception:
                qty_int = 0
            # 50% exit: round up to avoid 0 for small sizes (i.e., 1 -> 1)
            exit_qty = max(0, (qty_int + 1) // 2)
            if exit_qty <= 0:
                # No exit to place (quantity too small)
                return None
            return await self.place_limit_order(
                option, Decimal(str(exit_qty)), OrderAction.SELL_TO_CLOSE, exit_price
            )
        except Exception as exc:
            print(f"Error creating profit exit {exc}")
            return None
