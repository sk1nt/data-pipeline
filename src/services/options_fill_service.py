import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional
from datetime import datetime, timezone, date
from tastytrade import Account
from tastytrade.instruments import get_option_chain, Option, InstrumentType
from tastytrade.market_data import get_market_data
from tastytrade.order import NewOrder, OrderAction, OrderTimeInForce, OrderType

# Import via absolute paths so the module works when `src` is on sys.path
from services.tastytrade_client import tastytrade_client, TastytradeAuthError
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

    async def place_limit_order(
        self, option: Option, quantity: Decimal, action: OrderAction, price: Decimal
    ) -> Optional[str]:
        """Place a limit order for the option."""
        # Verify auth before attempting to place an order
        self.tastytrade_client.ensure_authorized()
        session = self.tastytrade_client.get_session()
        accounts = Account.get(session)
        if not accounts:
            return None
        account = accounts[0]

        leg = option.build_leg(quantity, action)
        order = NewOrder(
            time_in_force=OrderTimeInForce.DAY,
            order_type=OrderType.LIMIT,
            legs=[leg],
            price=price,
        )

        try:
            response = account.place_order(
                session, order, dry_run=config.tastytrade_use_sandbox
            )
            return str(response.order.id) if hasattr(response, "order") else None
        except TastytradeAuthError as e:
            print(f"TastyTrade auth error while placing order: {e}")
            raise
        except Exception as e:
            print(f"Error placing order: {e}")
            return None

    async def check_order_filled(self, order_id: str) -> bool:
        """Check if an order is filled."""
        try:
            self.tastytrade_client.ensure_authorized()
            session = self.tastytrade_client.get_session()
            accounts = Account.get(session)
            if accounts:
                account = accounts[0]
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
            if accounts:
                account = accounts[0]
                account.delete_order(session, order_id)
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
    ) -> Optional[str]:
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
            return None

        current_price = mid_price
        for attempt in range(self.max_retries + 1):
            print(f"Attempt {attempt + 1}: Placing order at {current_price}")
            order_id = await self.place_limit_order(
                option, Decimal(str(quantity)), order_action, current_price
            )
            if not order_id:
                return None

            # Wait for fill
            await asyncio.sleep(self.timeout_seconds)
            if await self.check_order_filled(order_id):
                print(f"Order filled on attempt {attempt + 1}")

                # Create profit-taking order if filled
                if order_action == OrderAction.BUY_TO_OPEN:
                    await self.create_profit_exit(
                        option, Decimal(str(quantity)), order_id
                    )

                return order_id

            # Adjust price
            if attempt < self.max_retries:
                await self.cancel_order(order_id)
                if order_action in [OrderAction.BUY_TO_OPEN]:
                    current_price += self.tick_increment * (attempt + 1)
                else:
                    current_price -= self.tick_increment * (attempt + 1)
                current_price = current_price.quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

        # Leave as DAY order
        print("Order not filled, leaving as DAY order")
        return order_id

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
        # Simplified: assume entry price known, create exit at 2x
        # TODO: Implement proper profit calculation
        exit_price = Decimal("1.00")  # Placeholder
        await self.place_limit_order(
            option, quantity, OrderAction.SELL_TO_CLOSE, exit_price
        )
