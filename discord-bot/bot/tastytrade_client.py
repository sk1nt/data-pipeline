"""Helper for interacting with TastyTrade accounts via OAuthSession."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional
from tastytrade.order import NewOrder, Leg, OrderType, OrderTimeInForce, OrderAction, InstrumentType
from tastytrade.instruments import Future, FutureProduct

try:  # pragma: no cover - optional dependency
    from tastytrade.session import Session
    from tastytrade.account import Account, AccountBalance
except ImportError as exc:  # pragma: no cover - optional dependency
    print(f"ImportError in tastytrade_client: {exc}")
    Session = None  # type: ignore
    Account = None  # type: ignore
    AccountBalance = None  # type: ignore


@dataclass
class AccountSummary:
    account_number: str
    nickname: Optional[str]
    account_type: str
    buying_power: float
    net_liq: float
    cash_balance: float


class TastyTradeClient:
    """Thin wrapper around OAuthSession for account/balance lookups."""

    def __init__(
        self,
        *,
        client_secret: str,
        refresh_token: str,
        default_account: Optional[str] = None,
        use_sandbox: bool = False,
        dry_run: bool = True,
    ) -> None:
        if Session is None or Account is None:
            raise RuntimeError("tastytrade package is not installed")
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._use_sandbox = use_sandbox
        self._dry_run = dry_run
        self._session: Optional[Session] = None
        self._lock = threading.Lock()
        self._accounts: Dict[str, Account] = {}
        self._active_account = default_account
        self._symbol_cache: Dict[str, Dict[str, any]] = {}


    @property
    def active_account(self) -> Optional[str]:
        return self._active_account

    def set_active_account(self, account_number: str) -> bool:
        account_number = account_number.strip()
        with self._lock:
            session = self._ensure_session()
            self._refresh_accounts(session)
            if account_number not in self._accounts:
                return False
            self._active_account = account_number
            return True

    def get_accounts(self) -> list:
        with self._lock:
            session = self._ensure_session()
            self._refresh_accounts(session)
            return [{'account-number': acc.account_number, 'description': acc.nickname or 'N/A'} for acc in self._accounts.values()]

    def get_account_summary(self) -> AccountSummary:
        with self._lock:
            try:
                session = self._ensure_session()
                account = self._ensure_active_account(session)
            except Exception as exc:
                msg = str(exc)
                print(f"TastyTrade session/account init failed: {msg}")
                if 'invalid_grant' in msg or 'Grant revoked' in msg or 'invalid_token' in msg:
                    raise RuntimeError(
                        "TastyTrade authentication failed (refresh token invalid or revoked). "
                        "Run `python scripts/get_tastytrade_refresh_token.py --sandbox` to retrieve a new token, "
                        "update your .env with TASTYTRADE_REFRESH_TOKEN, and restart the bot."
                    )
                raise
            balances = account.get_balances(session)
            bp = self._pick_buying_power(balances)
            return AccountSummary(
                account_number=account.account_number,
                nickname=account.nickname,
                account_type=account.margin_or_cash,
                buying_power=bp,
                net_liq=self._to_float(balances.net_liquidating_value),
                cash_balance=self._to_float(balances.cash_balance),
            )

    def get_account_overview(self) -> Dict[str, float]:
        with self._lock:
            session = self._ensure_session()
            account = self._ensure_active_account(session)
            balances = account.get_balances(session)
            overview = {
                "account_number": account.account_number,
                "available_trading_funds": self._to_float(balances.available_trading_funds),
                "equity_buying_power": self._to_float(balances.equity_buying_power),
                "derivative_buying_power": self._to_float(balances.derivative_buying_power),
                "day_trading_buying_power": self._to_float(balances.day_trading_buying_power),
                "net_liquidating_value": self._to_float(balances.net_liquidating_value),
                "cash_balance": self._to_float(balances.cash_balance),
                "margin_equity": self._to_float(balances.margin_equity),
                "maintenance_requirement": self._to_float(balances.maintenance_requirement),
                "day_trade_excess": self._to_float(balances.day_trade_excess),
                "pending_cash": self._to_float(balances.pending_cash),
            }
            return overview

    def get_trading_status(self) -> Dict[str, any]:
        with self._lock:
            session = self._ensure_session()
            account = self._ensure_active_account(session)
            # Use the session to make API call to trading-status endpoint
            return session._get(f'/accounts/{account.account_number}/trading-status')

    def get_positions(self) -> list:
        with self._lock:
            session = self._ensure_session()
            account = self._ensure_active_account(session)
            positions = account.get_positions(session)
            return [pos.__dict__ for pos in positions]

    def get_orders(self) -> list:
        with self._lock:
            session = self._ensure_session()
            account = self._ensure_active_account(session)
            orders = account.get_orders(session)
            return [order.__dict__ for order in orders]

    def get_futures_list(self) -> list:
        # Return a list of common futures symbols with descriptions
        return [
            {'symbol': '/NQ:XCME', 'description': 'E-mini Nasdaq-100 Futures'},
            {'symbol': '/ES:XCME', 'description': 'E-mini S&P 500 Futures'},
            {'symbol': '/MNQ:XCME', 'description': 'Micro E-mini Nasdaq-100 Futures'},
            {'symbol': '/MES:XCME', 'description': 'Micro E-mini S&P 500 Futures'},
            {'symbol': '/RTY:XCME', 'description': 'E-mini Russell 2000 Futures'},
            {'symbol': '/YM:XCME', 'description': 'E-mini Dow Futures'},
        ]

    # ------------------------------------------------------------------
    # internal helpers

    def _ensure_session(self) -> Session:
        assert Session is not None  # for type checkers
        if self._session is None:
            self._session = Session(
                provider_secret=self._client_secret,
                refresh_token=self._refresh_token,
                is_test=self._use_sandbox,
            )
        else:
            # refresh if token expired
            try:
                if datetime.now(timezone.utc) >= getattr(self._session, "session_expiration", datetime.now(timezone.utc)):
                    self._session.refresh()
            except Exception as exc:  # pragma: no cover - handle invalid grant
                msg = str(exc).lower()
                if 'invalid_grant' in msg or 'grant revoked' in msg or 'invalid_token' in msg:
                    # Attempt to recreate session once with existing refresh_token
                    try:
                        self._session = Session(
                            provider_secret=self._client_secret,
                            refresh_token=self._refresh_token,
                            is_test=self._use_sandbox,
                        )
                        return self._session
                    except Exception:
                        raise
                raise
        return self._session

    def set_refresh_token(self, refresh_token: str) -> None:
        """Replace stored refresh token and reinitialize session with new token."""
        refresh_token = (refresh_token or '').strip()
        if not refresh_token:
            raise ValueError('refresh_token must be provided')
        self._refresh_token = refresh_token
        # recreate session using new refresh token
        self._session = Session(provider_secret=self._client_secret, refresh_token=self._refresh_token, is_test=self._use_sandbox)
        # clear cached symbols to avoid possible stale mappings
        self._symbol_cache.clear()

    def _resolve_front_month_symbol(self, product_code: str) -> Optional[str]:
        """Return front-month TW symbol (like /NQZ5) for a product code like 'NQ'. Caches to avoid repeated API calls."""
        product_code = product_code.upper().replace('/', '') if product_code else ''
        if not product_code:
            return None
        cache = self._symbol_cache.get(product_code)
        now = datetime.now(timezone.utc)
        if cache and (now - cache.get('ts', now)).total_seconds() < 60 * 60:
            return cache.get('symbol')
        # Fetch futures by product code
        try:
            session = self._ensure_session()
            # Use Future.get with product_codes to enumerate contracts
            futures = Future.get(session, symbols=None, product_codes=[product_code])
            candidates = [f for f in futures if getattr(f, 'is_tradeable', True)]
            # Prefer next_active_month, then active_month, then the closest expiration
            selected: Optional[Future] = None
            for f in candidates:
                if getattr(f, 'next_active_month', False):
                    selected = f
                    break
            if selected is None:
                for f in candidates:
                    if getattr(f, 'active_month', False):
                        selected = f
                        break
            if selected is None and candidates:
                candidates.sort(key=lambda f: getattr(f, 'expiration_date', datetime.max))
                selected = candidates[0]
            if selected:
                # Use streamer_symbol if available, otherwise symbol
                sym = getattr(selected, 'streamer_symbol', None) or getattr(selected, 'symbol', None) or ''
                sym = f"/{sym}" if sym and not sym.startswith('/') else sym
                self._symbol_cache[product_code] = {'symbol': sym, 'ts': now}
                return sym
        except Exception:
            # if we can't fetch front month, return None and let caller fallback
            return None
        return None

    def _refresh_accounts(self, session: Session) -> None:
        accounts = Account.get(session)
        self._accounts = {acct.account_number: acct for acct in accounts}
        if not self._active_account and accounts:
            self._active_account = accounts[0].account_number

    def _ensure_active_account(self, session: Session) -> Account:
        self._refresh_accounts(session)
        if not self._active_account or self._active_account not in self._accounts:
            if not self._accounts:
                raise RuntimeError("No TastyTrade accounts available")
            self._active_account = next(iter(self._accounts))
        return self._accounts[self._active_account]

    @staticmethod
    def _to_float(value: Optional[float]) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _pick_buying_power(balances) -> float:
        # Prioritize derivative buying power for futures/options
        return (
            balances.derivative_buying_power or
            balances.equity_buying_power or
            balances.day_trading_buying_power or
            0.0
        )

    def place_market_order_with_tp(
        self,
        symbol: str,
        action: str,
        quantity: int,
        tp_ticks: float,
        tick_size: float = 0.25,
        dry_run: Optional[bool] = None,
    ) -> str:
        """Place a market order and a TP limit order."""
        with self._lock:
            session = self._ensure_session()
            account = self._ensure_active_account(session)

            # Normalize symbol
            symbol = self._normalize_symbol(symbol)

            # Determine instrument type enum
            inst_type = InstrumentType.FUTURE if symbol.startswith('/') else InstrumentType.EQUITY

            # Get current price (approximate from quote)
            current_price = self._get_current_price(session, symbol)

            # Calculate TP price
            if action.upper() == 'BUY':
                tp_price = current_price + (tp_ticks * tick_size)
            else:
                tp_price = current_price - (tp_ticks * tick_size)

            # Build Leg and NewOrder using SDK models
            leg_action = OrderAction.BUY if action.upper() == 'BUY' else OrderAction.SELL
            leg = Leg(instrument_type=inst_type, symbol=symbol, action=leg_action, quantity=quantity)
            market_order = NewOrder(time_in_force=OrderTimeInForce.DAY, order_type=OrderType.MARKET, legs=[leg])
            print(f"Sending market NewOrder (dry-run={self._use_sandbox}): {market_order}")
            # Use account.place_order to post the typed model (dry-run in sandbox)
            try:
                eff_dry = self._dry_run if dry_run is None else bool(dry_run)
                placed_market = account.place_order(session, market_order, dry_run=eff_dry)
                market_order_id = getattr(placed_market, 'id', None)
            except Exception as exc:
                msg = str(exc)
                print(f"Market order failed: {msg}")
                if 'invalid_grant' in msg or 'Grant revoked' in msg or 'invalid_token' in msg:
                    raise RuntimeError(
                        "TastyTrade authentication failed (refresh token invalid or revoked). "
                        "Run `python scripts/get_tastytrade_refresh_token.py --sandbox` to retrieve a new token, "
                        "update your .env with TASTYTRADE_REFRESH_TOKEN, and restart the bot."
                    )
                raise

            # Place TP limit order
            tp_action = OrderAction.BUY if action.upper() == 'SELL' else OrderAction.SELL
            tp_leg = Leg(instrument_type=inst_type, symbol=symbol, action=tp_action, quantity=quantity)
            tp_order = NewOrder(time_in_force=OrderTimeInForce.GTC, order_type=OrderType.LIMIT, price=tp_price, legs=[tp_leg])
            print(f"Sending TP NewOrder (dry-run={self._use_sandbox}): {tp_order}")
            try:
                placed_tp = account.place_order(session, tp_order, dry_run=eff_dry)
                tp_order_id = getattr(placed_tp, 'id', None)
            except Exception as exc:
                msg = str(exc)
                print(f"TP order failed: {msg}")
                if 'invalid_grant' in msg or 'Grant revoked' in msg or 'invalid_token' in msg:
                    raise RuntimeError(
                        "TastyTrade authentication failed (refresh token invalid or revoked). "
                        "Run `python scripts/get_tastytrade_refresh_token.py --sandbox` to retrieve a new token, "
                        "update your .env with TASTYTRADE_REFRESH_TOKEN, and restart the bot."
                    )
                raise
            # No additional outer error handling past TP order except

            return f"Placed market {action.lower()} {quantity} {symbol} (ID: {market_order_id}) and TP at {tp_price:.2f} (ID: {tp_order_id})"

    def _normalize_symbol(self, symbol: str) -> str:
        symbol = symbol.upper().strip()
        futures = {'NQ', 'MNQ', 'ES', 'MES', 'RTY', 'YM'}
        # if already a futures contract like /NQZ5 or NQZ5, ensure leading '/'
        if symbol.startswith('/'):
            return symbol
        # check product codes
        if symbol in futures:
            resolved = self._resolve_front_month_symbol(symbol)
            if resolved:
                return resolved
            # fallback: return a reasonable default with X CME code
            return f"/{symbol}Z9"
        return symbol

    def _get_current_price(self, session, symbol: str) -> float:
        # Get quote
        if symbol.startswith('/'):
            # Future
            quote_data = session._get(f'/quotes/futures?symbol={symbol}')['data']['items']
        else:
            # Equity
            quote_data = session._get(f'/quotes/equities?symbol={symbol}')['data']['items']
        if quote_data:
            bid = quote_data[0].get('bid-price')
            ask = quote_data[0].get('ask-price')
            if bid and ask:
                return (bid + ask) / 2
        # Fallback
        return 4000.0
