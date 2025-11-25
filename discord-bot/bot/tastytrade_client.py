"""Helper for interacting with TastyTrade accounts via OAuthSession."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

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
    ) -> None:
        if Session is None or Account is None:
            raise RuntimeError("tastytrade package is not installed")
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._use_sandbox = use_sandbox
        self._session: Optional[Session] = None
        self._lock = threading.Lock()
        self._accounts: Dict[str, Account] = {}
        self._active_account = default_account

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

    def list_account_numbers(self) -> list[str]:
        with self._lock:
            session = self._ensure_session()
            self._refresh_accounts(session)
            return list(self._accounts.keys())

    def get_account_summary(self) -> AccountSummary:
        with self._lock:
            session = self._ensure_session()
            account = self._ensure_active_account(session)
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
            return session._get(f'/accounts/{account.id}/trading-status')

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
            if datetime.now(timezone.utc) >= getattr(self._session, "session_expiration", datetime.now(timezone.utc)):
                self._session.refresh()
        return self._session

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
    ) -> str:
        """Place a market order and a TP limit order."""
        with self._lock:
            session = self._ensure_session()
            account = self._ensure_active_account(session)

            # Normalize symbol
            symbol = self._normalize_symbol(symbol)

            # Get current price (approximate from quote)
            current_price = self._get_current_price(session, symbol)

            # Calculate TP price
            if action.upper() == 'BUY':
                tp_price = current_price + (tp_ticks * tick_size)
            else:
                tp_price = current_price - (tp_ticks * tick_size)

            # Place market order
            market_order_data = {
                "time-in-force": "Day",
                "order-type": "Market",
                "legs": [
                    {
                        "instrument-type": "Equity" if not symbol.startswith('/') else "Future",
                        "symbol": symbol,
                        "quantity": quantity,
                        "action": action.capitalize()
                    }
                ]
            }
            market_order_id = session._post(f'/accounts/{account.id}/orders', market_order_data)['data']['id']

            # Place TP limit order
            tp_action = 'Buy' if action.upper() == 'SELL' else 'Sell'
            tp_order_data = {
                "time-in-force": "GTC",
                "order-type": "Limit",
                "price": tp_price,
                "legs": [
                    {
                        "instrument-type": "Equity" if not symbol.startswith('/') else "Future",
                        "symbol": symbol,
                        "quantity": quantity,
                        "action": tp_action
                    }
                ]
            }
            tp_order_id = session._post(f'/accounts/{account.id}/orders', tp_order_data)['data']['id']

            return f"Placed market {action.lower()} {quantity} {symbol} (ID: {market_order_id}) and TP at {tp_price:.2f} (ID: {tp_order_id})"

    def _normalize_symbol(self, symbol: str) -> str:
        symbol = symbol.upper().strip()
        if symbol in ['NQ', 'MNQ', 'ES', 'MES']:
            return f"/{symbol}:XCME"
        return f"/{symbol}"

    def _get_current_price(self, session, symbol: str) -> float:
        # Get quote
        if symbol.startswith('/'):
            # Future
            quote_data = session._get(f'/instruments/future-quotes?symbol={symbol}')['data']['items']
        else:
            # Equity
            quote_data = session._get(f'/instruments/equity-quotes?symbol={symbol}')['data']['items']
        if quote_data:
            bid = quote_data[0].get('bid-price')
            ask = quote_data[0].get('ask-price')
            if bid and ask:
                return (bid + ask) / 2
        # Fallback
        return 4000.0
