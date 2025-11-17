"""Helper for interacting with TastyTrade accounts via OAuthSession."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

try:  # pragma: no cover - optional dependency
    from tastytrade import OAuthSession
    from tastytrade.account import Account, AccountBalance
except ImportError:  # pragma: no cover
    OAuthSession = None  # type: ignore
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
        if OAuthSession is None or Account is None:
            raise RuntimeError("tastytrade package is not installed")
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._use_sandbox = use_sandbox
        self._session: Optional[OAuthSession] = None
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

    # ------------------------------------------------------------------
    # internal helpers

    def _ensure_session(self) -> OAuthSession:
        assert OAuthSession is not None  # for type checkers
        if self._session is None:
            self._session = OAuthSession(
                provider_secret=self._client_secret,
                refresh_token=self._refresh_token,
                is_test=self._use_sandbox,
            )
        else:
            # refresh if token expired
            if datetime.now(timezone.utc) >= getattr(self._session, "session_expiration", datetime.now(timezone.utc)):
                self._session.refresh()
        return self._session

    def _refresh_accounts(self, session: OAuthSession) -> None:
        accounts = Account.get(session)
        self._accounts = {acct.account_number: acct for acct in accounts}
        if not self._active_account and accounts:
            self._active_account = accounts[0].account_number

    def _ensure_active_account(self, session: OAuthSession) -> Account:
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

    def _pick_buying_power(self, balances: AccountBalance) -> float:
        for attr in (
            "available_trading_funds",
            "equity_buying_power",
            "derivative_buying_power",
            "day_trading_buying_power",
        ):
            value = getattr(balances, attr, None)
            if value is not None:
                return self._to_float(value)
        return self._to_float(balances.net_liquidating_value)
