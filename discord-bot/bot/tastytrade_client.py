"""Helper for interacting with TastyTrade accounts via OAuthSession."""

from __future__ import annotations

import json
import threading
import re
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
import logging
import hashlib
import time
from tastytrade.order import (
    NewOrder,
    Leg,
    OrderType,
    OrderTimeInForce,
    OrderAction,
    InstrumentType,
)
from tastytrade.instruments import Future

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


LOGGER = logging.getLogger(__name__)
AUTH_ERROR_TEXT = (
    "TastyTrade authentication failed (refresh token invalid or revoked). "
    "Use `set_refresh_token(...)` or run `python scripts/get_tastytrade_refresh_token.py --sandbox` "
    "to obtain a new token and then call `set_refresh_token` or restart the bot."
)


class TastytradeAuthError(RuntimeError):
    """Raised when the refresh token is invalid, revoked, or session requires reauth."""


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
        self._session_expiration: Optional[datetime] = None
        self._lock = threading.Lock()
        self._accounts: Dict[str, Account] = {}
        self._active_account = default_account
        self._symbol_cache: Dict[str, Dict[str, any]] = {}
        self._needs_reauth: bool = False
        self._reauth_backoff_seconds: int = 60
        self._reauth_thread: Optional[threading.Thread] = None
        # Start reauth worker thread (best-effort; don't raise on failure)
        try:
            self._start_reauth_worker()
        except Exception:
            LOGGER.warning(
                "_start_reauth_worker failed during init; continuing without background reauth worker"
            )

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
            return [
                {
                    "account-number": acc.account_number,
                    "description": acc.nickname or "N/A",
                }
                for acc in self._accounts.values()
            ]

    def get_account_summary(self) -> AccountSummary:
        with self._lock:
            try:
                session = self._ensure_session()
                account = self._ensure_active_account(session)
            except Exception as exc:
                msg = str(exc)
                LOGGER.warning("TastyTrade session/account init failed: %s", msg)
                self._raise_on_auth_error(exc)
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
                "available_trading_funds": self._to_float(
                    balances.available_trading_funds
                ),
                "equity_buying_power": self._to_float(balances.equity_buying_power),
                "derivative_buying_power": self._to_float(
                    balances.derivative_buying_power
                ),
                "day_trading_buying_power": self._to_float(
                    balances.day_trading_buying_power
                ),
                "net_liquidating_value": self._to_float(balances.net_liquidating_value),
                "cash_balance": self._to_float(balances.cash_balance),
                "margin_equity": self._to_float(balances.margin_equity),
                "maintenance_requirement": self._to_float(
                    balances.maintenance_requirement
                ),
                "day_trade_excess": self._to_float(balances.day_trade_excess),
                "pending_cash": self._to_float(balances.pending_cash),
            }
            return overview

    def get_trading_status(self) -> Dict[str, any]:
        with self._lock:
            session = self._ensure_session()
            account = self._ensure_active_account(session)
            # Use the session to make API call to trading-status endpoint
            return session._get(f"/accounts/{account.account_number}/trading-status")

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
            # Try SDK helper for today's orders; fall back to raw endpoint
            try:
                orders = account.get_live_orders(session)
            except Exception:
                raw = session._get(f"/accounts/{account.account_number}/orders")
                orders = raw.get("data", {}).get("items", [])
            return [order.__dict__ for order in orders]

    def get_futures_list(self) -> list:
        # Try to return a dynamic list via SDK; fallback to reasonable static list
        try:
            return self.list_futures()
        except Exception:
            return [
                {"symbol": "/NQ:XCME", "description": "E-mini Nasdaq-100 Futures"},
                {"symbol": "/ES:XCME", "description": "E-mini S&P 500 Futures"},
                {
                    "symbol": "/MNQ:XCME",
                    "description": "Micro E-mini Nasdaq-100 Futures",
                },
                {"symbol": "/MES:XCME", "description": "Micro E-mini S&P 500 Futures"},
                {"symbol": "/RTY:XCME", "description": "E-mini Russell 2000 Futures"},
                {"symbol": "/YM:XCME", "description": "E-mini Dow Futures"},
            ]

    def list_futures(self, product_codes: Optional[list] = None) -> list:
        """Return a list of futures contracts for given product codes (e.g., ['NQ','ES']).

        Each returned item is a dict with keys: `symbol`, `streamer_symbol`, `expiration_date`, `is_tradeable`.
        """
        with self._lock:
            session = self._ensure_session()
            # Use SDK to fetch futures; product_codes is optional
            try:
                futures = Future.get(session, symbols=None, product_codes=product_codes)
            except Exception:
                raise
            results = []
            for f in futures:
                try:
                    results.append(
                        {
                            "symbol": getattr(f, "symbol", None),
                            "streamer_symbol": getattr(f, "streamer_symbol", None),
                            "expiration_date": getattr(f, "expiration_date", None),
                            "is_tradeable": getattr(f, "is_tradeable", None),
                            "product_code": getattr(f, "product_code", None),
                            "description": getattr(f, "description", None),
                        }
                    )
                except Exception:
                    continue
            return results

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
            self._session_expiration = self._derive_expiration(self._session)
        else:
            # refresh if token expired
            try:
                if (
                    self._session_expiration
                    and datetime.now(timezone.utc) >= self._session_expiration
                ):
                    self._session.refresh()
                    self._session_expiration = self._derive_expiration(self._session)
            except Exception as exc:  # pragma: no cover - handle invalid grant
                self._raise_on_auth_error(exc)
                raise
        return self._session

    def ensure_authorized(self) -> bool:
        """Ensure that the session is authorized/valid. Returns True if valid.

        Will attempt to refresh or reinitialize the session when needed. If the
        refresh token is invalid or revoked, raises TastytradeAuthError.
        """
        with self._lock:
            # fast-path: if we have a session and it hasn't expired
            try:
                if self._session and not self._needs_reauth:
                    exp = self._session_expiration
                    if exp and datetime.now(timezone.utc) < exp:
                        return True
                # fall-back to creating or refreshing session via _ensure_session
                self._ensure_session()
                return True
            except TastytradeAuthError:
                # propagate so callers can handle it specifically
                raise
            except Exception:
                # For other errors, wrap or propagate
                raise

    def set_refresh_token(self, refresh_token: str) -> None:
        """Replace stored refresh token and reinitialize session with new token."""
        refresh_token = (refresh_token or "").strip()
        if not refresh_token:
            raise ValueError("refresh_token must be provided")
        with self._lock:
            self._refresh_token = refresh_token
            # recreate session using new refresh token
            try:
                self._session = Session(
                    provider_secret=self._client_secret,
                    refresh_token=self._refresh_token,
                    is_test=self._use_sandbox,
                )
                self._session_expiration = self._derive_expiration(self._session)
                self._needs_reauth = False
            except Exception:
                # If immediate creation fails, mark for reauth attempts
                self._session = None
                self._session_expiration = None
                self._needs_reauth = True
            # clear cached symbols to avoid possible stale mappings
            self._symbol_cache.clear()

    def set_dry_run(self, flag: bool) -> None:
        """Turn on/off dry-run (prevent actual order execution)."""
        self._dry_run = bool(flag)

    def get_session(self) -> Session:
        """Return an authorized Session, refreshing if needed."""
        with self._lock:
            return self._ensure_session()

    def set_use_sandbox(self, flag: bool) -> None:
        """Switch the client runtime environment between sandbox and live.

        Toggles the `is_test` flag used during session initialization, and
        attempts to reinitialize the session. If initialization fails, the
        client will mark itself as needing reauth and clear the session.
        """
        self._use_sandbox = bool(flag)
        # Clear session so the next call reinitializes using the updated flag
        with self._lock:
            self._session = None
            self._session_expiration = None
            try:
                self._session = Session(
                    provider_secret=self._client_secret,
                    refresh_token=self._refresh_token,
                    is_test=self._use_sandbox,
                )
                self._session_expiration = self._derive_expiration(self._session)
                self._needs_reauth = False
            except Exception:
                self._session = None
                self._session_expiration = None
                self._needs_reauth = True

    def _mark_needs_reauth(self) -> None:
        """Mark that the session needs reauthorization and clear current session."""
        self._needs_reauth = True
        self._session = None
        self._session_expiration = None

    def get_auth_status(self) -> Dict[str, any]:
        """Return authentication/session/account status."""
        status: Dict[str, any] = {
            "session_valid": False,
            "active_account": self._active_account,
            "accounts": [],
            "use_sandbox": self._use_sandbox,
            "dry_run": getattr(self, "_dry_run", True),
            "error": None,
        }
        try:
            session = self._ensure_session()
            # Refresh accounts
            self._refresh_accounts(session)
            status["accounts"] = list(self._accounts.keys())
            status["active_account"] = self._active_account
            status["session_valid"] = True
            status["session_expiration"] = self._session_expiration or getattr(
                session, "session_expiration", None
            )
        except Exception as exc:
            status["error"] = str(exc)
            # If it's an auth issue, mark for reauth
            msg = str(exc).lower() if exc else ""
            if any(key in msg for key in ("invalid_grant", "grant revoked", "invalid_token")):
                self._mark_needs_reauth()
        status["needs_reauth"] = getattr(self, "_needs_reauth", False)
        status["refresh_token_hash"] = self._hash_token(
            getattr(self, "_refresh_token", "")
        )
        return status

    def _hash_token(self, token: str) -> str:
        if not token:
            return ""
        # Return short SHA256 fingerprint to display safely
        h = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return h[:12]

    def _start_reauth_worker(self) -> None:
        # Defensive reauth worker start: avoid raising if threading is missing or failed
        if (
            self._reauth_thread
            and getattr(self._reauth_thread, "is_alive", lambda: False)()
        ):
            return
        try:
            # If 'threading.Thread' exists, use it; otherwise try to import threading dynamically
            thread_cls = getattr(threading, "Thread", None)
            if thread_cls is None:
                import importlib as _importlib

                threading_mod = _importlib.import_module("threading")
                thread_cls = getattr(threading_mod, "Thread", None)
            if thread_cls is None:
                LOGGER.warning(
                    "threading.Thread is not available; reauth worker will not be started"
                )
                return
            self._reauth_thread = thread_cls(target=self._reauth_worker, daemon=True)
            self._reauth_thread.start()
        except Exception as exc:
            LOGGER.warning("Failed to start reauth worker thread: %s", exc)

    def _reauth_worker(self) -> None:
        # Background thread that tries to reinitialize session when needed
        while True:
            try:
                if self._needs_reauth:
                    try:
                        self._session = Session(
                            provider_secret=self._client_secret,
                            refresh_token=self._refresh_token,
                            is_test=self._use_sandbox,
                        )
                        self._session_expiration = self._derive_expiration(
                            self._session
                        )
                        self._needs_reauth = False
                        # keep symbol cache fresh
                        self._symbol_cache.clear()
                        LOGGER.info("TastyTrade session reinitialized by reauth worker")
                    except Exception:
                        LOGGER.debug("TastyTrade reauth attempt failed; will retry")
                time.sleep(self._reauth_backoff_seconds)
            except Exception:
                time.sleep(self._reauth_backoff_seconds)

    def _resolve_front_month_symbol(self, product_code: str) -> Optional[str]:
        """Return front-month TW symbol (like /NQZ5) for a product code like 'NQ'. Caches to avoid repeated API calls."""
        product_code = product_code.upper().replace("/", "") if product_code else ""
        if not product_code:
            return None
        cache = self._symbol_cache.get(product_code)
        now = datetime.now(timezone.utc)
        if cache and (now - cache.get("ts", now)).total_seconds() < 60 * 60:
            return cache.get("symbol")
        # Fetch futures by product code
        try:
            session = self._ensure_session()
            # Use Future.get with product_codes to enumerate contracts
            futures = Future.get(session, symbols=None, product_codes=[product_code])
            candidates = [f for f in futures if getattr(f, "is_tradeable", True)]
            # Always select the closest expiration
            selected: Optional[Future] = None
            if candidates:
                candidates.sort(
                    key=lambda f: getattr(f, "expiration_date", datetime.max)
                )
                selected = candidates[0]
            if selected:
                # Use streamer_symbol if available, otherwise symbol
                sym = (
                    getattr(selected, "streamer_symbol", None)
                    or getattr(selected, "symbol", None)
                    or ""
                )
                sym = f"/{sym}" if sym and not sym.startswith("/") else sym
                self._symbol_cache[product_code] = {"symbol": sym, "ts": now}
                return sym
        except Exception:
            # if we can't fetch front month, return None and let caller fallback
            return None
        return None

    def _resolve_future_contract_symbol(
        self, session: Session, symbol: str
    ) -> Optional[str]:
        """Resolve a user-provided future symbol to the TastyTrade contract symbol (e.g., '/NQZ5').

        Attempts several common normalizations and uses the SDK to find a matching
        Future object, preferring the SDK-provided `symbol` value which the API
        expects when placing orders.
        """
        if not symbol or not isinstance(symbol, str):
            return None
        raw = symbol.strip().upper()
        # If already present in cache, return it quickly
        if raw in self._symbol_cache:
            return self._symbol_cache[raw].get("resolved")
        # Build candidate forms to match against SDK Future.symbol or streamer_symbol
        cand = raw.lstrip("/")
        candidates = set()
        candidates.add(cand)
        candidates.add("/" + cand)
        if ":" in cand:
            base = cand.split(":", 1)[0]
            candidates.add(base)
            candidates.add("/" + base)
        # If year supplied as 2-digits (e.g., NQZ25), add single-digit variant (NQZ5)
        m = re.match(r"^(?P<prod>[A-Z]{1,4})(?P<month>[A-Z])(?P<year>\d{2})$", cand)
        if m:
            prod = m.group("prod")
            month = m.group("month")
            year = m.group("year")
            short = f"{prod}{month}{year[-1]}"
            candidates.add(short)
            candidates.add("/" + short)
        # If no explicit month/year provided, try resolving front month for the product
        product_code = None
        pm = re.match(r"^(?P<prod>NQ|MNQ|ES|MES|RTY|YM)$", cand)
        if pm:
            product_code = pm.group("prod")
        # Fetch SDK futures for the product, if available
        try:
            # If we have a product code, query futures for that product; otherwise
            # query all futures by product_codes inferred from input
            futures = None
            if product_code:
                futures = Future.get(
                    session, symbols=None, product_codes=[product_code]
                )
            else:
                # Try to guess product code from the first alpha characters
                maybe_prod = re.match(r"^(?P<prod>[A-Z]+)", cand)
                if maybe_prod:
                    try:
                        futures = Future.get(
                            session,
                            symbols=None,
                            product_codes=[maybe_prod.group("prod")],
                        )
                    except Exception:
                        futures = Future.get(session, symbols=None)
                else:
                    futures = Future.get(session, symbols=None)
            if not futures:
                raise Exception("no futures returned")
            # search among futures for a match
            matches = []
            for f in futures:
                f_sym = getattr(f, "symbol", None)
                f_stream = getattr(f, "streamer_symbol", None)
                try:
                    # Compare by symbol (e.g., '/NQZ5')
                    if f_sym and any(f_sym.upper() == c.upper() for c in candidates):
                        matches.append(f_sym)
                        continue
                    # Compare streamer symbol base (e.g., '/NQZ25:XCME' -> '/NQZ25')
                    if f_stream:
                        f_stream_base = f_stream.split(":", 1)[0]
                        if any(f_stream_base.upper() == c.upper() for c in candidates):
                            matches.append(getattr(f, "symbol", f_stream_base))
                            continue
                except Exception:
                    continue
            resolved = None
            if matches:
                # Prefer single-digit year format if both are present (e.g., '/NQZ5')
                single_digit = [
                    m for m in matches if re.match(r"^/[A-Z]{1,4}[A-Z]\d$", m)
                ]
                if single_digit:
                    resolved = single_digit[0]
                else:
                    resolved = matches[0]
                # If we chose a two-digit year symbol, prefer a single-digit
                # version if the futures list has one for the same product/month.
                if resolved and re.match(r"^/[A-Z]{1,4}[A-Z]\d{2}$", resolved):
                    prod = resolved[1:3]
                    month = resolved[3]
                    # create one-digit variant
                    alt = f"/{prod}{month}{resolved[-1]}"
                    for f in futures:
                        if getattr(f, "symbol", None) == alt:
                            resolved = alt
                            break
            if resolved:
                # Ensure leading slash
                if not resolved.startswith("/"):
                    resolved = "/" + resolved
                # Cache
                self._symbol_cache[raw] = {
                    "resolved": resolved,
                    "ts": datetime.now(timezone.utc),
                }
                return resolved
        except Exception:
            # ignore and fallback to heuristic below
            pass
        # Heuristic fallback: convert 2-digit year to single-digit and ensure leading '/'
        short = None
        if ":" in raw:
            raw_no_ex = raw.split(":", 1)[0]
        else:
            raw_no_ex = raw
        # Normalize by removing leading slash so regex checks work consistently
        raw_no_ex = raw_no_ex.lstrip("/")
        m2 = re.match(
            r"^(?P<prod>[A-Z]{1,4})(?P<month>[A-Z])(?P<year>\d{2})$", raw_no_ex
        )
        if m2:
            prod = m2.group("prod")
            month = m2.group("month")
            year = m2.group("year")
            short = f"/{prod}{month}{year[-1]}"
        else:
            short = "/" + raw_no_ex.lstrip("/")
        self._symbol_cache[raw] = {"resolved": short, "ts": datetime.now(timezone.utc)}
        return short

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

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order by ID. Returns raw response dict."""
        with self._lock:
            session = self._ensure_session()
            account = self._ensure_active_account(session)
            try:
                # No SDK helper; call delete endpoint directly
                resp = session._delete(
                    f"/accounts/{account.account_number}/orders/{order_id}"
                )
                return resp
            except Exception as exc:
                msg = str(exc)
                LOGGER.warning("Failed to cancel order %s: %s", order_id, msg)
                self._raise_on_auth_error(exc)
                raise

    def get_order(self, order_id: str) -> dict:
        """Return the placed order payload for given ID."""
        with self._lock:
            session = self._ensure_session()
            account = self._ensure_active_account(session)
            try:
                orders = account.get_orders(session)
                for o in orders:
                    if (
                        str(getattr(o, "id", "")) == str(order_id)
                        or getattr(o, "id", "") == order_id
                    ):
                        return o.__dict__
                # Not found; try fetching directly
                resp = session._get(
                    f"/accounts/{account.account_number}/orders/{order_id}"
                )
                return resp
            except Exception as exc:
                msg = str(exc)
                LOGGER.warning("Failed to fetch order %s: %s", order_id, msg)
                raise

    def replace_order(
        self,
        order_id: str,
        *,
        price: Optional[float] = None,
        time_in_force: Optional[OrderTimeInForce] = None,
    ) -> dict:
        """Replace an existing order (adjust price or TIF)."""
        from tastytrade.order import NewOrder

        with self._lock:
            session = self._ensure_session()
            account = self._ensure_active_account(session)
            try:
                # Build new order with provided fields; no legs required for replace
                new_order = NewOrder(
                    time_in_force=time_in_force or OrderTimeInForce.DAY,
                    order_type=OrderType.LIMIT
                    if price is not None
                    else OrderType.MARKET,
                    price=price or None,
                    legs=[],
                )
                placed = account.replace_order(session, int(order_id), new_order)
                return getattr(placed, "__dict__", placed)
            except Exception as exc:
                msg = str(exc)
                LOGGER.warning("Failed to replace order %s: %s", order_id, msg)
                raise

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
            balances.derivative_buying_power
            or balances.equity_buying_power
            or balances.day_trading_buying_power
            or 0.0
        )

    def _extract_order_id(self, placed) -> Optional[str]:
        """Return an order ID string/int from either SDK objects or raw dict responses.

        Common SDK returns:
          - PlacedOrder object with attribute `id` (int or str)
          - Raw dict from REST with 'data' -> 'id' or top-level 'id'
        """
        if placed is None:
            return None
        # If object with attribute 'id'
        try:
            oid = getattr(placed, "id", None)
            if oid is not None:
                return str(oid)
        except Exception:
            pass
        # If dict-like response
        try:
            if isinstance(placed, dict):
                # common shapes: {'data': {'id': 123}} or {'id': 123}
                data = placed.get("data") if placed.get("data") is not None else placed
                if isinstance(data, dict) and "id" in data:
                    return str(data["id"])
                if "id" in placed:
                    return str(placed["id"])
        except Exception:
            pass
        return None

    def _derive_expiration(self, session: Session) -> datetime:
        """Return a best-effort expiration for the session.

        The SDK may not expose a real expiry; fallback to a conservative window to
        avoid refreshing on every call while still renewing periodically.
        """
        exp = getattr(session, "session_expiration", None)
        if isinstance(exp, datetime):
            return exp
        return datetime.now(timezone.utc) + timedelta(minutes=20)

    def place_market_order_with_tp(
        self,
        symbol: str,
        action: str,
        quantity: int,
        tp_ticks: float,
        tick_size: float = 0.25,
        dry_run: Optional[bool] = None,
        market_price: Optional[float] = None,
    ) -> str:
        """Place market entry order, then attach limit TP as bracketed OTO (separate submits)."""
        with self._lock:
            session = self._ensure_session()
            account = self._ensure_active_account(session)

            # Normalize symbol
            symbol = self._normalize_symbol(symbol)

            # Determine instrument type enum
            inst_type = (
                InstrumentType.FUTURE
                if symbol.startswith("/")
                else InstrumentType.EQUITY
            )

            # For futures, set tick_size to 1 (points), for equities 0.25
            if inst_type == InstrumentType.FUTURE:
                tick_size = 1.0
            else:
                tick_size = 0.25

            # For futures, try to resolve the exact contract symbol the API expects
            if inst_type == InstrumentType.FUTURE:
                try:
                    resolved = self._resolve_future_contract_symbol(session, symbol)
                    if resolved:
                        symbol = resolved
                except Exception:
                    # if resolve fails, proceed with the normalized symbol
                    pass

            # Get current price (approximate from quote) or use provided market_price
            if market_price is not None:
                current_price = float(market_price)
            else:
                current_price = self._get_current_price(session, symbol)

            # Determine entry action
            if inst_type == InstrumentType.FUTURE:
                entry_action = (
                    OrderAction.BUY_TO_OPEN
                    if action.upper() == "BUY"
                    else OrderAction.SELL_TO_OPEN
                )
                # TP is opposite close
                tp_action = (
                    OrderAction.SELL_TO_CLOSE
                    if action.upper() == "BUY"
                    else OrderAction.BUY_TO_CLOSE
                )
            else:
                entry_action = (
                    OrderAction.BUY if action.upper() == "BUY" else OrderAction.SELL
                )
                # For equities, reuse entry action but invert for close (simplified; equities don't need _TO_OPEN/CLOSE)
                tp_action = (
                    OrderAction.SELL if action.upper() == "BUY" else OrderAction.BUY
                )

            # Calculate TP price directionally
            if action.upper() == "BUY":  # Long: TP above
                tp_price = current_price + (tp_ticks * tick_size)
                round_dir = "up"  # Ceiling to ensure profit
            else:  # Short: TP below
                tp_price = current_price - (tp_ticks * tick_size)
                round_dir = "down"  # Floor to ensure profit

            # Determine tick_size from instrument metadata if possible (falls back to provided)
            try:
                if (
                    inst_type == InstrumentType.FUTURE
                    and isinstance(symbol, str)
                    and symbol.startswith("/")
                ):
                    # Try to query the SDK for contract-specific tick size
                    # strip leading slash for symbols passed to the SDK
                    sym_lookup = symbol.lstrip("/")
                    futs = Future.get(session, symbols=[sym_lookup])
                    if futs:
                        f = futs[0]
                        tick_meta = getattr(f, "tick_size", None) or getattr(
                            f, "min_tick", None
                        )
                        if isinstance(tick_meta, (int, float)) and tick_meta > 0:
                            tick_size = float(tick_meta)
            except Exception:
                # If we cannot resolve tick from SDK, continue with given tick_size
                pass

            # Round TP price to the instrument's tick size and ensure it is on the correct side
            import math

            def _round_to_tick(price: float, tick: float, direction: str) -> float:
                if tick <= 0 or math.isnan(price):
                    return price
                n = price / tick
                if direction == "down":
                    n2 = math.floor(n - 1e-9)
                elif direction == "up":
                    n2 = math.ceil(n + 1e-9)
                else:
                    n2 = round(n)
                return n2 * tick

            tp_price = _round_to_tick(tp_price, tick_size, round_dir)
            # Safety nudge: Ensure TP is strictly profitable
            if action.upper() == "BUY":
                if tp_price <= current_price:
                    tp_price = _round_to_tick(
                        current_price + tick_size, tick_size, "up"
                    )
            else:
                if tp_price >= current_price:
                    tp_price = _round_to_tick(
                        current_price - tick_size, tick_size, "down"
                    )

            # Clean symbol for leg (strip exchange if present, ensure no leading / for equities)
            leg_symbol = (
                symbol
                if inst_type == InstrumentType.FUTURE
                else (symbol.lstrip("/") if isinstance(symbol, str) else symbol)
            )
            if isinstance(leg_symbol, str) and ":" in leg_symbol:
                leg_symbol = leg_symbol.split(":", 1)[0]

            # Normalize year for futures (keep as-is, but ensure single-digit if needed)
            m = re.match(
                r"^(?P<prod>NQ|MNQ|ES|MES|RTY|YM)(?P<month>[A-Z])(?P<year>\d{1,2})$",
                leg_symbol,
            )
            if m:
                prod = m.group("prod")
                month = m.group("month")
                year = m.group("year")
                leg_symbol = f"{prod}{month}{year}"

            eff_dry = self._dry_run if dry_run is None else bool(dry_run)
            market_order_id = None
            tp_order_id = None

            # Build and send market entry first (single leg)
            entry_leg = Leg(
                instrument_type=inst_type,
                symbol=leg_symbol,
                action=entry_action,
                quantity=quantity,
            )
            entry_order = NewOrder(
                time_in_force=OrderTimeInForce.DAY,
                order_type=OrderType.MARKET,
                legs=[entry_leg],
            )

            LOGGER.info("Sending market entry (dry-run=%s): %s", eff_dry, entry_order)
            try:
                order_json = entry_order.model_dump_json(
                    exclude_none=True, by_alias=True
                )
                LOGGER.debug("Market entry JSON: %s", order_json)
                placed_entry = account.place_order(
                    session, entry_order, dry_run=eff_dry
                )
                market_order_id = self._extract_order_id(placed_entry)
                if market_order_id is None:
                    LOGGER.warning(
                        "Market entry returned no ID (dry_run=%s): %s",
                        eff_dry,
                        repr(placed_entry),
                    )
                    return "Market entry submitted but no ID returned (check logs)."
            except Exception as exc:
                msg = str(exc)
                LOGGER.warning("Market entry failed: %s", msg)
                self._raise_on_auth_error(exc)
                raise

            # Build TP as a separate limit order (legging in to satisfy API rules)
            # Sign the price so the API infers price_effect correctly
            price_sign = (
                -1
                if tp_action
                in {OrderAction.BUY, OrderAction.BUY_TO_CLOSE, OrderAction.BUY_TO_OPEN}
                else 1
            )
            tp_price_signed = Decimal(str(tp_price)) * Decimal(price_sign)
            tp_leg = Leg(
                instrument_type=inst_type,
                symbol=leg_symbol,
                action=tp_action,
                quantity=quantity,
            )
            tp_order = NewOrder(
                time_in_force=OrderTimeInForce.GTC,
                order_type=OrderType.LIMIT,
                price=tp_price_signed,
                legs=[tp_leg],
            )

            LOGGER.info(
                "Sending TP limit (dry-run=%s): %s @ %s", eff_dry, tp_order, tp_price
            )
            try:
                # Attach bracket/OTO reference for TP leg; use raw POST to include bracket-order-id
                tp_payload = tp_order.model_dump(exclude_none=True, by_alias=True)
                tp_payload["bracket-order-id"] = market_order_id
                tp_json = json.dumps(tp_payload, default=str)
                tp_url = f"/accounts/{account.account_number}/orders"
                if eff_dry:
                    tp_url += "/dry-run"
                placed_tp = session._post(tp_url, data=tp_json)
                tp_order_id = self._extract_order_id(placed_tp)
                if tp_order_id is None:
                    LOGGER.warning(
                        "TP order returned no ID (dry_run=%s): %s",
                        eff_dry,
                        repr(placed_tp),
                    )
            except Exception as exc:
                msg = str(exc)
                LOGGER.warning(
                    "TP order failed (entry_id=%s): %s", market_order_id, msg
                )
                self._raise_on_auth_error(exc)
                # Entry already placed; surface TP failure but keep entry ID
                return (
                    f"[{'DRY-RUN' if eff_dry else 'LIVE'}] Placed {action.lower()} entry {quantity} {symbol} @ market "
                    f"(ID: {market_order_id}) but TP failed: {msg}"
                )

            return (
                f"[{'DRY-RUN' if eff_dry else 'LIVE'}] Placed {action.lower()} entry {quantity} {symbol} @ market "
                f"(ID: {market_order_id}) + TP limit @ {tp_price:.2f} (ID: {tp_order_id or 'n/a'})"
            )

    def _normalize_symbol(self, symbol: str) -> str:
        symbol = symbol.upper().strip()
        futures = {"NQ", "MNQ", "ES", "MES", "RTY", "YM"}
        # if already a futures contract like /NQZ5 or NQZ5, ensure leading '/'
        if symbol.startswith("/"):
            return symbol
        # check product codes (e.g., 'NQ' -> resolve front-month)
        if symbol in futures:
            resolved = self._resolve_front_month_symbol(symbol)
            if resolved:
                return resolved
            # fallback: return a reasonable default based on current date
            now = datetime.now(timezone.utc)
            month = (now.month % 12) + 1
            month_codes = "FGHJKMNQUVXZ"
            month_code = month_codes[month - 1]
            year = now.year if month != 1 else now.year + 1
            year_digit = str(year)[-2:]
            return f"/{symbol}{month_code}{year_digit}"
        # check for explicit contract codes like NQZ25 or NQH26 (with optional exchange suffix)
        # Accept patterns like 'NQZ25', 'NQZ25:XCME', '/NQZ25', '/NQZ25:XCME'
        m = re.match(
            r"^/?(?P<prod>NQ|MNQ|ES|MES|RTY|YM)(?P<month>[A-Z])(?P<year>\d{1,2})(?::(?P<exch>\w+))?$",
            symbol,
        )
        if m:
            prod = m.group("prod")
            month = m.group("month")
            year = m.group("year")
            # Preserve short year (e.g., '25') to construct contract symbol
            return f"/{prod}{month}{year}"
        return symbol

    def _get_current_price(self, session, symbol: str) -> float:
        # Get quote
        # Consider both forms: leading '/' or numeric-month contract like 'NQZ25'
        is_future = False
        if isinstance(symbol, str) and symbol.startswith("/"):
            is_future = True
        else:
            # detect explicit future contract like NQZ25
            if isinstance(symbol, str) and re.match(
                r"^(NQ|MNQ|ES|MES|RTY|YM)[A-Z]\d{1,2}(:\w+)?$", symbol.upper()
            ):
                is_future = True

        if is_future:
            # Future: API sometimes expects symbol without exchange suffix or leading '/'.
            # Try several common variants to avoid 404 responses.
            variants = []
            raw = symbol.lstrip("/")
            variants.append("/" + raw)
            variants.append(raw)
            # If symbol contains exchange suffix (like :XCME), strip it
            if ":" in raw:
                base = raw.split(":", 1)[0]
                variants.append("/" + base)
                variants.append(base)
            quote_data = None
            last_exc = None
            for v in variants:
                try:
                    resp = session._get(f"/quotes/futures?symbol={v}")
                    quote_data = resp["data"]["items"]
                    LOGGER.debug("Quote fetch succeeded for variant: %s", v)
                    break
                except Exception as exc:
                    last_exc = exc
                    LOGGER.debug("Quote fetch failed for variant %s: %s", v, exc)
            if quote_data is None and last_exc is not None:
                # Re-raise last exception to bubble up
                raise last_exc
        else:
            # Equity
            quote_data = session._get(f"/quotes/equities?symbol={symbol}")["data"][
                "items"
            ]
        if quote_data:
            bid = quote_data[0].get("bid-price")
            ask = quote_data[0].get("ask-price")
            if bid is not None and ask is not None:
                return (bid + ask) / 2
            # Fall back to last/mark if bid/ask missing
            for key in ("last-price", "mark-price", "close-price", "last", "mark"):
                val = quote_data[0].get(key)
                if val is not None:
                    return float(val)
        raise RuntimeError(f"No quote data available for symbol: {symbol}")

    def _raise_on_auth_error(self, exc: Exception) -> None:
        """Raise a standardized auth error when the refresh token is invalid."""
        msg = str(exc).lower() if exc else ""
        if any(key in msg for key in ("invalid_grant", "grant revoked", "invalid_token")):
            self._mark_needs_reauth()
            raise TastytradeAuthError(AUTH_ERROR_TEXT) from exc
