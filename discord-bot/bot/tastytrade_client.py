"""Helper for interacting with TastyTrade accounts via OAuthSession."""

from __future__ import annotations

import threading
from pathlib import Path
import sys

import httpx
import re
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
from tastytrade.utils import TastytradeError

try:  # pragma: no cover - optional dependency
    from tastytrade.session import Session
    from tastytrade.account import Account, AccountBalance
except ImportError as exc:  # pragma: no cover - optional dependency
    print(f"ImportError in tastytrade_client: {exc}")
    Session = None  # type: ignore
    Account = None  # type: ignore
    AccountBalance = None  # type: ignore

try:
    from services.tastytrade_auth_service import (
        TastytradeAuthError as SharedTastytradeAuthError,
        TastytradeAuthService,
        TastytradeAuthSettings,
        TastytradeTransientAuthError,
    )
except ImportError:  # pragma: no cover - backend imports may not have src/ on path yet
    src_path = Path(__file__).resolve().parents[2] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from services.tastytrade_auth_service import (
        TastytradeAuthError as SharedTastytradeAuthError,
        TastytradeAuthService,
        TastytradeAuthSettings,
        TastytradeTransientAuthError,
    )


# Shared pure helper for atomic OTOCO/OTO bracket complex orders (no heavy deps).
try:
    from services.complex_order_builder import build_bracket_complex_order
except ImportError:  # pragma: no cover - src/ on path is ensured by the auth import above
    src_path = Path(__file__).resolve().parents[2] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from services.complex_order_builder import build_bracket_complex_order

@dataclass
class AccountSummary:
    account_number: str
    nickname: Optional[str]
    account_type: str
    buying_power: float
    net_liq: float
    cash_balance: float


LOGGER = logging.getLogger(__name__)


def round_to_tick(price: float, tick: float, direction: str) -> float:
    """Round price to tick increments in given direction ('up'|'down'|'')"""
    import math

    if tick <= 0:
        return price
    try:
        if math.isnan(price):
            return price
    except Exception:
        pass
    n = price / tick
    if direction == "down":
        n2 = math.floor(n - 1e-9)
    elif direction == "up":
        n2 = math.ceil(n + 1e-9)
    else:
        n2 = round(n)
    return n2 * tick
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
        self._last_connect_failure: float = 0.0
        self._connect_cooldown_seconds: float = 15.0
        self._auth_service = TastytradeAuthService(
            TastytradeAuthSettings(
                client_secret=client_secret,
                refresh_token=refresh_token,
                use_sandbox=use_sandbox,
            )
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
            # Use model_dump() for proper Pydantic serialization
            return [pos.model_dump() if hasattr(pos, 'model_dump') else pos.__dict__ for pos in positions]

    def get_positions_raw(self) -> list:
        """Get positions via direct REST API call (fallback if SDK fails)."""
        with self._lock:
            session = self._ensure_session()
            account = self._ensure_active_account(session)
            try:
                resp = session._get(f"/accounts/{account.account_number}/positions")
                items = resp.get("data", {}).get("items", [])
                return items
            except Exception as exc:
                LOGGER.warning("get_positions_raw failed: %s", exc)
                raise

    def flatten_position(
        self,
        symbol: str,
        dry_run: Optional[bool] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> dict:
        """Flatten (close) an entire position for a symbol.
        
        Args:
            symbol: Product symbol like 'MNQ' or '/MNQ'
            dry_run: If True, validate order but don't submit
            max_retries: Number of retries if position not found (API propagation delay)
            retry_delay: Seconds to wait between retries
            
        Returns:
            Dict with cancelled_orders, position details, and order result
        """
        with self._lock:
            session = self._ensure_session()
            account = self._ensure_active_account(session)
            
            # Normalize symbol (strip leading /)
            search_symbol = symbol.upper().lstrip("/")
            
            # Step 1: Get all positions via SDK (same method as get_positions)
            pos = None
            positions = []
            for attempt in range(max_retries + 1):
                try:
                    sdk_positions = account.get_positions(session)
                    # Convert to dicts for consistent handling
                    positions = [p.model_dump() if hasattr(p, 'model_dump') else p.__dict__ for p in sdk_positions]
                    LOGGER.debug("flatten_position: attempt %d, SDK positions: %s", attempt + 1, positions)
                except Exception as exc:
                    LOGGER.error("Failed to get positions: %s", exc)
                    raise RuntimeError(f"Failed to get positions: {exc}")
                
                # Find matching position - SDK uses underscored keys
                for p in positions:
                    p_symbol = (p.get("symbol") or "").upper().lstrip("/")
                    p_underlying = (p.get("underlying_symbol") or "").upper().lstrip("/")
                    LOGGER.debug("flatten_position: checking position symbol=%s underlying=%s against search=%s", 
                               p_symbol, p_underlying, search_symbol)
                    if search_symbol in p_symbol or search_symbol in p_underlying:
                        pos = p
                        break
                
                if pos:
                    break
                
                # Position not found - retry after delay (API propagation can take a few seconds)
                if attempt < max_retries:
                    LOGGER.info("flatten_position: position not found for %s, retrying in %.1fs (attempt %d/%d)", 
                              search_symbol, retry_delay, attempt + 1, max_retries)
                    time.sleep(retry_delay)
            
            if not pos:
                # List available positions for debugging
                available_positions = [p.get("symbol", "?") for p in positions]
                LOGGER.warning("flatten_position: no position found for %s in %d positions: %s", 
                             search_symbol, len(positions), available_positions)
                msg = f"No position found for {symbol}"
                if available_positions:
                    msg += f" (available: {', '.join(available_positions)})"
                return {
                    "status": "no_position",
                    "message": msg,
                    "cancelled_orders": 0,
                }
            
            # Get position details (SDK uses underscored keys)
            pos_symbol = pos.get("symbol", "")  # Full contract like /MNQH6
            pos_qty = int(float(pos.get("quantity", 0) or 0))
            # SDK may use quantity_direction or direction - check both
            pos_direction = pos.get("quantity_direction") or pos.get("direction") or ""
            # Handle enum values (may be PositionDirection.LONG instead of "Long")
            if hasattr(pos_direction, 'value'):
                pos_direction = pos_direction.value
            pos_direction = str(pos_direction)
            LOGGER.debug("flatten_position: found position symbol=%s qty=%d direction='%s' raw_pos=%s", 
                       pos_symbol, pos_qty, pos_direction, pos)
            
            if pos_qty == 0:
                return {
                    "status": "no_position",
                    "message": f"Position quantity is zero for {symbol}",
                    "cancelled_orders": 0,
                }
            
            # Step 2: Cancel all working orders for this product
            cancelled_orders = []
            try:
                live_orders = account.get_live_orders(session)
                for order in live_orders:
                    order_status = getattr(order, 'status', None)
                    status_str = str(order_status.value).lower() if order_status else ''
                    if status_str in ('filled', 'cancelled', 'rejected', 'expired'):
                        continue
                    
                    # Check if order is for our symbol
                    order_underlying = getattr(order, 'underlying_symbol', '') or ''
                    legs = getattr(order, 'legs', [])
                    order_symbol = ''
                    if legs:
                        order_symbol = getattr(legs[0], 'symbol', '') or ''
                    
                    if (search_symbol in order_underlying.upper() or 
                        search_symbol in order_symbol.upper()):
                        order_id = getattr(order, 'id', None)
                        if order_id:
                            try:
                                session._delete(
                                    f"/accounts/{account.account_number}/orders/{order_id}"
                                )
                                cancelled_orders.append(order_id)
                                LOGGER.info("Cancelled order %s for flatten", order_id)
                            except Exception as cancel_exc:
                                LOGGER.warning("Failed to cancel order %s: %s", order_id, cancel_exc)
            except Exception as exc:
                LOGGER.warning("Error cancelling orders during flatten: %s", exc)
            
            # Wait for order cancellations to propagate through the risk system
            if cancelled_orders:
                LOGGER.info("Waiting for %d cancelled orders to clear...", len(cancelled_orders))
                time.sleep(0.5)
            
            # Step 3: Place market order to close position
            # SDK returns quantity as always positive with quantity_direction indicating Long/Short
            # If Long, we SELL to close; if Short, we BUY to close
            direction_lower = pos_direction.lower()
            if direction_lower == "long":
                is_long = True
            elif direction_lower == "short":
                is_long = False
            else:
                LOGGER.warning("flatten_position: unknown direction '%s', assuming short (BUY_TO_CLOSE)", pos_direction)
                is_long = False
            LOGGER.info("flatten_position: is_long=%s (direction='%s'), will %s %d", 
                       is_long, pos_direction, "SELL" if is_long else "BUY", pos_qty)
            if is_long:
                close_action = OrderAction.SELL_TO_CLOSE
            else:
                close_action = OrderAction.BUY_TO_CLOSE
            
            close_qty = abs(pos_qty)
            
            # Build market order
            leg = Leg(
                instrument_type=InstrumentType.FUTURE,
                symbol=pos_symbol,
                action=close_action,
                quantity=close_qty,
            )
            
            order = NewOrder(
                time_in_force=OrderTimeInForce.DAY,
                order_type=OrderType.MARKET,
                legs=[leg],
            )
            
            # Determine dry_run
            if dry_run is None:
                dry_run = self._dry_run
            
            # Try placing the flatten order with retries (margin system may need time to update)
            last_error = None
            for order_attempt in range(3):
                try:
                    result = account.place_order(session, order, dry_run=dry_run)
                    order_id = getattr(result, 'order', None)
                    if order_id:
                        order_id = getattr(order_id, 'id', None)
                    
                    return {
                        "status": "success",
                        "position_symbol": pos_symbol,
                        "position_qty": pos_qty,
                        "position_direction": pos_direction,
                        "close_action": str(close_action),
                        "close_qty": close_qty,
                        "cancelled_orders": len(cancelled_orders),
                        "cancelled_order_ids": cancelled_orders,
                        "order_id": order_id,
                        "dry_run": dry_run,
                        "message": f"{'[DRY RUN] ' if dry_run else ''}Flattened {pos_symbol}: {close_action.name} {close_qty}",
                    }
                except Exception as exc:
                    last_error = exc
                    error_str = str(exc).lower()
                    # Retry on margin/concentration issues (may resolve after order cancellations propagate)
                    if 'margin' in error_str or 'concentration' in error_str or 'buying_power' in error_str:
                        LOGGER.warning("Flatten order attempt %d failed (margin): %s, retrying...", 
                                     order_attempt + 1, exc)
                        time.sleep(1.0)
                        continue
                    # Non-margin error, don't retry
                    LOGGER.error("Failed to place flatten order: %s", exc)
                    self._raise_on_auth_error(exc)
                    raise RuntimeError(f"Failed to place flatten order: {exc}")
            
            # All retries exhausted
            LOGGER.error("Failed to place flatten order after 3 attempts: %s", last_error)
            self._raise_on_auth_error(last_error)
            raise RuntimeError(f"Failed to place flatten order: {last_error}")

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
            # Use model_dump() for proper Pydantic serialization
            return [order.model_dump() if hasattr(order, 'model_dump') else order.__dict__ for order in orders]

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
        try:
            session = self._auth_service.get_session()
            self._session = session
            self._session_expiration = self._auth_service.session_expiration
            self._needs_reauth = False
            self._last_connect_failure = 0.0
            return session
        except SharedTastytradeAuthError as exc:
            self._mark_needs_reauth()
            raise TastytradeAuthError(str(exc)) from exc
        except TastytradeTransientAuthError as exc:
            raise ConnectionError(
                f"TastyTrade authorization temporarily unavailable: {exc}"
            ) from exc
        except (httpx.TransportError, ConnectionError, TimeoutError, OSError) as exc:
            self._last_connect_failure = time.monotonic()
            raise ConnectionError(f"TastyTrade API unreachable: {exc}") from exc

    def ensure_authorized(self) -> bool:
        """Ensure that the session is authorized/valid. Returns True if valid.

        Will attempt to refresh or reinitialize the session when needed. If the
        refresh token is invalid or revoked, raises TastytradeAuthError.
        """
        with self._lock:
            try:
                self._ensure_session()
                return True
            except TastytradeAuthError:
                raise

    def set_refresh_token(self, refresh_token: str) -> None:
        """Replace stored refresh token and reinitialize session with new token."""
        refresh_token = (refresh_token or "").strip()
        if not refresh_token:
            raise ValueError("refresh_token must be provided")
        with self._lock:
            self._refresh_token = refresh_token
            try:
                self._auth_service.set_refresh_token(refresh_token)
                self._session = self._auth_service.get_session()
                self._session_expiration = self._auth_service.session_expiration
                self._needs_reauth = False
            except SharedTastytradeAuthError as exc:
                self._session = None
                self._session_expiration = None
                self._needs_reauth = True
                raise TastytradeAuthError(str(exc)) from exc
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
            self._auth_service.set_use_sandbox(self._use_sandbox)
            try:
                self._session = self._auth_service.get_session()
                self._session_expiration = self._auth_service.session_expiration
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
        LOGGER.debug("TastyTrade reauth worker disabled; shared auth service owns refresh")

    def _reauth_worker(self) -> None:
        return

    # Number of days before expiration to roll to the next quarterly contract.
    ROLL_BUFFER_DAYS = 7

    def _resolve_front_month_symbol(self, product_code: str) -> Optional[str]:
        """Return front-month TW symbol (like /NQM6) for a product code like 'NQ'.

        Automatically rolls to the next quarterly contract when the nearest
        expiration is within ``ROLL_BUFFER_DAYS`` days.  Caches to avoid
        repeated API calls (1-hour TTL).
        """
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
            # Select the nearest expiration that is NOT within the roll buffer.
            # This ensures we automatically roll to the next contract when the
            # current one is about to expire.
            selected: Optional[Future] = None
            if candidates:
                candidates.sort(
                    key=lambda f: getattr(f, "expiration_date", datetime.max)
                )
                roll_cutoff = (now + timedelta(days=self.ROLL_BUFFER_DAYS)).date()
                for c in candidates:
                    exp = getattr(c, "expiration_date", None)
                    if exp is None:
                        continue
                    # expiration_date may be a datetime; normalise to date
                    if hasattr(exp, "date"):
                        exp = exp.date()
                    if exp >= roll_cutoff:
                        selected = c
                        break
                # Fallback: if every candidate expires within the buffer, use the
                # last (furthest-out) one rather than trading an expiring contract.
                if selected is None and candidates:
                    selected = candidates[-1]
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
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                accounts = Account.get(session)
                break
            except TastytradeError as e:
                if "empty_response" in str(e) and attempt < max_retries:
                    LOGGER.warning(
                        "TastyTrade empty_response on Account.get, retry %d/%d",
                        attempt,
                        max_retries,
                    )
                    time.sleep(1)
                else:
                    raise
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
          - PlacedOrderResponse object with `order` attribute containing PlacedOrder with `id`
          - PlacedOrder object with attribute `id` (int or str)
          - Raw dict from REST with 'data' -> 'id' or top-level 'id'
        """
        if placed is None:
            return None
        # If PlacedOrderResponse with nested 'order' attribute
        try:
            order_obj = getattr(placed, "order", None)
            if order_obj is not None:
                oid = getattr(order_obj, "id", None)
                if oid is not None and oid != -1:
                    return str(oid)
        except Exception:
            pass
        # If object with attribute 'id' directly
        try:
            oid = getattr(placed, "id", None)
            if oid is not None and oid != -1:
                return str(oid)
        except Exception:
            pass
        # If dict-like response
        try:
            if isinstance(placed, dict):
                # common shapes: {'data': {'order': {'id': 123}}}, {'data': {'id': 123}} or {'id': 123}
                data = placed.get("data") if placed.get("data") is not None else placed
                if isinstance(data, dict):
                    # Check for nested order object
                    if "order" in data and isinstance(data["order"], dict) and "id" in data["order"]:
                        return str(data["order"]["id"])
                    if "id" in data:
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
        sl_ticks: float = 0.0,
        tick_size: float = 0.25,
        dry_run: Optional[bool] = None,
        market_price: Optional[float] = None,
    ) -> str:
        """Place market entry order with TP exit (and optional SL via OTOCO bracket).

        Submits an atomic complex order so the entry and exit(s) cannot desync:
        - sl_ticks > 0 -> OTOCO bracket (entry + TP limit + SL stop)
        - sl_ticks = 0 -> OTO bracket   (entry + TP limit only)

        The TP level is derived from the pre-trade reference price; the atomic
        bracket avoids the former cancel/poll/restore dance (no 5s fill poll,
        no restore failure, no buy/sell race) used for the TP-only case.
        """
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

            # Default tick sizes - will be refined from SDK if possible
            if inst_type == InstrumentType.FUTURE:
                tick_size = 0.25  # MNQ/NQ default tick size
            else:
                tick_size = 0.01  # Equities

            # For futures: resolve symbol, tick_size, and price in ONE SDK call
            _future_obj = None  # cache the Future object to avoid repeated API calls
            if inst_type == InstrumentType.FUTURE:
                try:
                    resolved = self._resolve_future_contract_symbol(session, symbol)
                    if resolved:
                        symbol = resolved
                except Exception:
                    pass

                # Single Future.get() call to get tick_size + price data
                try:
                    sym_lookup = symbol.lstrip("/")
                    futs = Future.get(session, symbols=[sym_lookup])
                    if futs:
                        _future_obj = futs[0]
                        # Extract tick_size
                        tick_meta = getattr(_future_obj, "tick_size", None)
                        min_tick = getattr(_future_obj, "min_tick", None)
                        if isinstance(tick_meta, (int, float)) and 0 < tick_meta < 1:
                            tick_size = float(tick_meta)
                        elif isinstance(min_tick, (int, float)) and 0 < min_tick < 1:
                            tick_size = float(min_tick)
                        LOGGER.debug("Future %s: tick_size=%s", symbol, tick_size)
                except Exception as ex:
                    LOGGER.debug("SDK Future lookup exception: %s", ex)

            # Get current price — use cached Future object (free, no extra API call).
            # For futures, TastyTrade doesn't serve live quotes via REST — the fill
            # price from the market order is used to calculate TP (Step 4), which is
            # more accurate anyway.  Only call _get_current_price for equities.
            current_price = None
            if market_price is not None:
                current_price = float(market_price)
            else:
                # Try price attrs on the Future object we already have (no network call)
                if _future_obj is not None:
                    for attr in ('mark_price', 'last_price', 'close_price', 'settlement_price'):
                        val = getattr(_future_obj, attr, None)
                        if val is not None and float(val) > 0:
                            current_price = float(val)
                            LOGGER.debug("Price from cached Future.%s: %s", attr, current_price)
                            break
                # For non-futures, try the full quote lookup
                if current_price is None and inst_type != InstrumentType.FUTURE:
                    try:
                        current_price = self._get_current_price(session, symbol)
                    except Exception as exc:
                        LOGGER.warning("Could not fetch current price for %s: %s — will use fill price for TP", symbol, exc)
                elif current_price is None:
                    LOGGER.debug("No pre-trade quote for %s — TP will be calculated from fill price", symbol)

            # Determine entry action
            # For futures on TastyTrade:
            # - Entries: BUY_TO_OPEN (BTO) / SELL_TO_OPEN (STO)
            # - Exits (TP): SELL_TO_CLOSE (STC) / BUY_TO_CLOSE (BTC)
            if inst_type == InstrumentType.FUTURE:
                entry_action = (
                    OrderAction.BUY_TO_OPEN
                    if action.upper() == "BUY"
                    else OrderAction.SELL_TO_OPEN
                )
                # TP is opposite side - use _TO_CLOSE to properly close the position
                tp_action = (
                    OrderAction.SELL_TO_CLOSE
                    if action.upper() == "BUY"
                    else OrderAction.BUY_TO_CLOSE
                )
            else:
                entry_action = (
                    OrderAction.BUY_TO_OPEN if action.upper() == "BUY" else OrderAction.SELL_TO_OPEN
                )
                # For equities, use _TO_CLOSE for exits
                tp_action = (
                    OrderAction.SELL_TO_CLOSE if action.upper() == "BUY" else OrderAction.BUY_TO_CLOSE
                )

            # Calculate TP price directionally (tick_size already resolved from SDK above)
            def _round_to_tick(price: float, tick: float, direction: str) -> float:
                return round_to_tick(price, tick, direction)

            tp_price = None
            if current_price is not None:
                LOGGER.debug("Calculating TP: current_price=%s, tp_ticks=%s, tick_size=%s", 
                            current_price, tp_ticks, tick_size)
                if action.upper() == "BUY":  # Long: TP above
                    tp_price = current_price + (tp_ticks * tick_size)
                    round_dir = "up"  # Ceiling to ensure profit
                else:  # Short: TP below
                    tp_price = current_price - (tp_ticks * tick_size)
                    round_dir = "down"  # Floor to ensure profit
                
                LOGGER.debug("TP price before rounding: %s", tp_price)
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

            # Calculate SL price if sl_ticks > 0
            sl_price = None
            if sl_ticks > 0 and current_price is not None:
                if action.upper() == "BUY":  # Long: SL below
                    sl_price = current_price - (sl_ticks * tick_size)
                    sl_price = _round_to_tick(sl_price, tick_size, "down")
                else:  # Short: SL above
                    sl_price = current_price + (sl_ticks * tick_size)
                    sl_price = _round_to_tick(sl_price, tick_size, "up")
                LOGGER.debug("SL price: %s (sl_ticks=%s)", sl_price, sl_ticks)

            # Build entry order
            entry_leg = Leg(
                instrument_type=inst_type,
                symbol=leg_symbol,
                action=entry_action,
                quantity=quantity,
            )
            
            # Build closing leg for TP/SL
            closing_leg = Leg(
                instrument_type=inst_type,
                symbol=leg_symbol,
                action=tp_action,
                quantity=quantity,
            )

            # Atomic bracket order via the shared helper:
            #   sl_ticks > 0 -> OTOCO (entry triggers TP-limit OR SL-stop)
            #   sl_ticks = 0 -> OTO    (entry triggers TP-limit only)
            # Replaces the former cancel/restore dance for the TP-only case
            # (no 5s fill poll, no restore failure, no buy/sell race condition).
            if tp_price is None:
                # No reference price to compute a TP - plain market entry, no exit leg.
                entry_order = NewOrder(
                    time_in_force=OrderTimeInForce.DAY,
                    order_type=OrderType.MARKET,
                    legs=[entry_leg],
                )
                LOGGER.info("Sending market entry, no TP price (dry-run=%s)", eff_dry)
                try:
                    resp = account.place_order(session, entry_order, dry_run=eff_dry)
                    order_id = self._extract_order_id(resp)
                    return (
                        f"{'[DRY-RUN] ' if eff_dry else ''}Placed {action.lower()} entry "
                        f"{quantity} {symbol} @ market (ID: {order_id or 'n/a'}) - TP skipped (no price data)"
                    )
                except Exception as exc:
                    msg = str(exc)
                    LOGGER.warning("Market entry failed: %s", msg)
                    self._raise_on_auth_error(exc)
                    return f"Market entry order failed: {msg}"

            sl_price_arg = sl_price if (sl_ticks > 0 and sl_price is not None) else None
            bracket = build_bracket_complex_order(
                entry_leg=entry_leg,
                closing_leg=closing_leg,
                tp_price=tp_price,
                tp_action=tp_action,
                sl_price=sl_price_arg,
            )
            order_kind = "OTOCO" if sl_price_arg is not None else "OTO"
            LOGGER.info(
                "Sending %s bracket (dry-run=%s): trigger=market %s, TP=%s%s",
                order_kind,
                eff_dry,
                action,
                tp_price,
                f", SL={sl_price}" if sl_price_arg is not None else "",
            )
            try:
                resp = account.place_complex_order(session, bracket, dry_run=eff_dry)
                order_id = (
                    getattr(resp.complex_order, "id", None) if resp.complex_order else None
                )
                LOGGER.info("%s bracket placed, complex_order_id=%s", order_kind, order_id)
                if sl_price_arg is not None:
                    return (
                        f"{'[DRY-RUN] ' if eff_dry else ''}OTOCO bracket placed: "
                        f"Market {action} {quantity} {symbol} + TP @ {tp_price:.2f} + SL @ {sl_price:.2f} "
                        f"(ID: {order_id or 'n/a'})"
                    )
                return (
                    f"{'[DRY-RUN] ' if eff_dry else ''}OTO placed: "
                    f"Market {action} {quantity} {symbol} + TP @ {tp_price:.2f} "
                    f"(ID: {order_id or 'n/a'})"
                )
            except Exception as exc:
                msg = str(exc)
                LOGGER.warning("%s bracket failed: %s", order_kind, msg)
                self._raise_on_auth_error(exc)
                return f"{order_kind} order failed: {msg}"

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
            # Futures use quarterly months: H(Mar), M(Jun), U(Sep), Z(Dec)
            now = datetime.now(timezone.utc)
            quarterly = [
                (3, "H"), (6, "M"), (9, "U"), (12, "Z"),
            ]
            # Pick the next quarterly month whose expiration hasn't passed yet,
            # accounting for the roll buffer.
            roll_month = now.month
            roll_year = now.year
            month_code = None
            for q_month, q_code in quarterly:
                if q_month >= roll_month:
                    month_code = q_code
                    break
            if month_code is None:
                # Past December — next is March of next year
                month_code = "H"
                roll_year += 1
            year_digit = str(roll_year)[-2:]
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
        """Fetch current price — fast methods first, streamer as last resort."""
        sym_lookup = symbol.lstrip("/")

        # Method 1 (fast): Get price attributes from the Future instrument via REST
        try:
            futs = Future.get(session, symbols=[sym_lookup])
            if futs:
                f = futs[0]
                for attr in ('mark_price', 'last_price', 'close_price', 'settlement_price'):
                    val = getattr(f, attr, None)
                    if val is not None and float(val) > 0:
                        LOGGER.debug("Got price from Future.%s for %s: %s", attr, symbol, val)
                        return float(val)
        except Exception as exc:
            LOGGER.debug("Future instrument price lookup failed for %s: %s", symbol, exc)

        # Method 2 (fast): Legacy REST endpoint fallback
        is_future = isinstance(symbol, str) and (
            symbol.startswith("/") or
            bool(re.match(r"^(NQ|MNQ|ES|MES|RTY|YM)[A-Z]\d{1,2}(:\w+)?$", symbol.upper()))
        )
        endpoints = ["/market-data", "/quotes/futures", "/quotes"] if is_future else ["/quotes/equities"]
        raw = symbol.lstrip("/")
        variants = [f"/{raw}", raw]
        if ":" in raw:
            base = raw.split(":", 1)[0]
            variants.extend([f"/{base}", base])

        for endpoint in endpoints:
            for v in variants:
                try:
                    resp = session._get(f"{endpoint}?symbol={v}")
                    items = resp.get("data", {}).get("items", [])
                    if items:
                        bid = items[0].get("bid-price")
                        ask = items[0].get("ask-price")
                        if bid is not None and ask is not None:
                            return (float(bid) + float(ask)) / 2.0
                        for key in ("last-price", "mark-price", "close-price"):
                            val = items[0].get(key)
                            if val is not None:
                                return float(val)
                except Exception:
                    continue

        # Method 3 (slow, ~1-2s): DXLinkStreamer one-shot quote — only if fast methods failed
        try:
            from tastytrade import DXLinkStreamer
            from tastytrade.dxfeed import Quote as DXQuote
            import asyncio

            async def _fetch_quote():
                async with DXLinkStreamer(session) as streamer:
                    subs = [sym_lookup]
                    try:
                        futs = Future.get(session, symbols=[sym_lookup])
                        if futs and getattr(futs[0], 'streamer_symbol', None):
                            subs = [futs[0].streamer_symbol]
                    except Exception:
                        pass
                    await streamer.subscribe(DXQuote, subs)
                    quote = await asyncio.wait_for(streamer.get_event(DXQuote), timeout=2.0)
                    bid = getattr(quote, 'bid_price', None)
                    ask = getattr(quote, 'ask_price', None)
                    if bid and ask and bid > 0 and ask > 0:
                        return (float(bid) + float(ask)) / 2.0
                    raise RuntimeError("No valid bid/ask from streamer")

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    price = pool.submit(lambda: asyncio.run(_fetch_quote())).result(timeout=4)
            else:
                price = asyncio.run(_fetch_quote())
            LOGGER.debug("Got price from DXLink streamer for %s: %s", symbol, price)
            return price
        except Exception as exc:
            LOGGER.debug("DXLink streamer quote failed for %s: %s", symbol, exc)

        raise RuntimeError(f"No quote data available for symbol: {symbol}")

    def _raise_on_auth_error(self, exc: Exception) -> None:
        """Raise a standardized auth error when the refresh token is invalid."""
        msg = str(exc).lower() if exc else ""
        if any(key in msg for key in ("invalid_grant", "grant revoked", "invalid_token")):
            self._mark_needs_reauth()
            raise TastytradeAuthError(AUTH_ERROR_TEXT) from exc
