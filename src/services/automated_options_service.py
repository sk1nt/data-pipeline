import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

# Import via absolute paths (src/ is assumed to be in sys.path)
from services.alert_parser import AlertParser
from services.options_fill_service import OptionsFillService, InsufficientBuyingPowerError
from services.tastytrade_client import tastytrade_client, TastytradeAuthError
from config.settings import config
from services.notifications import notify_operator

# Setup logger
try:
    from src.lib.trading_logger import get_options_logger, log_trade_event
    logger = get_options_logger()
except ImportError:
    from lib.logging import get_logger
    logger = get_logger(__name__)
    def log_trade_event(*args, **kwargs):
        pass

from lib.redis_client import get_redis_client

try:
    from tastytrade.account import Account  # type: ignore
except Exception:  # pragma: no cover - optional dependency path
    Account = None  # type: ignore


class AutomatedOptionsService:
    def __init__(self, tastytrade_client=tastytrade_client):
        self.alert_parser = AlertParser()
        self.tastytrade_client = tastytrade_client
        self.fill_service = OptionsFillService(tastytrade_client=self.tastytrade_client)

    async def create_entry_order(
        self,
        symbol: str,
        strike: Decimal,
        option_type: str,
        expiry: str,
        quantity: int,
        action: str,
        user_id: str,
        channel_id: str,
        initial_price: Optional[Decimal] = None,
        dry_run: bool = False,
    ) -> Optional[dict]:
        """Create entry order with price discovery and optional dry-run mode.
        
        Args:
            symbol: Underlying symbol (e.g., 'UBER', 'SPX')
            strike: Option strike price
            option_type: 'CALL' or 'PUT'
            expiry: Expiration date (YYYY-MM-DD or contract format)
            quantity: Number of contracts
            action: 'BTO', 'BUY', 'STO', 'SELL'
            user_id: Discord user ID for audit
            channel_id: Discord channel ID for audit
            initial_price: Starting price for limit order (falls back to mid-market)
            dry_run: If True, simulates order without placing (default: False)
            
        Returns:
            Dict with status, order_id, quantity, entry_price, dry_run flag; or error dict
        """
        # Verify authorization before order placement
        try:
            await self.tastytrade_client.ensure_authorized()
        except TastytradeAuthError as exc:
            msg = f"TastyTrade authentication invalid: {exc}. Please update the refresh token."
            logger.error(msg)
            return {
                "status": "error",
                "reason": "authentication_failed",
                "error": msg,
            }

        # Place order using fill service with price discovery
        try:
            result = await self.fill_service.fill_options_order(
                symbol=symbol,
                strike=float(strike),  # fill_options_order expects float
                option_type=option_type,
                expiry=expiry,
                quantity=quantity,
                action=action,
                user_id=user_id,
                channel_id=channel_id,
                initial_price=initial_price,
            )
        except InsufficientBuyingPowerError as exc:
            logger.error("Insufficient buying power: %s", exc)
            await notify_operator(str(exc))
            return {
                "status": "error",
                "reason": "insufficient_buying_power",
                "error": str(exc),
            }
        except TastytradeAuthError as exc:
            logger.error("TastyTrade auth error during order creation: %s", exc)
            return {
                "status": "error",
                "reason": "authentication_failed",
                "error": str(exc),
            }
        except Exception as exc:
            logger.exception("Failed to create entry order: %s", exc)
            return {
                "status": "error",
                "reason": "order_creation_failed",
                "error": str(exc),
            }

        if not result:
            return {
                "status": "error",
                "reason": "fill_service_returned_none",
                "error": "Fill service did not return a result",
            }

        # Extract order details from result (fill_options_order returns {"order_id": ..., "entry_price": ...})
        order_id = result.get("order_id")
        entry_price_str = result.get("entry_price")
        entry_price = Decimal(entry_price_str) if entry_price_str else None

        # Audit successful entry order
        try:
            redis_client = get_redis_client()
            redis_conn = redis_client.client
            audit_key = "audit:automated_alerts" if not dry_run else "audit:automated_alerts_dryrun"
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "channel_id": channel_id,
                "symbol": symbol,
                "strike": float(strike),
                "option_type": option_type,
                "expiry": expiry,
                "action": action,
                "quantity": quantity,
                "entry_price": float(entry_price) if entry_price else None,
                "order_id": order_id,
                "dry_run": dry_run,
                "status": "entry_created",
            }
            redis_conn.lpush(audit_key, json.dumps(payload, default=str))
            logger.info(
                "Entry order %s: order_id=%s qty=%s entry_price=%s",
                "simulated" if dry_run else "placed",
                order_id,
                quantity,
                entry_price,
            )
        except Exception as audit_exc:
            logger.warning("Failed to audit entry order: %s", audit_exc)

        return {
            "status": "success",
            "order_id": order_id,
            "quantity": quantity,
            "entry_price": entry_price,
            "dry_run": dry_run,
        }

    async def process_alert(
        self, message: str, channel_id: str, user_id: str
    ) -> Optional[dict]:
        """Process an alert message and place automated order."""

        # Parse alert
        alert_data = self.alert_parser.parse_alert(message, user_id)
        if not alert_data:
            return None

        # Verify allowed user & channel for automated trades
        from services.auth_service import AuthService

        if not AuthService.verify_user_and_channel_for_automated_trades(user_id, channel_id):
            # Not permitted to trigger automated trades
            logger.warning(
                "User %s or channel %s is not permitted for automated trades",
                user_id,
                channel_id,
            )
            return None

        # Verify auth before attempting to place an order
        try:
            self.tastytrade_client.ensure_authorized()
        except TastytradeAuthError as exc:
            msg = f"TastyTrade authentication invalid: {exc}. Please update the refresh token."
            logger.error(msg)
            raise TastytradeAuthError(msg) from exc

        # Calculate quantity based on allocation and account balances
        quantity, buying_power, est_notional = self._compute_quantity(alert_data, channel_id)

        if buying_power <= 0 or est_notional > buying_power:
            reason = "insufficient_buying_power"
            err_msg = (
                f"Insufficient buying power for order: est_notional={est_notional} buying_power={buying_power}"
            )
            logger.warning(err_msg)
            notify_operator(err_msg)
            self._audit_failure(
                user_id=user_id,
                channel_id=channel_id,
                alert_message=message,
                parsed_alert=alert_data,
                reason=reason,
                error=err_msg,
            )
            return {
                "status": "error",
                "reason": reason,
                "estimated_notional": est_notional,
                "buying_power": buying_power,
            }

        # Place order
        try:
            result = await self.fill_service.fill_options_order(
                symbol=alert_data["symbol"],
                strike=alert_data["strike"],
                option_type=alert_data["option_type"],
                expiry=alert_data["expiry"],
                quantity=int(quantity),
                action=alert_data["action"],
                user_id=user_id,
                channel_id=channel_id,
                initial_price=Decimal(str(alert_data.get("price") or 0.0)),
            )
        except TastytradeAuthError as exc:
            msg = f"TastyTrade authentication invalid while placing order: {exc}. Please update your refresh token."
            logger.error(msg)
            # Audit auth failure
            try:
                redis_client = get_redis_client()
                redis_conn = redis_client.client
                audit_key = "audit:automated_alerts"
                fail_payload = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "user_id": user_id,
                    "channel_id": channel_id,
                    "alert_message": message,
                    "parsed_alert": alert_data,
                    "status": "failed",
                    "reason": "auth",
                    "error": str(exc),
                }
                redis_conn.lpush(audit_key, json.dumps(fail_payload, default=str))
            except Exception:
                pass
            raise TastytradeAuthError(msg) from exc
        except InsufficientBuyingPowerError as exc:
            err_msg = str(exc)
            notify_operator(err_msg)
            self._audit_failure(
                user_id=user_id,
                channel_id=channel_id,
                alert_message=message,
                parsed_alert=alert_data,
                reason="insufficient_buying_power",
                error=err_msg,
            )
            return {
                "status": "error",
                "reason": "insufficient_buying_power",
                "error": err_msg,
            }
        except Exception as exc:
            # Generic failure while placing order — audit and return
            logger.exception("Unhandled error placing order: %s", exc)
            self._audit_failure(
                user_id=user_id,
                channel_id=channel_id,
                alert_message=message,
                parsed_alert=alert_data,
                reason="exception",
                error=str(exc),
            )
            return None

        if not result:
            # Record a failure audit entry if order was not created
            self._audit_failure(
                user_id=user_id,
                channel_id=channel_id,
                alert_message=message,
                parsed_alert=alert_data,
                reason="order_not_created_or_rejected",
            )
            return None

        # Always return structured info: order_id, quantity, entry_price
        order_id = result.get("order_id") if isinstance(result, dict) else result
        entry_price = result.get("entry_price") if isinstance(result, dict) else None

        # Audit log to Redis for compliance and debugging
        try:
            redis_client = get_redis_client()
            redis_conn = redis_client.client
            audit_key = "audit:automated_alerts"
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "channel_id": channel_id,
                "alert_message": message,
                "parsed_alert": alert_data,
                "computed_quantity": int(quantity),
                "entry_price": float(entry_price) if entry_price is not None else None,
                "order_id": order_id,
            }
            redis_conn.lpush(audit_key, json.dumps(payload, default=str))
            logger.info(
                "Audit logged automated alert: order_id=%s qty=%s entry_price=%s",
                order_id,
                int(quantity),
                entry_price,
            )
        except Exception:
            # Non-fatal; we still return success, but log for the operator
            logger.warning("Failed to write audit log to Redis for automated alert")

        # If order was not created (result is falsy), log a failure audit entry for troubleshooting
        if not result:
            self._audit_failure(
                user_id=user_id,
                channel_id=channel_id,
                alert_message=message,
                parsed_alert=alert_data,
                reason="order_not_created_or_rejected",
            )

        return {
            "order_id": order_id,
            "quantity": int(quantity),
            "entry_price": entry_price,
        }

    def _compute_quantity(self, alert_data: dict, channel_id: str) -> tuple[int, float, float]:
        """Compute contract quantity using allocation, price, and account balances."""
        # Get allocation percentage from config (percentage, e.g., 10.0 = 10%)
        alloc_pct = config.tastytrade_allocation_percentage / 100.0
        price = float(alert_data.get("price") or 0.0)
        contract_price = max(price, 0.01) * 100  # options are 100x multiplier

        buying_power = 0.0
        net_liq = 0.0
        try:
            if Account is not None:
                session = self.tastytrade_client.get_session()
                accounts = Account.get(session)
                if accounts:
                    target = getattr(config, "tastytrade_account", None)
                    account = None
                    if target:
                        for acc in accounts:
                            acc_number = getattr(acc, "account_number", None) or getattr(acc, "number", None)
                            if acc_number == target:
                                account = acc
                                break
                    if account is None:
                        account = accounts[0]
                    balances = account.get_balances(session)
                    
                    # Prioritize derivative buying power for options (matches !tt account logic)
                    buying_power = float(
                        balances.derivative_buying_power
                        or balances.equity_buying_power
                        or balances.day_trading_buying_power
                        or balances.net_liquidating_value
                        or 0.0
                    )
                    net_liq = float(balances.net_liquidating_value or buying_power)
        except Exception as e:
            # Log failure for debugging and raise - don't proceed with zero balances
            logger.error("Failed to fetch account balances: %s", e)
            raise

        if buying_power <= 0:
            raise ValueError(f"No buying power available: {buying_power}")

        alloc_dollars = buying_power * alloc_pct
        qty = int(alloc_dollars // contract_price) if contract_price > 0 else 0

        # Adjust for buy alert channels (more conservative sizing)
        if self.alert_parser.is_buy_message(channel_id):
            qty = max(1, int(qty * 0.5))

        # Ensure at least 1 contract
        if qty <= 0:
            qty = 1

        # No max_contracts cap - let allocation percentage control position size
        est_notional = qty * contract_price

        return qty, float(buying_power), float(est_notional)

    def _audit_failure(
        self,
        user_id: str,
        channel_id: str,
        alert_message: str,
        parsed_alert: dict,
        reason: str,
        error: Optional[str] = None,
    ) -> None:
        """Write a failure audit log (best-effort)."""
        try:
            redis_client = get_redis_client()
            redis_conn = redis_client.client
            audit_key = "audit:automated_alerts"
            fail_payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "channel_id": channel_id,
                "alert_message": alert_message,
                "parsed_alert": parsed_alert,
                "status": "failed",
                "reason": reason,
            }
            if error:
                fail_payload["error"] = error
            redis_conn.lpush(audit_key, json.dumps(fail_payload, default=str))
        except Exception:
            pass
