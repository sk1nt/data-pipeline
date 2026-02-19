"""Quick Order Panel API - FastAPI router for order routing, flattening, and position management."""

import asyncio
import logging
import os
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------
TRADING_API_KEY = os.getenv("TRADING_API_KEY", "")


def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if not TRADING_API_KEY:
        raise HTTPException(503, "TRADING_API_KEY not configured on server")
    if x_api_key != TRADING_API_KEY:
        raise HTTPException(401, "Invalid API key")
    return True


# ---------------------------------------------------------------------------
# TastyTrade client singleton (lazy init)
# ---------------------------------------------------------------------------
_tt_client = None


def _get_tt_client():
    """Lazily initialise a TastyTradeClient from env vars."""
    global _tt_client
    if _tt_client is not None:
        return _tt_client

    # Import here to avoid circular / missing-dep issues at module level
    try:
        import sys, pathlib

        # Add discord-bot to path so we can reuse its client
        bot_path = str(pathlib.Path(__file__).resolve().parents[3] / "discord-bot")
        if bot_path not in sys.path:
            sys.path.insert(0, bot_path)
        from bot.tastytrade_client import TastyTradeClient
    except ImportError as exc:
        raise RuntimeError(f"Cannot import TastyTradeClient: {exc}")

    client_secret = os.getenv("TASTYTRADE_CLIENT_SECRET", "")
    refresh_token = os.getenv("TASTYTRADE_REFRESH_TOKEN", "")
    default_account = os.getenv("TASTYTRADE_DEFAULT_ACCOUNT") or os.getenv("TASTYTRADE_ACCOUNT")
    use_sandbox = os.getenv("TASTYTRADE_USE_SANDBOX", "false").lower() == "true"
    dry_run = os.getenv("TASTYTRADE_DRY_RUN", "true").lower() == "true"

    if not client_secret or not refresh_token:
        raise RuntimeError("TASTYTRADE_CLIENT_SECRET and TASTYTRADE_REFRESH_TOKEN must be set")

    _tt_client = TastyTradeClient(
        client_secret=client_secret,
        refresh_token=refresh_token,
        default_account=default_account,
        use_sandbox=use_sandbox,
        dry_run=dry_run,
    )
    logger.info("TastyTradeClient initialised (sandbox=%s, dry_run=%s)", use_sandbox, dry_run)
    return _tt_client


def get_tt():
    try:
        return _get_tt_client()
    except Exception as exc:
        raise HTTPException(503, f"TastyTrade client unavailable: {exc}")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class OrderRequest(BaseModel):
    symbol: str = Field(..., description="Product code: MNQ, MES, NQ, ES")
    action: str = Field(..., pattern="^(BUY|SELL)$", description="BUY or SELL")
    quantity: int = Field(..., ge=1, le=100)
    tp_ticks: float = Field(default=35.0, ge=0, description="Take-profit in ticks")
    sl_ticks: float = Field(default=0.0, ge=0, description="Stop-loss in ticks (0=none)")
    dry_run: Optional[bool] = Field(None, description="Override global dry_run setting")


class FlattenRequest(BaseModel):
    symbol: str = Field(..., description="Product code to flatten: MNQ, MES, NQ, ES")
    dry_run: Optional[bool] = None


class CancelRequest(BaseModel):
    order_id: str


class SwitchAccountRequest(BaseModel):
    account_number: str = Field(..., description="Account number to switch to")


class SizePreset(BaseModel):
    label: str
    quantity: int


class ConfigResponse(BaseModel):
    symbols: list
    size_presets: list
    default_tp_ticks: float
    default_sl_ticks: float
    dry_run: bool
    sandbox: bool
    accounts: list = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/trading/config", response_model=ConfigResponse)
async def get_config(_=Depends(verify_api_key)):
    """Return panel configuration (symbols, presets, defaults)."""
    tt = get_tt()
    # Fetch accounts list (best-effort)
    try:
        accounts = await asyncio.to_thread(tt.get_accounts)
    except Exception:
        accounts = []
    return ConfigResponse(
        symbols=[
            {"code": "MNQ", "name": "Micro Nasdaq", "tick_size": 0.25, "tick_value": 0.50},
            {"code": "MES", "name": "Micro S&P", "tick_size": 0.25, "tick_value": 1.25},
            {"code": "NQ", "name": "E-mini Nasdaq", "tick_size": 0.25, "tick_value": 5.00},
            {"code": "ES", "name": "E-mini S&P", "tick_size": 0.25, "tick_value": 12.50},
        ],
        size_presets=[
            {"label": "1", "quantity": 1},
            {"label": "2", "quantity": 2},
            {"label": "3", "quantity": 3},
            {"label": "5", "quantity": 5},
            {"label": "10", "quantity": 10},
        ],
        default_tp_ticks=35.0,
        default_sl_ticks=0.0,
        dry_run=tt._dry_run,
        sandbox=tt._use_sandbox,
        accounts=accounts,
    )


@router.get("/trading/account")
async def get_account(_=Depends(verify_api_key)):
    """Account summary: net liq, buying power, positions."""
    tt = get_tt()
    summary = await asyncio.to_thread(tt.get_account_summary)
    return {
        "account_number": summary.account_number,
        "nickname": summary.nickname,
        "account_type": summary.account_type,
        "net_liq": summary.net_liq,
        "buying_power": summary.buying_power,
        "cash_balance": summary.cash_balance,
    }


@router.get("/trading/positions")
async def get_positions(_=Depends(verify_api_key)):
    """Current open positions."""
    tt = get_tt()
    positions = await asyncio.to_thread(tt.get_positions)
    return {"positions": positions}


@router.get("/trading/orders")
async def get_orders(_=Depends(verify_api_key)):
    """Open / working orders."""
    tt = get_tt()
    orders = await asyncio.to_thread(tt.get_orders)
    return {"orders": orders}


@router.post("/trading/order")
async def place_order(req: OrderRequest, _=Depends(verify_api_key)):
    """Place a market order with TP (and optional SL)."""
    tt = get_tt()
    result = await asyncio.to_thread(
        tt.place_market_order_with_tp,
        symbol=req.symbol,
        action=req.action,
        quantity=req.quantity,
        tp_ticks=req.tp_ticks,
        sl_ticks=req.sl_ticks,
        dry_run=req.dry_run,
    )
    return {"result": result}


@router.post("/trading/flatten")
async def flatten(req: FlattenRequest, _=Depends(verify_api_key)):
    """Flatten entire position for a symbol."""
    tt = get_tt()
    result = await asyncio.to_thread(tt.flatten_position, req.symbol, dry_run=req.dry_run)
    return result


@router.post("/trading/cancel")
async def cancel_order(req: CancelRequest, _=Depends(verify_api_key)):
    """Cancel a single order by ID."""
    tt = get_tt()
    result = await asyncio.to_thread(tt.cancel_order, req.order_id)
    return {"result": result}


@router.post("/trading/cancel-all")
async def cancel_all_orders(req: FlattenRequest, _=Depends(verify_api_key)):
    """Cancel all open orders for a symbol."""
    tt = get_tt()
    orders = await asyncio.to_thread(tt.get_orders)
    cancelled = []
    search = req.symbol.upper().lstrip("/")
    for order in orders:
        underlying = (order.get("underlying_symbol") or "").upper().lstrip("/")
        legs = order.get("legs") or []
        leg_sym = ""
        if legs:
            leg_sym = (legs[0].get("symbol") or "").upper().lstrip("/")
        if search in underlying or search in leg_sym:
            status = order.get("status")
            if isinstance(status, dict):
                status = status.get("value", "")
            status = str(status).lower() if status else ""
            if status in ("filled", "cancelled", "rejected", "expired"):
                continue
            oid = str(order.get("id", ""))
            if oid:
                try:
                    await asyncio.to_thread(tt.cancel_order, oid)
                    cancelled.append(oid)
                except Exception as exc:
                    logger.warning("Failed to cancel order %s: %s", oid, exc)
    return {"cancelled": cancelled, "count": len(cancelled)}


@router.get("/trading/accounts")
async def list_accounts(_=Depends(verify_api_key)):
    """List all available TastyTrade accounts."""
    tt = get_tt()
    accounts = await asyncio.to_thread(tt.get_accounts)
    active = tt.active_account
    return {"accounts": accounts, "active_account": active}


@router.post("/trading/switch-account")
async def switch_account(req: SwitchAccountRequest, _=Depends(verify_api_key)):
    """Switch the active trading account."""
    tt = get_tt()
    ok = await asyncio.to_thread(tt.set_active_account, req.account_number)
    if not ok:
        raise HTTPException(400, f"Account {req.account_number} not found")
    summary = await asyncio.to_thread(tt.get_account_summary)
    return {
        "account_number": summary.account_number,
        "nickname": summary.nickname,
        "account_type": summary.account_type,
        "net_liq": summary.net_liq,
        "buying_power": summary.buying_power,
        "cash_balance": summary.cash_balance,
        "message": f"Switched to {summary.account_number}",
    }


@router.post("/trading/dry-run")
async def set_dry_run(enabled: bool, _=Depends(verify_api_key)):
    """Toggle dry-run mode."""
    tt = get_tt()
    tt.set_dry_run(enabled)
    return {"dry_run": tt._dry_run}
