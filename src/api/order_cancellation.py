from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..services.order_validation import OrderValidationService
from ..services.tastytrade_client import tastytrade_client
from tastytrade import Account
from ..config.settings import config


def _select_account(accounts):
    if not accounts:
        return None
    target = getattr(config, "tastytrade_account", None)
    if target:
        for acc in accounts:
            acc_number = getattr(acc, "account_number", None) or getattr(acc, "number", None)
            if acc_number == target:
                return acc
    return accounts[0]

router = APIRouter()


class OrderCancellationRequest(BaseModel):
    order_id: str
    user_id: str
    reason: str = "User requested cancellation"


class OrderCancellationResponse(BaseModel):
    message: str
    success: bool


@router.delete("/orders/{order_id}", response_model=OrderCancellationResponse)
async def cancel_order(order_id: str, request: OrderCancellationRequest):
    """Cancel an order."""

    # Validate request
    errors = OrderValidationService.validate_order_cancellation(
        order_id, request.user_id
    )
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    # Cancel order via Tastytrade
    try:
        session = tastytrade_client.get_session()
        accounts = Account.get(session)
        account = _select_account(accounts)
        if account:
            account.delete_order(session, order_id)
            return OrderCancellationResponse(
                message="Order cancelled successfully", success=True
            )
        else:
            raise HTTPException(status_code=500, detail="No accounts found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cancellation failed: {e}")
