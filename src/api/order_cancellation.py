from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..services.order_validation import OrderValidationService
from ..services.tastytrade_client import tastytrade_client, select_account
from tastytrade import Account

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

    errors = OrderValidationService.validate_order_cancellation(
        order_id, request.user_id
    )
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    try:
        session = tastytrade_client.get_session()
        accounts = Account.get(session)
        account = select_account(accounts)
        if account:
            account.delete_order(session, int(order_id))
            return OrderCancellationResponse(
                message="Order cancelled successfully", success=True
            )
        else:
            raise HTTPException(status_code=500, detail="No accounts found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cancellation failed: {e}")
