from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from ..services.futures_order_parser import FuturesOrderParser
from ..services.futures_order_service import FuturesOrderService
from ..services.auth_service import AuthService

router = APIRouter()


class FuturesOrderRequest(BaseModel):
    action: str
    symbol: str
    tp_ticks: float
    quantity: int = 1
    mode: str = "dry"
    user_id: str


class FuturesOrderResponse(BaseModel):
    message: str
    order_id: Optional[str]


@router.post("/futures/orders", response_model=FuturesOrderResponse)
async def place_futures_order(request: FuturesOrderRequest):
    """Place a futures order."""

    # Verify user permissions
    if not AuthService.verify_user_for_futures(request.user_id):
        raise HTTPException(
            status_code=403, detail="User not authorized for futures orders"
        )

    # Parse parameters
    parser = FuturesOrderParser()
    try:
        params = parser.parse(
            request.action,
            request.symbol,
            request.tp_ticks,
            request.quantity,
            request.mode,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Place order
    service = FuturesOrderService()
    try:
        result = await service.place_order(params, request.user_id)
        return FuturesOrderResponse(
            message=result, order_id=None
        )  # TODO: extract order_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Order placement failed: {e}")
