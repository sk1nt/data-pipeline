from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from ..services.automated_options_service import AutomatedOptionsService

router = APIRouter()


class OptionsOrderRequest(BaseModel):
    message: str
    channel_id: str
    user_id: str


class OptionsOrderResponse(BaseModel):
    message: str
    order_id: Optional[str]


@router.post("/options/orders", response_model=OptionsOrderResponse)
async def process_options_alert(request: OptionsOrderRequest):
    """Process an options alert and place automated order."""

    service = AutomatedOptionsService()
    try:
        order_id = await service.process_alert(
            request.message, request.channel_id, request.user_id
        )
        if order_id:
            return OptionsOrderResponse(
                message="Options order placed", order_id=order_id
            )
        else:
            return OptionsOrderResponse(message="No valid alert found", order_id=None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert processing failed: {e}")
