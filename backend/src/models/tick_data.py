from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime
from decimal import Decimal

class TickData(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10)
    timestamp: datetime
    price: Decimal = Field(..., gt=0)
    volume: Optional[int] = Field(None, ge=0)
    tick_type: str  # trade, bid, ask
    source: str  # sierra_chart, gexbot, tastyttrade

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Symbol must contain only alphanumeric characters, underscores, and hyphens')
        return v.upper()

    @field_validator('tick_type')
    @classmethod
    def validate_tick_type(cls, v):
        if v not in ['trade', 'bid', 'ask']:
            raise ValueError('tick_type must be one of: trade, bid, ask')
        return v