from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from decimal import Decimal

class EnrichedData(BaseModel):
    symbol: str
    interval_start: datetime
    interval_end: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    total_volume: int
    vwap: Optional[Decimal] = None