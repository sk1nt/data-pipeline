from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from decimal import Decimal

class TickData(BaseModel):
    symbol: str
    timestamp: datetime
    price: Decimal
    volume: Optional[int] = None
    tick_type: str  # trade, bid, ask
    source: str  # sierra_chart, gexbot, tastyttrade