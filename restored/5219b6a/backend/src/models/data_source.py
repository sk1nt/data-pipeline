from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class DataSource(BaseModel):
    source_id: str
    name: str
    type: str  # sierra_chart, gexbot, tastyttrade
    status: str  # active, inactive, error
    last_update: Optional[datetime] = None
    error_count: int = 0