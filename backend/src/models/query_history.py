from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

class QueryHistory(BaseModel):
    query_id: str  # UUID
    model_id: str
    query_type: str  # realtime, historical
    parameters: Dict[str, Any]
    timestamp: datetime
    response_time_ms: int
    data_points_returned: int