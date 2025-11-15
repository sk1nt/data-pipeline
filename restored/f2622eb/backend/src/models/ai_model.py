from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class AIModel(BaseModel):
    model_id: str
    name: Optional[str] = None
    access_permissions: Dict[str, Any]
    api_key_hash: str
    created_at: datetime
    last_access: Optional[datetime] = None
    query_count: int = 0