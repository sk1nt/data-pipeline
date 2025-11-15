from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ServiceStatus(BaseModel):
    service_name: str
    current_status: str  # healthy, degraded, down
    last_update_time: datetime
    uptime_percentage: Optional[float] = None
    error_message: Optional[str] = None