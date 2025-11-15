from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class TickRecord(BaseModel):
    timestamp: datetime = Field(..., description="Trade timestamp")
    symbol: str = Field(..., description="MNQ or NQ")
    bid: Optional[float] = Field(None, ge=0, description="Bid price")
    ask: Optional[float] = Field(None, ge=0, description="Ask price")
    last: Optional[float] = Field(None, ge=0, description="Last traded price")
    volume: Optional[int] = Field(None, ge=0, description="Trade volume")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TickRecord':
        """Create instance from dictionary, handling timestamp conversion."""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return self.dict()

    def validate(self) -> List[str]:
        """Validate tick record."""
        errors = []
        if self.symbol not in ['MNQ', 'NQ', 'MES']:
            errors.append(f"Invalid symbol: {self.symbol}")
        if self.bid is not None and self.ask is not None and self.bid > self.ask:
            errors.append("Bid price cannot be higher than ask price")
        return errors