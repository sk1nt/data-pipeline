from pydantic import BaseModel, Field
from typing import List, Any, Dict
from datetime import datetime

class MarketDepth(BaseModel):
    timestamp: datetime = Field(..., description="Depth snapshot timestamp")
    symbol: str = Field(..., description="MNQ")
    bids: List[List[float]] = Field(..., description="Bid levels [price, size]")
    asks: List[List[float]] = Field(..., description="Ask levels [price, size]")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketDepth':
        """Create instance from dictionary, handling timestamp conversion."""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        d = self.dict()
        d['bids'] = str(d['bids'])  # Store as string
        d['asks'] = str(d['asks'])
        return d

    def validate(self) -> List[str]:
        """Validate market depth."""
        errors = []
        if self.symbol not in ['MNQ', 'NQ', 'MES']:
            errors.append(f"Invalid symbol: {self.symbol}")
        if not self.bids or not self.asks:
            errors.append("Bids and asks cannot be empty")
        # Check bids are sorted descending (best bid first)
        bid_prices = [level[0] for level in self.bids]
        if bid_prices != sorted(bid_prices, reverse=True):
            errors.append("Bids not properly sorted")
        # Check asks are sorted ascending (best ask first)
        ask_prices = [level[0] for level in self.asks]
        if ask_prices != sorted(ask_prices):
            errors.append("Asks not properly sorted")
        return errors