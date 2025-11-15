"""Market data models for streaming publishers."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional


@dataclass
class TickEvent:
    """High-frequency tick update."""

    symbol: str
    timestamp: datetime
    last: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[float] = None

    def to_payload(self) -> dict:
        """Serialize to dict for downstream transports."""
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


@dataclass
class Level2Quote:
    """Represents a single level of market depth."""

    price: float
    size: float


@dataclass
class Level2Event:
    """Order book snapshot/update."""

    symbol: str
    timestamp: datetime
    bids: List[Level2Quote]
    asks: List[Level2Quote]

    def to_payload(self) -> dict:
        """Serialize to dict for downstream transports."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bids": [asdict(bid) for bid in self.bids],
            "asks": [asdict(ask) for ask in self.asks],
        }
