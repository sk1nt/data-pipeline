"""
Data models for market depth data.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class DepthSnapshot:
    """
    Represents market depth at a point in time.

    Fields:
        timestamp: Snapshot timestamp (UTC)
        bid_price_1: Best bid price
        bid_size_1: Best bid size
        ask_price_1: Best ask price
        ask_size_1: Best ask size
        ... (additional levels as available)
    """
    timestamp: datetime
    bid_price_1: float
    bid_size_1: int
    ask_price_1: float
    ask_size_1: int
    bid_price_2: Optional[float] = None
    bid_size_2: Optional[int] = None
    ask_price_2: Optional[float] = None
    ask_size_2: Optional[int] = None
    bid_price_3: Optional[float] = None
    bid_size_3: Optional[int] = None
    ask_price_3: Optional[float] = None
    ask_size_3: Optional[int] = None
    bid_price_4: Optional[float] = None
    bid_size_4: Optional[int] = None
    ask_price_4: Optional[float] = None
    ask_size_4: Optional[int] = None
    bid_price_5: Optional[float] = None
    bid_size_5: Optional[int] = None
    ask_price_5: Optional[float] = None
    ask_size_5: Optional[int] = None

    def __post_init__(self):
        """Validation after initialization."""
        if self.bid_price_1 >= self.ask_price_1:
            raise ValueError("Bid price must be less than ask price")
        if self.bid_size_1 < 0 or self.ask_size_1 < 0:
            raise ValueError("Sizes must be non-negative")
        if self.timestamp.tzinfo is not None:
            self.timestamp = self.timestamp.replace(tzinfo=None)

    @classmethod
    def from_scid_records(cls, records: list) -> 'DepthSnapshot':
        """
        Create DepthSnapshot from list of SCID records.

        Note: This is a placeholder - actual depth parsing would require
        separate .scdd file parsing for incremental updates.

        Args:
            records: List of parsed SCID records

        Returns:
            DepthSnapshot instance
        """
        # For now, create from first record
        # In practice, depth data comes from separate files
        if not records:
            raise ValueError("No records provided")

        record = records[0]
        return cls(
            timestamp=record['timestamp'],
            bid_price_1=record['low'],   # Bid price
            bid_size_1=int(record.get('bid_volume', 0)),
            ask_price_1=record['high'],  # Ask price
            ask_size_1=int(record.get('ask_volume', 0))
        )