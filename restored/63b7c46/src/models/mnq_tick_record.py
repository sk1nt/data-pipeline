"""
Data models for MNQ tick data extraction.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class MnqTickRecord:
    """
    Represents an individual futures trade tick from SCID files.

    Fields:
        timestamp: Trade timestamp (UTC)
        price: Trade price
        volume: Trade volume
        tick_type: Trade direction (optional)
        ticker: Ticker symbol (normalized futures name)
    """
    timestamp: datetime
    price: float
    volume: int
    tick_type: Optional[str] = None
    ticker: str = "MNQ"

    def __post_init__(self):
        """Validation after initialization."""
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.volume <= 0:
            raise ValueError("Volume must be positive")
        if self.timestamp.tzinfo is not None:
            # Ensure UTC
            self.timestamp = self.timestamp.replace(tzinfo=None)

    @classmethod
    def from_scid_record(cls, record: dict) -> 'MnqTickRecord':
        """
        Create MnqTickRecord from parsed SCID record.

        Args:
            record: Parsed SCID record dict

        Returns:
            MnqTickRecord instance
        """
        return cls(
            timestamp=record['timestamp'],
            price=record['close'],  # Trade price
            volume=int(record['total_volume']),
            tick_type=cls._determine_tick_type(record)
        )

    @staticmethod
    def _determine_tick_type(record: dict) -> Optional[str]:
        """
        Determine trade direction from bid/ask volumes.

        Args:
            record: Parsed SCID record

        Returns:
            'buy', 'sell', or None
        """
        bid_vol = record.get('bid_volume', 0)
        ask_vol = record.get('ask_volume', 0)

        if bid_vol > ask_vol:
            return 'buy'
        elif ask_vol > bid_vol:
            return 'sell'
        return None