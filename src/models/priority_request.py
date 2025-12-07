"""
PriorityRequest model for the GEX priority system.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from uuid import uuid4, UUID

from .enums import GEXDataType, PriorityLevel
from lib.logging import get_logger

logger = get_logger(__name__)


class PriorityRequest(BaseModel):
    """Model representing a priority request for GEX data ingestion."""

    request_id: UUID = Field(default_factory=uuid4)
    source_url: str = Field(..., min_length=1)
    data_type: GEXDataType
    market_symbol: str = Field(..., min_length=1, max_length=10)
    requested_at: datetime = Field(default_factory=datetime.utcnow)
    priority_score: float = Field(..., ge=0.0, le=1.0)
    priority_level: PriorityLevel
    estimated_processing_time: Optional[timedelta] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: f"PT{int(v.total_seconds())}S" if v else None,
        }

    @validator('source_url')
    def validate_source_url(cls, v):
        """Validate that source_url is a valid HTTP/HTTPS URL."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('source_url must be a valid HTTP/HTTPS URL')
        return v

    @validator('market_symbol')
    def validate_market_symbol(cls, v):
        """Validate market symbol format."""
        # Basic validation for common optionable symbols
        valid_symbols = {'NDX', 'SPX', 'QQQ', 'IWM', 'MNQ', 'ES', 'NQ', 'RTY'}

        symbol_upper = v.upper()

        # Check if it's a known index
        if symbol_upper in valid_symbols:
            return symbol_upper

        # Check if it's an option symbol (e.g., NDXC230616, NDXP230616)
        if len(symbol_upper) >= 6:
            base_symbol = symbol_upper[:-7]  # Remove date and C/P
            option_type = symbol_upper[-7]   # C or P
            date_part = symbol_upper[-6:]    # Date part

            if (base_symbol in valid_symbols and
                option_type in ('C', 'P') and
                date_part.isdigit() and
                len(date_part) == 6):
                return symbol_upper

        raise ValueError(f'Invalid market symbol: {v}')

    @validator('requested_at')
    def validate_requested_at(cls, v):
        """Validate that requested_at is not in the future."""
        if v > datetime.utcnow():
            raise ValueError('requested_at cannot be in the future')
        return v

    def dict_for_db(self) -> Dict[str, Any]:
        """Convert to dictionary format suitable for database storage."""
        data = self.dict()
        # Convert UUID to string for JSON storage
        data['request_id'] = str(data['request_id'])
        # Convert timedelta to string if present
        if data['estimated_processing_time']:
            data['estimated_processing_time'] = str(data['estimated_processing_time'])
        return data

    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> 'PriorityRequest':
        """Create instance from database dictionary."""
        # Convert string fields back to appropriate types
        if 'request_id' in data and isinstance(data['request_id'], str):
            data['request_id'] = UUID(data['request_id'])
        if 'estimated_processing_time' in data and isinstance(data['estimated_processing_time'], str):
            # Parse duration string (simple implementation)
            if data['estimated_processing_time'].startswith('PT'):
                seconds_str = data['estimated_processing_time'][2:-1]  # Remove PT and S
                try:
                    seconds = int(seconds_str)
                    data['estimated_processing_time'] = timedelta(seconds=seconds)
                except ValueError:
                    data['estimated_processing_time'] = None
        return cls(**data)

    def get_processing_deadline(self) -> datetime:
        """Calculate processing deadline based on priority level."""
        if self.estimated_processing_time:
            return self.requested_at + self.estimated_processing_time

        # Default deadlines based on priority level
        deadlines = {
            PriorityLevel.CRITICAL: timedelta(seconds=30),
            PriorityLevel.HIGH: timedelta(minutes=5),
            PriorityLevel.MEDIUM: timedelta(minutes=30),
            PriorityLevel.LOW: timedelta(hours=2),
        }

        return self.requested_at + deadlines[self.priority_level]

    def is_overdue(self) -> bool:
        """Check if request is overdue for processing."""
        return datetime.utcnow() > self.get_processing_deadline()

    def __str__(self) -> str:
        return f"PriorityRequest(id={self.request_id}, symbol={self.market_symbol}, priority={self.priority_level.value})"