"""
DataSource model for the GEX priority system.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from uuid import uuid4, UUID

from lib.logging import get_logger

logger = get_logger(__name__)


class DataSource(BaseModel):
    """Model representing a GEX data source with reliability metrics."""

    source_id: UUID = Field(default_factory=uuid4)
    base_url: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    reliability_score: float = Field(default=1.0, ge=0.0, le=1.0)
    last_successful_fetch: Optional[datetime] = None
    total_requests: int = Field(default=0, ge=0)
    successful_requests: int = Field(default=0, ge=0)
    average_response_time: Optional[timedelta] = None
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: f"PT{int(v.total_seconds())}S" if v else None,
        }

    @validator('base_url')
    def validate_base_url(cls, v):
        """Validate that base_url is a valid HTTP/HTTPS URL."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('base_url must be a valid HTTP/HTTPS URL')
        # Ensure it doesn't end with a path (should be base URL)
        if '/' in v[len('https://'):] and not v.endswith('/'):
            # Allow trailing slash
            pass
        return v.rstrip('/')

    @validator('successful_requests')
    def validate_successful_requests(cls, v, values):
        """Validate that successful_requests doesn't exceed total_requests."""
        if v > values.get('total_requests', 0):
            raise ValueError('successful_requests cannot exceed total_requests')
        return v

    @validator('updated_at')
    def validate_updated_at(cls, v, values):
        """Validate that updated_at is not before created_at."""
        if values.get('created_at') and v < values['created_at']:
            raise ValueError('updated_at cannot be before created_at')
        return v

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a property."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    def record_request(self, success: bool, response_time: Optional[timedelta] = None) -> None:
        """Record a request attempt and update metrics."""
        self.total_requests += 1

        if success:
            self.successful_requests += 1
            self.last_successful_fetch = datetime.utcnow()

        if response_time is not None:
            if self.average_response_time is None:
                self.average_response_time = response_time
            else:
                # Exponential moving average
                alpha = 0.1  # Weight for new measurement
                current_seconds = self.average_response_time.total_seconds()
                new_seconds = response_time.total_seconds()
                smoothed_seconds = (1 - alpha) * current_seconds + alpha * new_seconds
                self.average_response_time = timedelta(seconds=smoothed_seconds)

        # Update reliability score based on recent performance
        self.reliability_score = self._calculate_reliability_score()
        self.updated_at = datetime.utcnow()

    def _calculate_reliability_score(self) -> float:
        """Calculate reliability score based on success rate and recency."""
        if self.total_requests == 0:
            return 1.0

        # Base score from success rate
        base_score = self.success_rate

        # Penalty for low request volume (less confidence)
        if self.total_requests < 10:
            base_score *= 0.8

        # Recency factor - reduce score if no recent success
        if self.last_successful_fetch:
            days_since_success = (datetime.utcnow() - self.last_successful_fetch).days
            if days_since_success > 7:
                recency_penalty = max(0.5, 1.0 - (days_since_success - 7) * 0.1)
                base_score *= recency_penalty

        return round(base_score, 3)

    def dict_for_db(self) -> Dict[str, Any]:
        """Convert to dictionary format suitable for database storage."""
        data = self.dict()
        # Convert UUID to string for JSON storage
        data['source_id'] = str(data['source_id'])
        # Convert timedelta to string if present
        if data['average_response_time']:
            data['average_response_time'] = str(data['average_response_time'])
        return data

    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> 'DataSource':
        """Create instance from database dictionary."""
        # Convert string fields back to appropriate types
        if 'source_id' in data and isinstance(data['source_id'], str):
            data['source_id'] = UUID(data['source_id'])
        if 'average_response_time' in data and isinstance(data['average_response_time'], str):
            # Parse duration string (simple implementation)
            if data['average_response_time'].startswith('PT'):
                seconds_str = data['average_response_time'][2:-1]  # Remove PT and S
                try:
                    seconds = int(seconds_str)
                    data['average_response_time'] = timedelta(seconds=seconds)
                except ValueError:
                    data['average_response_time'] = None
        return cls(**data)

    def is_reliable(self, threshold: float = 0.8) -> bool:
        """Check if source meets reliability threshold."""
        return self.reliability_score >= threshold

    def __str__(self) -> str:
        return f"DataSource(id={self.source_id}, name={self.name}, reliability={self.reliability_score:.2f})"