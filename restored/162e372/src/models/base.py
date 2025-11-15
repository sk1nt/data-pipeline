"""
Base Pydantic models and enums for the GEX priority system.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, Field, validator, UUID4
from uuid import uuid4


# Enums
class GEXDataType(str, Enum):
    """Types of GEX data that can be processed."""
    HISTORICAL = "historical"
    REAL_TIME = "real_time"
    SNAPSHOT = "snapshot"


class PriorityLevel(str, Enum):
    """Priority levels for data processing."""
    CRITICAL = "critical"  # <30 second processing guarantee
    HIGH = "high"          # <5 minute processing
    MEDIUM = "medium"      # <30 minute processing
    LOW = "low"           # Best effort processing


class JobStatus(str, Enum):
    """Status of processing jobs."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Base models
class BaseEntity(BaseModel):
    """Base class for all entities with common fields."""
    id: UUID4 = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            UUID4: str,
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            time: lambda v: v.isoformat(),
        }


class PriorityRequestBase(BaseModel):
    """Base model for priority requests."""
    request_id: UUID4 = Field(default_factory=uuid4)
    source_url: str = Field(..., min_length=1)
    data_type: GEXDataType
    market_symbol: str = Field(..., min_length=1, max_length=10)
    requested_at: datetime = Field(default_factory=datetime.utcnow)
    priority_score: float = Field(..., ge=0.0, le=1.0)
    priority_level: PriorityLevel
    estimated_processing_time: Optional[timedelta] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('source_url')
    def validate_source_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('source_url must be a valid HTTP/HTTPS URL')
        return v

    @validator('market_symbol')
    def validate_market_symbol(cls, v):
        # Basic validation for common optionable symbols
        valid_symbols = {'NDX', 'SPX', 'QQQ', 'IWM', 'MNQ', 'ES', 'NQ', 'RTY'}
        if v.upper() not in valid_symbols and not v.upper().endswith(('C', 'P')):
            # Allow option symbols like NDXC230616 or NDXP230616
            pass
        return v.upper()


class ProcessingJobBase(BaseModel):
    """Base model for processing jobs."""
    job_id: UUID4 = Field(default_factory=uuid4)
    request_id: UUID4
    status: JobStatus = JobStatus.QUEUED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_duration: Optional[timedelta] = None
    records_processed: int = Field(default=0, ge=0)
    data_size_bytes: int = Field(default=0, ge=0)
    error_message: Optional[str] = None
    retry_count: int = Field(default=0, ge=0)

    @validator('completed_at')
    def validate_completed_at(cls, v, values):
        if v and values.get('started_at') and v < values['started_at']:
            raise ValueError('completed_at cannot be before started_at')
        return v

    @validator('processing_duration')
    def calculate_processing_duration(cls, v, values):
        if not v and values.get('started_at') and values.get('completed_at'):
            return values['completed_at'] - values['started_at']
        return v


class DataSourceBase(BaseModel):
    """Base model for data sources."""
    source_id: UUID4 = Field(default_factory=uuid4)
    base_url: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    reliability_score: float = Field(default=1.0, ge=0.0, le=1.0)
    last_successful_fetch: Optional[datetime] = None
    total_requests: int = Field(default=0, ge=0)
    successful_requests: int = Field(default=0, ge=0)
    average_response_time: Optional[timedelta] = None
    is_active: bool = Field(default=True)

    @validator('base_url')
    def validate_base_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('base_url must be a valid HTTP/HTTPS URL')
        return v

    @validator('successful_requests')
    def validate_successful_requests(cls, v, values):
        if v > values.get('total_requests', 0):
            raise ValueError('successful_requests cannot exceed total_requests')
        return v


class PriorityRuleBase(BaseModel):
    """Base model for priority rules."""
    rule_id: UUID4 = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    condition: str = Field(..., min_length=1)
    priority_score: float = Field(..., ge=0.0, le=1.0)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class GEXSnapshotBase(BaseModel):
    """Base model for GEX snapshots."""
    snapshot_id: UUID4 = Field(default_factory=uuid4)
    job_id: UUID4
    market_symbol: str = Field(..., min_length=1, max_length=10)
    snapshot_date: date
    snapshot_time: time
    total_open_interest: int = Field(default=0, ge=0)
    total_volume: int = Field(default=0, ge=0)
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    data_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GEXStrikeBase(BaseModel):
    """Base model for individual GEX strikes."""
    snapshot_id: UUID4
    strike_price: float = Field(..., gt=0)
    call_open_interest: int = Field(default=0, ge=0)
    put_open_interest: int = Field(default=0, ge=0)
    call_volume: int = Field(default=0, ge=0)
    put_volume: int = Field(default=0, ge=0)
    call_bid: Optional[float] = Field(None, ge=0)
    call_ask: Optional[float] = Field(None, ge=0)
    put_bid: Optional[float] = Field(None, ge=0)
    put_ask: Optional[float] = Field(None, ge=0)
    gamma: Optional[float] = None
    delta: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

    @validator('call_ask', 'put_ask')
    def validate_ask_prices(cls, v, values, field):
        if v is not None:
            bid_field = field.name.replace('ask', 'bid')
            bid_price = values.get(bid_field)
            if bid_price is not None and v < bid_price:
                raise ValueError(f'{field.name} cannot be less than {bid_field}')
        return v


# Utility functions
def validate_uuid_format(uuid_str: str) -> bool:
    """Validate UUID string format."""
    try:
        UUID4(uuid_str)
        return True
    except ValueError:
        return False


def create_priority_request(**kwargs) -> PriorityRequestBase:
    """Factory function for creating priority requests."""
    return PriorityRequestBase(**kwargs)


def create_processing_job(**kwargs) -> ProcessingJobBase:
    """Factory function for creating processing jobs."""
    return ProcessingJobBase(**kwargs)


def create_data_source(**kwargs) -> DataSourceBase:
    """Factory function for creating data sources."""
    return DataSourceBase(**kwargs)