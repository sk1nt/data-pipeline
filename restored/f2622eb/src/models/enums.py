"""
Priority level enumerations for the GEX priority system.
"""

from enum import Enum


class PriorityLevel(str, Enum):
    """Priority levels for data processing with associated guarantees."""
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


class GEXDataType(str, Enum):
    """Types of GEX data that can be processed."""
    HISTORICAL = "historical"
    REAL_TIME = "real_time"
    SNAPSHOT = "snapshot"