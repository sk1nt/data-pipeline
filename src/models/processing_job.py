"""
Processing job model for GEX data processing pipeline.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from uuid import UUID

from lib.logging import get_logger
from lib.utils import generate_uuid
from ..models.base import BaseModel
from ..models.enums import JobStatus, DataType

logger = get_logger(__name__)


class ProcessingJob(BaseModel):
    """Model representing a data processing job in the priority queue."""

    def __init__(
        self,
        job_id: Optional[UUID] = None,
        request_id: Optional[UUID] = None,
        data_type: DataType = DataType.TICK_DATA,
        priority_score: float = 0.0,
        market_symbol: Optional[str] = None,
        data_source_id: Optional[UUID] = None,
        status: JobStatus = JobStatus.PENDING,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        processing_time_seconds: Optional[float] = None,
        error_message: Optional[str] = None,
        retry_count: int = 0,
        max_retries: int = 3,
    ):
        """
        Initialize a processing job.

        Args:
            job_id: Unique identifier for the job
            request_id: ID of the priority request this job fulfills
            data_type: Type of data to process
            priority_score: Priority score (0.0-1.0)
            market_symbol: Market symbol for the data
            data_source_id: ID of the data source
            status: Current job status
            metadata: Additional job metadata
            created_at: When the job was created
            started_at: When processing started
            completed_at: When processing completed
            processing_time_seconds: Total processing time
            error_message: Error message if failed
            retry_count: Number of retries attempted
            max_retries: Maximum number of retries allowed
        """
        self.job_id = job_id or generate_uuid()
        self.request_id = request_id
        self.data_type = data_type
        self.priority_score = priority_score
        self.market_symbol = market_symbol
        self.data_source_id = data_source_id
        self.status = status
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.utcnow()
        self.started_at = started_at
        self.completed_at = completed_at
        self.processing_time_seconds = processing_time_seconds
        self.error_message = error_message
        self.retry_count = retry_count
        self.max_retries = max_retries

    def dict_for_db(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'job_id': str(self.job_id),
            'request_id': str(self.request_id) if self.request_id else None,
            'data_type': self.data_type.value,
            'priority_score': self.priority_score,
            'market_symbol': self.market_symbol,
            'data_source_id': str(self.data_source_id) if self.data_source_id else None,
            'status': self.status.value,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'processing_time_seconds': self.processing_time_seconds,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
        }

    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> 'ProcessingJob':
        """Create instance from database dictionary."""
        # Convert string UUIDs back to UUID objects
        job_id = UUID(data['job_id']) if data.get('job_id') else None
        request_id = UUID(data['request_id']) if data.get('request_id') else None
        data_source_id = UUID(data['data_source_id']) if data.get('data_source_id') else None

        # Convert ISO strings back to datetime
        created_at = datetime.fromisoformat(data['created_at']) if data.get('created_at') else None
        started_at = datetime.fromisoformat(data['started_at']) if data.get('started_at') else None
        completed_at = datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None

        # Convert data_type string to enum
        data_type = DataType(data['data_type']) if data.get('data_type') else DataType.TICK_DATA

        # Convert status string to enum
        status = JobStatus(data['status']) if data.get('status') else JobStatus.PENDING

        return cls(
            job_id=job_id,
            request_id=request_id,
            data_type=data_type,
            priority_score=data.get('priority_score', 0.0),
            market_symbol=data.get('market_symbol'),
            data_source_id=data_source_id,
            status=status,
            metadata=data.get('metadata', {}),
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            processing_time_seconds=data.get('processing_time_seconds'),
            error_message=data.get('error_message'),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3),
        )

    def start_processing(self) -> None:
        """Mark the job as started."""
        if self.status != JobStatus.PENDING:
            logger.warning(f"Attempting to start job {self.job_id} with status {self.status.value}")
            return

        self.status = JobStatus.PROCESSING
        self.started_at = datetime.utcnow()
        logger.info(f"Started processing job {self.job_id}")

    def complete_processing(self, processing_time: Optional[float] = None) -> None:
        """Mark the job as completed successfully."""
        if self.status != JobStatus.PROCESSING:
            logger.warning(f"Attempting to complete job {self.job_id} with status {self.status.value}")
            return

        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        if processing_time is not None:
            self.processing_time_seconds = processing_time
        elif self.started_at:
            self.processing_time_seconds = (self.completed_at - self.started_at).total_seconds()

        logger.info(f"Completed processing job {self.job_id} in {self.processing_time_seconds:.2f}s")

    def fail_processing(self, error_message: str, processing_time: Optional[float] = None) -> None:
        """Mark the job as failed."""
        self.status = JobStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.utcnow()

        if processing_time is not None:
            self.processing_time_seconds = processing_time
        elif self.started_at:
            self.processing_time_seconds = (self.completed_at - self.started_at).total_seconds()

        logger.warning(f"Failed processing job {self.job_id}: {error_message}")

    def can_retry(self) -> bool:
        """Check if the job can be retried."""
        return self.retry_count < self.max_retries and self.status in [JobStatus.FAILED, JobStatus.PENDING]

    def increment_retry(self) -> None:
        """Increment the retry count."""
        self.retry_count += 1
        self.status = JobStatus.PENDING  # Reset to pending for retry
        logger.info(f"Incremented retry count for job {self.job_id} to {self.retry_count}")

    def is_overdue(self, timeout_seconds: int = 300) -> bool:
        """
        Check if the job is overdue based on processing timeout.

        Args:
            timeout_seconds: Maximum allowed processing time in seconds

        Returns:
            True if the job has been processing longer than timeout
        """
        if self.status != JobStatus.PROCESSING or not self.started_at:
            return False

        elapsed = (datetime.utcnow() - self.started_at).total_seconds()
        return elapsed > timeout_seconds

    def get_processing_duration(self) -> Optional[float]:
        """Get the current or total processing duration in seconds."""
        if self.processing_time_seconds is not None:
            return self.processing_time_seconds

        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()

        if self.started_at:
            return (datetime.utcnow() - self.started_at).total_seconds()

        return None

    def __str__(self) -> str:
        """String representation of the job."""
        return f"ProcessingJob(job_id={self.job_id}, request_id={self.request_id}, status={self.status.value}, priority={self.priority_score:.3f})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"ProcessingJob(job_id={self.job_id!r}, request_id={self.request_id!r}, "
                f"data_type={self.data_type.value!r}, priority_score={self.priority_score!r}, "
                f"status={self.status.value!r})")