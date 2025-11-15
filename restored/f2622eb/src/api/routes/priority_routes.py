"""
Priority API routes for the GEX data ingestion system.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ...lib.logging import get_logger
from ...lib.exceptions import GEXPriorityError, ValidationError
from ...services.priority_service import PriorityService
from ...models.priority_request import PriorityRequest
from ...models.data_source import DataSource
from ...models.enums import PriorityLevel, GEXDataType

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["priority"])


# Pydantic models for API requests/responses
class PriorityRequestCreate(BaseModel):
    """Request model for creating a priority request."""
    data_type: GEXDataType
    priority_level: PriorityLevel
    source_id: UUID
    deadline: Optional[datetime] = None
    metadata: Optional[dict] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class PriorityRequestResponse(BaseModel):
    """Response model for priority requests."""
    request_id: UUID
    data_type: str
    priority_level: str
    priority_score: float
    status: str
    submitted_at: datetime
    deadline: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class DataSourceResponse(BaseModel):
    """Response model for data sources."""
    source_id: UUID
    base_url: str
    name: str
    reliability_score: float
    success_rate: float
    total_requests: int
    successful_requests: int
    is_active: bool
    last_successful_fetch: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class DataSourceCreate(BaseModel):
    """Request model for creating a data source."""
    base_url: str
    name: str


# Dependency to get priority service
def get_priority_service() -> PriorityService:
    """Get priority service instance."""
    # This would typically come from dependency injection
    # For now, we'll create it here (in production, use proper DI)
    from ...lib.priority_db import PriorityDatabaseManager
    from ...lib.redis_client import RedisClient

    db_manager = PriorityDatabaseManager()
    redis_client = RedisClient()
    return PriorityService(db_manager, redis_client)


@router.post("/ingest/priority", response_model=PriorityRequestResponse)
async def submit_priority_request(
    request_data: PriorityRequestCreate,
    background_tasks: BackgroundTasks,
    service: PriorityService = Depends(get_priority_service)
) -> PriorityRequestResponse:
    """
    Submit a new priority data ingestion request.

    This endpoint accepts requests for GEX data ingestion with specified
    priority levels and deadlines. The system will queue and process
    requests based on calculated priority scores.
    """
    try:
        # Get the data source
        data_source = await service.get_data_source(request_data.source_id)
        if not data_source:
            raise HTTPException(
                status_code=404,
                detail=f"Data source {request_data.source_id} not found"
            )

        if not data_source.is_active:
            raise HTTPException(
                status_code=400,
                detail=f"Data source {request_data.source_id} is not active"
            )

        # Create priority request
        priority_request = PriorityRequest(
            data_type=request_data.data_type,
            priority_level=request_data.priority_level,
            source_id=request_data.source_id,
            deadline=request_data.deadline,
            metadata=request_data.metadata
        )

        # Submit the request
        submitted_request = await service.submit_priority_request(
            priority_request,
            data_source
        )

        # Add background task to check for overdue requests
        background_tasks.add_task(check_overdue_requests, service)

        logger.info(f"Submitted priority request {submitted_request.request_id}")

        return PriorityRequestResponse(
            request_id=submitted_request.request_id,
            data_type=submitted_request.data_type.value,
            priority_level=submitted_request.priority_level.value,
            priority_score=submitted_request.priority_score,
            status=submitted_request.status.value,
            submitted_at=submitted_request.submitted_at,
            deadline=submitted_request.deadline,
            completed_at=submitted_request.completed_at,
            error_message=submitted_request.error_message
        )

    except ValidationError as e:
        logger.warning(f"Validation error in priority request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except GEXPriorityError as e:
        logger.error(f"Priority service error: {e}")
        raise HTTPException(status_code=500, detail="Internal priority service error")
    except Exception as e:
        logger.error(f"Unexpected error in submit_priority_request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sources", response_model=List[DataSourceResponse])
async def get_data_sources(
    service: PriorityService = Depends(get_priority_service)
) -> List[DataSourceResponse]:
    """
    Get all registered data sources.

    Returns a list of all data sources with their current reliability
    metrics and status information.
    """
    try:
        sources = await service.get_all_data_sources()

        return [
            DataSourceResponse(
                source_id=source.source_id,
                base_url=source.base_url,
                name=source.name,
                reliability_score=source.reliability_score,
                success_rate=source.success_rate,
                total_requests=source.total_requests,
                successful_requests=source.successful_requests,
                is_active=source.is_active,
                last_successful_fetch=source.last_successful_fetch,
                created_at=source.created_at
            )
            for source in sources
        ]

    except Exception as e:
        logger.error(f"Error getting data sources: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/sources", response_model=DataSourceResponse)
async def register_data_source(
    source_data: DataSourceCreate,
    service: PriorityService = Depends(get_priority_service)
) -> DataSourceResponse:
    """
    Register a new data source.

    This endpoint allows registering new data sources for GEX data
    ingestion with initial reliability metrics.
    """
    try:
        # Create data source
        data_source = DataSource(
            base_url=source_data.base_url,
            name=source_data.name
        )

        # Register the source
        registered_source = await service.register_data_source(data_source)

        logger.info(f"Registered new data source {registered_source.source_id}")

        return DataSourceResponse(
            source_id=registered_source.source_id,
            base_url=registered_source.base_url,
            name=registered_source.name,
            reliability_score=registered_source.reliability_score,
            success_rate=registered_source.success_rate,
            total_requests=registered_source.total_requests,
            successful_requests=registered_source.successful_requests,
            is_active=registered_source.is_active,
            last_successful_fetch=registered_source.last_successful_fetch,
            created_at=registered_source.created_at
        )

    except ValidationError as e:
        logger.warning(f"Validation error in data source registration: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error registering data source: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/requests/pending", response_model=List[PriorityRequestResponse])
async def get_pending_requests(
    limit: int = 100,
    service: PriorityService = Depends(get_priority_service)
) -> List[PriorityRequestResponse]:
    """
    Get pending priority requests.

    Returns a list of pending requests ordered by priority score.
    """
    try:
        requests = await service.get_pending_requests(limit)

        return [
            PriorityRequestResponse(
                request_id=request.request_id,
                data_type=request.data_type.value,
                priority_level=request.priority_level.value,
                priority_score=request.priority_score,
                status=request.status.value,
                submitted_at=request.submitted_at,
                deadline=request.deadline,
                completed_at=request.completed_at,
                error_message=request.error_message
            )
            for request in requests
        ]

    except Exception as e:
        logger.error(f"Error getting pending requests: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def check_overdue_requests(service: PriorityService) -> None:
    """Background task to check and log overdue requests."""
    try:
        overdue = await service.get_overdue_requests()
        if overdue:
            logger.warning(f"Found {len(overdue)} overdue priority requests")
            # In a real implementation, you might want to escalate these
    except Exception as e:
        logger.error(f"Error checking overdue requests: {e}")