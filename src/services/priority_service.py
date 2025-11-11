"""
Priority service for managing GEX data ingestion requests.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from ..lib.logging import get_logger
from ..lib.priority_db import PriorityDatabaseManager
from ..lib.redis_client import RedisClient
from ..lib.exceptions import DatabaseError
from ..models.priority_request import PriorityRequest
from ..models.data_source import DataSource
from ..models.enums import JobStatus
from .rule_engine import RuleEngine

logger = get_logger(__name__)


class PriorityService:
    """Service for managing priority-based data ingestion requests."""

    def __init__(self, db_manager: PriorityDatabaseManager, redis_client: RedisClient):
        self.db = db_manager
        self.redis = redis_client
        self.rule_engine = RuleEngine()

    async def submit_priority_request(
        self,
        request: PriorityRequest,
        data_source: DataSource
    ) -> PriorityRequest:
        """
        Submit a new priority request for processing.

        Args:
            request: The priority request to submit
            data_source: The data source for this request

        Returns:
            The submitted request with assigned ID and calculated priority

        Raises:
            ValidationError: If request validation fails
            DatabaseError: If database operation fails
        """
        try:
            # Validate data source reliability
            if not data_source.is_reliable():
                logger.warning(f"Data source {data_source.source_id} has low reliability score: {data_source.reliability_score}")
                # Still allow but log warning

            # Get active priority rules (use defaults for now)
            active_rules = self.rule_engine.create_default_rules()

            # Use rule engine for automatic priority assignment
            calculated_score, matched_rules = self.rule_engine.evaluate_priority_request(
                request, data_source, active_rules
            )

            # Override the manual priority level with rule-based score
            request.priority_score = calculated_score

            # Update priority level based on calculated score
            request.priority_level = self.rule_engine.get_priority_level_from_score(calculated_score)

            # Set status and timestamp
            request.status = JobStatus.PENDING
            request.submitted_at = datetime.utcnow()

            # Log rule matches for audit
            if matched_rules:
                logger.info(f"AUDIT: Request {request.request_id} auto-assigned priority {request.priority_level.value} ({calculated_score:.3f}) based on rules: {', '.join(matched_rules)}")
            else:
                logger.info(f"AUDIT: Request {request.request_id} auto-assigned default priority {request.priority_level.value} ({calculated_score:.3f}) - no rules matched")

            # Save to database
            request_dict = request.dict_for_db()
            request_id = await self.db.insert_priority_request(request_dict)

            # Update request with database ID
            request.request_id = request_id

            # Add to Redis priority queue
            await self.redis.add_to_priority_queue(
                request_id=str(request_id),
                priority_score=request.priority_score,
                data={
                    'source_id': str(data_source.source_id),
                    'data_type': request.data_type.value,
                    'deadline': request.deadline.isoformat() if request.deadline else None
                }
            )

            logger.info(f"Submitted priority request {request_id} with score {request.priority_score}")
            return request

        except Exception as e:
            logger.error(f"Failed to submit priority request: {e}")
            raise DatabaseError(f"Failed to submit priority request: {str(e)}") from e

    async def get_next_request(self) -> Optional[PriorityRequest]:
        """
        Get the next highest priority request from the queue.

        Returns:
            The next priority request, or None if queue is empty
        """
        try:
            # Get next item from Redis priority queue
            queue_item = await self.redis.get_next_from_priority_queue()

            if not queue_item:
                return None

            request_id_str, data = queue_item
            request_id = UUID(request_id_str)

            # Fetch full request from database
            request_dict = await self.db.get_priority_request(request_id)
            if not request_dict:
                logger.warning(f"Request {request_id} not found in database")
                return None

            request = PriorityRequest.from_db_dict(request_dict)

            # Update status to processing
            await self.update_request_status(request_id, JobStatus.PROCESSING)

            logger.info(f"Retrieved next priority request {request_id}")
            return request

        except Exception as e:
            logger.error(f"Failed to get next request: {e}")
            raise DatabaseError(f"Failed to get next request: {str(e)}") from e

    async def update_request_status(
        self,
        request_id: UUID,
        status: JobStatus,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update the status of a priority request.

        Args:
            request_id: The request ID to update
            status: New status
            error_message: Optional error message for failed requests
        """
        try:
            update_data = {
                'status': status.value,
                'updated_at': datetime.utcnow().isoformat()
            }

            if status == JobStatus.COMPLETED:
                update_data['completed_at'] = datetime.utcnow().isoformat()
            elif status == JobStatus.FAILED and error_message:
                update_data['error_message'] = error_message

            await self.db.update_priority_request(request_id, update_data)

            # If failed, remove from processing queue
            if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                await self.redis.remove_from_processing(str(request_id))

            logger.info(f"Updated request {request_id} status to {status.value}")

        except Exception as e:
            logger.error(f"Failed to update request status: {e}")
            raise DatabaseError(f"Failed to update request status: {str(e)}") from e

    async def get_pending_requests(self, limit: int = 100) -> List[PriorityRequest]:
        """
        Get all pending requests ordered by priority score.

        Args:
            limit: Maximum number of requests to return

        Returns:
            List of pending priority requests
        """
        try:
            request_dicts = await self.db.get_pending_requests(limit)
            requests = [PriorityRequest.from_db_dict(rd) for rd in request_dicts]
            logger.info(f"Retrieved {len(requests)} pending requests")
            return requests

        except Exception as e:
            logger.error(f"Failed to get pending requests: {e}")
            raise DatabaseError(f"Failed to get pending requests: {str(e)}") from e

    async def get_overdue_requests(self) -> List[PriorityRequest]:
        """
        Get requests that are past their deadline.

        Returns:
            List of overdue priority requests
        """
        try:
            request_dicts = await self.db.get_overdue_requests()
            requests = [PriorityRequest.from_db_dict(rd) for rd in request_dicts]
            if requests:
                logger.warning(f"Found {len(requests)} overdue requests")
            return requests

        except Exception as e:
            logger.error(f"Failed to get overdue requests: {e}")
            raise DatabaseError(f"Failed to get overdue requests: {str(e)}") from e

    async def get_data_source(self, source_id: UUID) -> Optional[DataSource]:
        """
        Get a data source by ID.

        Args:
            source_id: The source ID to fetch

        Returns:
            The data source, or None if not found
        """
        try:
            source_dict = await self.db.get_data_source(source_id)
            if not source_dict:
                return None
            return DataSource.from_db_dict(source_dict)

        except Exception as e:
            logger.error(f"Failed to get data source {source_id}: {e}")
            raise DatabaseError(f"Failed to get data source: {str(e)}") from e

    async def update_data_source_metrics(
        self,
        source_id: UUID,
        success: bool,
        response_time: Optional[timedelta] = None
    ) -> None:
        """
        Update data source metrics after a request.

        Args:
            source_id: The source ID to update
            success: Whether the request was successful
            response_time: Optional response time for the request
        """
        try:
            source = await self.get_data_source(source_id)
            if not source:
                logger.warning(f"Data source {source_id} not found for metrics update")
                return

            # Record the request
            source.record_request(success, response_time)

            # Save updated metrics
            await self.db.update_data_source(source_id, source.dict_for_db())

            logger.info(f"Updated metrics for data source {source_id}")

        except Exception as e:
            logger.error(f"Failed to update data source metrics: {e}")
            raise DatabaseError(f"Failed to update data source metrics: {str(e)}") from e

    async def register_data_source(self, source: DataSource) -> DataSource:
        """
        Register a new data source.

        Args:
            source: The data source to register

        Returns:
            The registered data source with assigned ID
        """
        try:
            source_dict = source.dict_for_db()
            source_id = await self.db.insert_data_source(source_dict)
            source.source_id = source_id

            logger.info(f"Registered new data source {source_id}: {source.name}")
            return source

        except Exception as e:
            logger.error(f"Failed to register data source: {e}")
            raise DatabaseError(f"Failed to register data source: {str(e)}") from e

    async def get_all_data_sources(self) -> List[DataSource]:
        """
        Get all registered data sources.

        Returns:
            List of all data sources
        """
        try:
            source_dicts = await self.db.get_all_data_sources()
            sources = [DataSource.from_db_dict(sd) for sd in source_dicts]
            logger.info(f"Retrieved {len(sources)} data sources")
            return sources

        except Exception as e:
            logger.error(f"Failed to get data sources: {e}")
            raise DatabaseError(f"Failed to get data sources: {str(e)}") from e