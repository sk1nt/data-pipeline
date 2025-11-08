from fastapi import APIRouter
import duckdb
import os
import logging
from datetime import datetime
from backend.src.models.service_status import ServiceStatus

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/status")
async def get_system_status():
    """Get operational status of all pipeline services."""
    logger.info("Status check requested")

    # Mock service statuses - in full implementation, check actual services
    services = [
        ServiceStatus(
            service_name="ingestion",
            current_status="healthy",
            last_update_time=datetime.now(),
            uptime_percentage=99.5
        ),
        ServiceStatus(
            service_name="processing",
            current_status="healthy",
            last_update_time=datetime.now(),
            uptime_percentage=98.2
        ),
        ServiceStatus(
            service_name="api",
            current_status="healthy",
            last_update_time=datetime.now(),
            uptime_percentage=99.9
        ),
        ServiceStatus(
            service_name="storage",
            current_status="healthy",
            last_update_time=datetime.now(),
            uptime_percentage=100.0
        )
    ]

    return {
        "services": [service.model_dump() for service in services],
        "timestamp": datetime.now()
    }