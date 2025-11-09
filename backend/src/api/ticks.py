from fastapi import APIRouter, Depends, Query, HTTPException
from typing import List
import time
import logging
from datetime import datetime
from backend.src.services.tick_service import tick_service
from backend.src.services.enriched_service import enriched_service
from backend.src.services.auth_service import verify_api_key, check_permissions

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/ticks/realtime")
async def get_realtime_ticks(
    symbols: List[str] = Query(..., description="Comma-separated list of financial symbols"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of ticks to return per symbol"),
    model_id: str = Depends(verify_api_key)
):
    """Get real-time tick data for specified symbols."""
    start_time = time.time()

    logger.info(f"Real-time query from model {model_id} for symbols {symbols}, limit {limit}")

    # Check permissions
    required_perms = {"symbols": symbols, "query_types": ["realtime"]}
    if not check_permissions(model_id, required_perms):
        logger.warning(f"Permission denied for model {model_id}")
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    # Get ticks
    ticks = tick_service.get_realtime_ticks(symbols, limit)

    response_time = int((time.time() - start_time) * 1000)

    logger.info(f"Real-time query completed in {response_time}ms, returned {len(ticks)} ticks")

    return {
        "data": [tick.model_dump() for tick in ticks],
        "metadata": {
            "query_id": "mock-uuid",  # Generate proper UUID in full implementation
            "response_time_ms": response_time,
            "data_points_returned": len(ticks),
            "cache_hit": True  # Determine based on source
        }
    }

@router.get("/ticks/historical")
async def get_historical_ticks(
    symbols: List[str] = Query(..., description="Comma-separated list of financial symbols"),
    start_time: datetime = Query(..., description="Start of time range (ISO 8601)"),
    end_time: datetime = Query(..., description="End of time range (ISO 8601)"),
    interval: str = Query("1h", regex="^(1s|1m|1h|4h)$", description="Aggregation interval"),
    model_id: str = Depends(verify_api_key)
):
    """Get historical tick data for backtesting."""
    start_time_query = time.time()

    logger.info(f"Historical query from model {model_id} for symbols {symbols}, interval {interval}")

    # Check permissions
    required_perms = {"symbols": symbols, "query_types": ["historical"]}
    if not check_permissions(model_id, required_perms):
        logger.warning(f"Permission denied for model {model_id}")
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    # Validate time range
    if end_time <= start_time:
        raise HTTPException(status_code=400, detail="end_time must be after start_time")

    # Get historical data
    data = enriched_service.get_historical_data(symbols, start_time, end_time, interval)

    response_time = int((time.time() - start_time_query) * 1000)

    logger.info(f"Historical query completed in {response_time}ms, returned {len(data)} data points")

    return {
        "data": [item.model_dump() for item in data],
        "metadata": {
            "query_id": "mock-uuid",
            "response_time_ms": response_time,
            "data_points_returned": len(data),
            "cache_hit": False  # Historical data typically from disk
        }
    }