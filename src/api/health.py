from fastapi import APIRouter
from ..services.tastytrade_client import tastytrade_client
from ..lib.redis_client import redis_client
from ..lib.duckdb_client import duckdb_client

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy"}


@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check of all services."""
    health = {"status": "healthy", "services": {}}

    # Check Tastytrade
    try:
        tastytrade_client.get_session()
        health["services"]["tastytrade"] = "healthy"
    except Exception as e:
        health["services"]["tastytrade"] = f"unhealthy: {e}"
        health["status"] = "unhealthy"

    # Check Redis
    try:
        redis_client.get("health_check")
        health["services"]["redis"] = "healthy"
    except Exception as e:
        health["services"]["redis"] = f"unhealthy: {e}"
        health["status"] = "unhealthy"

    # Check DuckDB
    try:
        duckdb_client.execute("SELECT 1")
        health["services"]["duckdb"] = "healthy"
    except Exception as e:
        health["services"]["duckdb"] = f"unhealthy: {e}"
        health["status"] = "unhealthy"

    return health
