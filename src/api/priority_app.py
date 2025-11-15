"""
FastAPI application for the GEX priority system.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from ..lib.logging import setup_logging, get_logger
from ..lib.priority_db import get_priority_db_manager
from ..lib.redis_client import get_redis_client

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    logger.info("Starting GEX Priority API")

    # Initialize database
    try:
        db_manager = get_priority_db_manager()
        db_manager.initialize_schema()
        logger.info("Database schema initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    # Check Redis connectivity
    try:
        redis_client = get_redis_client()
        if redis_client.health_check():
            logger.info("Redis connection established")
        else:
            logger.warning("Redis connection failed - caching will be disabled")
    except Exception as e:
        logger.warning(f"Redis health check failed: {e} - caching will be disabled")

    yield

    logger.info("Shutting down GEX Priority API")


# Create FastAPI application
app = FastAPI(
    title="GEX Data Ingest Priority API",
    description="API for managing high-speed GEX data ingestion with priority-based processing",
    version="1.0.0",
    lifespan=lifespan,
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware (configure for production)
if os.getenv("ENVIRONMENT") == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["your-domain.com"],  # Configure appropriately
    )


# Custom middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    logger.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)},
    )


# Health check endpoint
@app.get("/health")
@limiter.limit("10/minute")
async def health_check(request: Request):
    """
    Health check endpoint.

    Returns basic service health status.
    """
    health_status = {
        "status": "healthy",
        "service": "gex-priority-api",
        "version": "1.0.0",
    }

    # Check database health
    try:
        db_manager = get_priority_db_manager()
        if db_manager.health_check():
            health_status["database"] = "healthy"
        else:
            health_status["database"] = "unhealthy"
            health_status["status"] = "degraded"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["database"] = "unhealthy"
        health_status["status"] = "degraded"

    # Check Redis health
    try:
        redis_client = get_redis_client()
        if redis_client.health_check():
            health_status["redis"] = "healthy"
        else:
            health_status["redis"] = "unhealthy"
            # Redis failure doesn't mark overall status as unhealthy
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        health_status["redis"] = "unhealthy"

    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(status_code=status_code, content=health_status)


# Root endpoint
@app.get("/")
@limiter.limit("60/minute")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "GEX Data Ingest Priority API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# Import and include route modules
# These will be created in subsequent tasks
try:
    from .routes.priority_routes import router as priority_router

    app.include_router(
        priority_router,
        tags=["priority"],
    )

    logger.info("Priority API routes loaded successfully")

except ImportError as e:
    logger.warning(f"Priority route modules not available yet: {e}")
    logger.info("API will start with limited functionality")

try:
    from .routes.rules import router as rules_router

    app.include_router(
        rules_router,
        tags=["rules"],
    )

    logger.info("Rules API routes loaded successfully")

except ImportError as e:
    logger.warning(f"Rules route modules not available yet: {e}")
    logger.info("API will start with limited functionality")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
        log_level="info",
    )