"""
GEX Data Pipeline Server

FastAPI implementation of the migrated data-pipeline.py functionality.
Provides endpoints for GEX data capture, historical imports, and webhook handling.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler
import uvicorn

from .lib.gex_database import gex_db
from .lib.logging_config import setup_logging
from .models.api_models import GEXPayload, GEXHistoryRequest, GEXHistoryResponse, APIResponse

# Setup logging
logger = setup_logging()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    logger.info("Starting GEX Data Pipeline Server")

    # Startup: Initialize database connections
    try:
        with gex_db.gex_data_connection():
            logger.info("GEX data database connection established")
        with gex_db.gex_history_connection():
            logger.info("GEX history database connection established")
    except Exception as e:
        logger.error(f"Failed to initialize database connections: {e}")
        raise

    yield

    # Shutdown: Close connections
    logger.info("Shutting down GEX Data Pipeline Server")
    gex_db.close_all_connections()

# Create FastAPI app
app = FastAPI(
    title="GEX Data Pipeline",
    description="Financial data pipeline for GEX capture and processing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Global exception handler
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    logger.warning(f"Validation error for {request.url}: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "type": "validation_error"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception for {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "server_error"}
    )

# Health check endpoint
@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint."""
    return {"status": "healthy", "service": "gex-data-pipeline"}

# GEX data capture endpoint
@app.post("/gex", response_model=APIResponse)
async def gex_endpoint(payload: GEXPayload, request: Request):
    """GEX data capture endpoint."""
    try:
        logger.info(f"Received GEX payload for {payload.ticker} at {payload.timestamp}")

        # Store GEX snapshot in legacy gex_snapshots table
        with gex_db.gex_data_connection() as conn:
            # Handle net_gex - use net_gex_vol if available, otherwise net_gex
            net_gex_value = payload.net_gex_vol if payload.net_gex_vol is not None else payload.net_gex
            
            # Handle max_priors - convert list to JSON string if needed
            max_priors_value = payload.max_priors
            if isinstance(max_priors_value, list):
                import json
                max_priors_value = json.dumps(max_priors_value)
            
            conn.execute("""
                INSERT INTO gex_snapshots (
                    timestamp, ticker, spot_price, zero_gamma, net_gex,
                    min_dte, sec_min_dte, major_pos_vol, major_pos_oi,
                    major_neg_vol, major_neg_oi, sum_gex_vol, sum_gex_oi,
                    delta_risk_reversal, max_priors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                payload.timestamp, payload.ticker, payload.spot_price,
                payload.zero_gamma, net_gex_value, payload.min_dte,
                payload.sec_min_dte, payload.major_pos_vol, payload.major_pos_oi,
                payload.major_neg_vol, payload.major_neg_oi, payload.sum_gex_vol,
                payload.sum_gex_oi, payload.delta_risk_reversal, max_priors_value
            ])

        logger.info(f"Successfully stored GEX data for {payload.ticker}")
        return APIResponse(status="success", message="GEX data stored successfully")

    except Exception as e:
        logger.error(f"Failed to store GEX data: {e}")
        raise HTTPException(status_code=500, detail="Failed to store GEX data")

# Placeholder for historical data import endpoint
@app.post("/gex_history_url", response_model=GEXHistoryResponse)
@limiter.limit("100/minute")
async def gex_history_url_endpoint(payload: GEXHistoryRequest, request: Request):
    """Historical GEX data import endpoint."""
    try:
        logger.info(f"Queuing historical import for {payload.ticker} from {payload.url}")

        # Use the new safe import system instead of queue
        from .import_gex_history_safe import download_to_staging, safe_import
        import asyncio

        # Download to staging in a thread
        loop = asyncio.get_event_loop()
        staging_path = await loop.run_in_executor(None, download_to_staging, payload.url, payload.ticker, payload.endpoint)

        # Run safe import in a thread
        result = await loop.run_in_executor(None, safe_import, staging_path)

        logger.info(f"Completed historical import for {payload.ticker}: {result}")
        return GEXHistoryResponse(
            job_id=result.get("job_id", "unknown"),
            status="completed",
            message=f"Historical import completed: {result.get('records', 0)} records"
        )

    except Exception as e:
        logger.error(f"Failed to import historical data: {e}")
        raise HTTPException(status_code=500, detail="Failed to import historical data")

# Placeholder for universal webhook endpoint
@app.post("/uw")
@limiter.limit("100/minute")
async def universal_webhook_endpoint(request: Request):
    """Universal webhook endpoint."""
    # TODO: Implement webhook payload processing
    return {"status": "not_implemented"}

if __name__ == "__main__":
    uvicorn.run(
        "data_pipeline:app",
        host="0.0.0.0",
        port=8877,
        reload=True,
        log_level="info"
    )