"""
GEX Data Pipeline Server

FastAPI implementation of the migrated data-pipeline.py functionality.
Provides endpoints for GEX data capture, historical imports, and webhook handling.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to sys.path for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler
import uvicorn

from .lib.gex_database import gex_db
from .lib.logging_config import setup_logging
from .models.api_models import GEXHistoryResponse, APIResponse
from .lib.gex_history_queue import gex_history_queue
from src.import_gex_history import process_historical_imports
from fastapi import BackgroundTasks

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
def _json_safe(value):
    """Convert objects (like bytes) into JSON-safe structures."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return repr(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    return value


async def _parse_json_body(request: Request) -> dict:
    """Return a dict from any JSON payload, even if wrapped in a string."""
    try:
        data = await request.json()
    except Exception as exc:  # pragma: no cover - FastAPI logs details
        logger.warning("Invalid JSON payload for %s: %s", request.url, exc)
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="JSON string payload must contain an object")

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")

    return data


def _normalize_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle FastAPI request validation errors and log payload."""
    body = exc.body if hasattr(exc, "body") else None
    body = _json_safe(body)
    safe_errors = _json_safe(exc.errors())
    logger.warning("Request validation error for %s: %s", request.url, safe_errors)
    if body is not None:
        logger.warning("Invalid request body: %s", body)
    return JSONResponse(
        status_code=422,
        content={"detail": safe_errors, "type": "validation_error", "body": body},
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors raised manually."""
    logger.warning(f"Pydantic validation error for {request.url}: {exc}")
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
async def gex_endpoint(request: Request):
    """GEX data capture endpoint with manual JSON parsing."""
    payload = await _parse_json_body(request)

    ticker = _normalize_string(payload.get("ticker"))
    if not ticker:
        raise HTTPException(status_code=400, detail="Missing ticker")

    timestamp = payload.get("timestamp") or payload.get("received_at")
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            logger.warning("Invalid timestamp format for %s: %s", ticker, timestamp)
            raise HTTPException(status_code=400, detail="Invalid timestamp")

    spot_price = payload.get("spot_price", payload.get("spot"))
    zero_gamma = payload.get("zero_gamma")
    net_gex_value = payload.get("net_gex_vol")
    if net_gex_value is None:
        net_gex_value = payload.get("net_gex")

    if spot_price is None or zero_gamma is None or net_gex_value is None or timestamp is None:
        raise HTTPException(status_code=400, detail="Missing required GEX fields")

    max_priors_value = payload.get("max_priors")
    if isinstance(max_priors_value, list):
        max_priors_value = json.dumps(max_priors_value)

    try:
        with gex_db.gex_data_connection() as conn:
            conn.execute(
                """
                INSERT INTO gex_snapshots (
                    timestamp, ticker, spot_price, zero_gamma, net_gex,
                    min_dte, sec_min_dte, major_pos_vol, major_pos_oi,
                    major_neg_vol, major_neg_oi, sum_gex_vol, sum_gex_oi,
                    delta_risk_reversal, max_priors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    timestamp,
                    ticker,
                    spot_price,
                    zero_gamma,
                    net_gex_value,
                    payload.get("min_dte"),
                    payload.get("sec_min_dte"),
                    payload.get("major_pos_vol"),
                    payload.get("major_pos_oi"),
                    payload.get("major_neg_vol"),
                    payload.get("major_neg_oi"),
                    payload.get("sum_gex_vol"),
                    payload.get("sum_gex_oi"),
                    payload.get("delta_risk_reversal"),
                    max_priors_value,
                ],
            )
    except Exception as exc:
        logger.error("Failed to store GEX data for %s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail="Failed to store GEX data")

    logger.info("Stored GEX data for %s", ticker)
    return APIResponse(status="success", message="GEX data stored successfully")

# Historical data import endpoint (queue only)
@app.post("/gex_history_url", response_model=GEXHistoryResponse)
@limiter.limit("100/minute")
async def gex_history_url_endpoint(request: Request, background_tasks: BackgroundTasks):
    """Queue historical GEX imports without downloading immediately."""
    payload = await _parse_json_body(request)

    url = _normalize_string(payload.get("url"))
    ticker = _normalize_string(payload.get("ticker"))
    endpoint = _normalize_string(payload.get("endpoint") or payload.get("feed") or payload.get("kind"))

    if not ticker:
        ticker = _maybe_extract_ticker(url)

    if not url or not ticker:
        raise HTTPException(status_code=400, detail="Missing url or ticker")

    if not endpoint:
        endpoint = "gex_zero"

    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"url", "ticker", "endpoint", "feed", "kind"}
    }

    try:
        queue_id = gex_history_queue.enqueue_request(
            url=url,
            ticker=ticker,
            endpoint=endpoint,
            payload=metadata,
        )
    except Exception as exc:
        logger.error("Failed to queue historical import: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to queue historical import")

    logger.info("Queued history import url=%s ticker=%s endpoint=%s id=%s", url, ticker, endpoint, queue_id)
    # Start background processing immediately so POST triggers download/import.
    try:
        background_tasks.add_task(process_historical_imports)
    except Exception:
        logger.exception("Failed to start background import task")

    return GEXHistoryResponse(
        job_id=str(queue_id),
        status="queued",
        message="Historical import request accepted and will be processed",
    )


def _maybe_extract_ticker(url: str) -> str:
    if not url:
        return ""
    import re

    match = re.search(r"/\d{4}-\d{2}-\d{2}_([^_]+)_classic", url)
    if match:
        return match.group(1)
    return ""

# Placeholder for universal webhook endpoint
@app.post("/uw")
@limiter.limit("100/minute")
async def universal_webhook_endpoint(request: Request):
    """Universal webhook endpoint."""
    # TODO: Implement webhook payload processing
    return {"status": "not_implemented"}

if __name__ == "__main__":
    uvicorn.run(
        "src.data_pipeline:app",
        host="0.0.0.0",
        port=8877,
        reload=True,
        log_level="info"
    )
