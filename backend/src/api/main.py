from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

app = FastAPI(
    title="Financial Tick Data Pipeline API",
    version="1.0.0",
    description="API for secure querying of financial tick data by AI models"
)

# Add rate limiting middleware
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

# Include routers
from .api import ticks, status

app.include_router(ticks.router, prefix="/api/v1", tags=["ticks"])
app.include_router(status.router, prefix="/api/v1", tags=["status"])

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Financial Tick Data Pipeline API"}

@app.get("/health")
async def health():
    logger.info("Health check performed")
    return {"status": "healthy"}

if __name__ == "__main__":
    logger.info("Starting Financial Tick Data Pipeline API")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )