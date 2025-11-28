import uvicorn
from fastapi import FastAPI
from .config.settings import config
from .discord_bot.commands import bot
from .api.futures_orders import router as futures_router
from .api.options_orders import router as options_router
from .api.order_cancellation import router as cancellation_router
from .api.health import router as health_router
from .discord_bot.error_handler import setup_error_handling

# Create FastAPI app
app = FastAPI(title="Tastytrade Ordering API", version="1.0.0")

# Include routers
app.include_router(futures_router, prefix="/api", tags=["futures"])
app.include_router(options_router, prefix="/api", tags=["options"])
app.include_router(cancellation_router, prefix="/api", tags=["orders"])
app.include_router(health_router, prefix="/api", tags=["health"])

# Setup Discord bot error handling
setup_error_handling(bot)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    # Start Discord bot
    import asyncio

    asyncio.create_task(bot.start(config.discord_token))


@app.get("/")
async def root():
    return {"message": "Tastytrade Ordering API"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
