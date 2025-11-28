import os
from typing import Optional


class Config:
    # Database
    DATABASE_PATH: str = os.path.join(
        os.path.dirname(__file__), "../../data/tick_data.db"
    )

    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")

    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Data sources
    SIERRA_CHART_ENABLED: bool = (
        os.getenv("SIERRA_CHART_ENABLED", "false").lower() == "true"
    )
    GEXBOT_API_KEY: Optional[str] = os.getenv("GEXBOT_API_KEY")
    TASTYTRADE_CLIENT_ID: Optional[str] = os.getenv("TASTYTRADE_CLIENT_ID")
    TASTYTRADE_CLIENT_SECRET: Optional[str] = os.getenv("TASTYTRADE_CLIENT_SECRET")

    # Data processing
    MEMORY_RETENTION_HOURS: int = int(os.getenv("MEMORY_RETENTION_HOURS", "1"))
    COMPRESSION_INTERVAL_MINUTES: int = int(
        os.getenv("COMPRESSION_INTERVAL_MINUTES", "60")
    )
    GAP_CHECK_INTERVAL_HOURS: int = int(os.getenv("GAP_CHECK_INTERVAL_HOURS", "24"))

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-this-in-production")


config = Config()
