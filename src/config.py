"""
Configuration management for GEX Data Pipeline.

Handles environment variables, CLI arguments, and application settings.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    reload: bool = Field(default=False, env="RELOAD")

    # Database settings
    data_dir: str = Field(default="data", env="DATA_DIR")

    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")

    # Rate limiting
    rate_limit_gex: str = Field(default="100/minute", env="RATE_LIMIT_GEX")
    rate_limit_history: str = Field(default="10/minute", env="RATE_LIMIT_HISTORY")
    rate_limit_webhook: str = Field(default="100/minute", env="RATE_LIMIT_WEBHOOK")
    rate_limit_health: str = Field(default="60/minute", env="RATE_LIMIT_HEALTH")

    # CORS settings
    cors_origins: str = Field(default="*", env="CORS_ORIGINS")

    # Data pipeline settings
    max_import_attempts: int = Field(default=3, env="MAX_IMPORT_ATTEMPTS")
    import_timeout: int = Field(default=300, env="IMPORT_TIMEOUT")  # seconds
    staging_dir: str = Field(default="data/source/gexbot", env="STAGING_DIR")
    parquet_dir: str = Field(default="data/parquet/gex", env="PARQUET_DIR")

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False

    @property
    def data_path(self) -> Path:
        """Get data directory path."""
        return Path(self.data_dir)

    @property
    def staging_path(self) -> Path:
        """Get staging directory path."""
        return Path(self.staging_dir)

    @property
    def parquet_path(self) -> Path:
        """Get parquet directory path."""
        return Path(self.parquet_dir)

    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]

    def ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            self.data_path,
            self.staging_path,
            self.parquet_path,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_directories()