"""
Configuration management for GEX Data Pipeline.

Handles environment variables, CLI arguments, and application settings.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    redis_retention_ms: int = Field(default=86_400_000, env="REDIS_RETENTION_MS")
    flush_interval_seconds: int = Field(default=600, env="FLUSH_INTERVAL_SECONDS")

    # TastyTrade DXLink streamer
    tastytrade_stream_enabled: bool = Field(default=False, env="TASTYTRADE_STREAM_ENABLED")
    tastytrade_symbols: str = Field(default="MES,MNQ,NQ,SPY,QQQ,VIX", env="TASTYTRADE_STREAM_SYMBOLS")
    tastytrade_depth_levels: int = Field(default=40, env="TASTYTRADE_DEPTH_LEVELS")
    tastytrade_client_id: Optional[str] = Field(default=None, env="TASTYTRADE_CLIENT_ID")
    tastytrade_client_secret: Optional[str] = Field(default=None, env="TASTYTRADE_CLIENT_SECRET")
    tastytrade_refresh_token: Optional[str] = Field(default=None, env="TASTYTRADE_REFRESH_TOKEN")

    # GEXBot poller
    gex_polling_enabled: bool = Field(default=False, env="GEXBOT_POLLING_ENABLED")
    gex_poll_interval_seconds: int = Field(default=60, env="GEXBOT_POLL_INTERVAL_SECONDS")
    gex_poll_symbols: str = Field(default="NQ_NDX,ES_SPX,SPY,QQQ,SPX,NDX", env="GEXBOT_POLL_SYMBOLS")
    gexbot_api_key: Optional[str] = Field(default=None, env="GEXBOT_API_KEY")

    # Schwab streaming
    schwab_enabled: bool = Field(default=False, env="SCHWAB_ENABLED")
    schwab_client_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("SCHWAB_CLIENT_ID", "SCHWAB_APPKEY"),
    )
    schwab_client_secret: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("SCHWAB_CLIENT_SECRET", "SCHWAB_SECRET", "SCHWAB_SECRECT"),
    )
    schwab_refresh_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("SCHWAB_REFRESH_TOKEN", "SCHWAB_RTOKEN"),
    )
    schwab_account_id: Optional[str] = Field(default=None, env="SCHWAB_ACCOUNT_ID")
    schwab_rest_url: str = Field(default="https://api.schwab.com/v1", env="SCHWAB_REST_URL")
    schwab_auth_url: str = Field(
        default="https://api.schwab.com/oauth2/v1/authorize",
        env="SCHWAB_AUTH_URL",
    )
    schwab_token_url: str = Field(
        default="https://api.schwab.com/v1/oauth/token",
        env="SCHWAB_TOKEN_URL",
    )
    schwab_stream_url: str = Field(default="wss://stream.schwab.com/v1", env="SCHWAB_STREAM_URL")
    schwab_symbols: str = Field(default="MNQ,MES,SPY,QQQ,VIX", env="SCHWAB_SYMBOLS")
    schwab_tick_channel: str = Field(default="market_data:ticks", env="SCHWAB_TICK_CHANNEL")
    schwab_level2_channel: str = Field(default="market_data:level2", env="SCHWAB_LEVEL2_CHANNEL")
    schwab_heartbeat_seconds: int = Field(default=15, env="SCHWAB_HEARTBEAT_SECONDS")
    schwab_redirect_uri: Optional[str] = Field(default=None, env="SCHWAB_REDIRECT_URI")
    schwab_scope: str = Field(
        default="readonly",
        env="SCHWAB_SCOPE",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="allow",
    )

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
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def schwab_symbol_list(self) -> list[str]:
        """Return Schwab symbol list."""
        return [symbol.strip().upper() for symbol in self.schwab_symbols.split(",") if symbol.strip()]

    @property
    def tastytrade_symbol_list(self) -> list[str]:
        return [symbol.strip().upper() for symbol in self.tastytrade_symbols.split(",") if symbol.strip()]

    @property
    def gex_symbol_list(self) -> list[str]:
        return [symbol.strip().upper() for symbol in self.gex_poll_symbols.split(",") if symbol.strip()]

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
