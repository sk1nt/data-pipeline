"""
Configuration management for GEX Data Pipeline.

Handles environment variables, CLI arguments, and application settings.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(".env")


def env_field(*env_names: str, default: Any = None, **kwargs: Any):
    """Map environment variables to BaseSettings fields without deprecated env= usage."""
    if not env_names:
        raise ValueError("env_field requires at least one environment variable name")
    kwargs["validation_alias"] = AliasChoices(*env_names)
    return Field(default=default, **kwargs)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server settings
    host: str = env_field("HOST", default="0.0.0.0")
    port: int = env_field("PORT", default=8000)
    reload: bool = env_field("RELOAD", default=False)

    # Database settings
    data_dir: str = env_field("DATA_DIR", default="data")

    # Logging settings
    log_level: str = env_field("LOG_LEVEL", default="INFO")
    log_file: Optional[str] = env_field("LOG_FILE")

    # Rate limiting
    rate_limit_gex: str = env_field("RATE_LIMIT_GEX", default="100/minute")
    rate_limit_history: str = env_field("RATE_LIMIT_HISTORY", default="10/minute")
    rate_limit_webhook: str = env_field("RATE_LIMIT_WEBHOOK", default="100/minute")
    rate_limit_health: str = env_field("RATE_LIMIT_HEALTH", default="60/minute")

    # CORS settings
    cors_origins: str = env_field("CORS_ORIGINS", default="*")

    # Data pipeline settings
    max_import_attempts: int = env_field("MAX_IMPORT_ATTEMPTS", default=3)
    import_timeout: int = env_field("IMPORT_TIMEOUT", default=300)  # seconds
    staging_dir: str = env_field("STAGING_DIR", default="data/source/gexbot")
    parquet_dir: str = env_field("PARQUET_DIR", default="data/parquet/gexbot")
    redis_retention_ms: int = env_field("REDIS_RETENTION_MS", default=86_400_000)
    flush_interval_seconds: int = env_field("FLUSH_INTERVAL_SECONDS", default=600)
    flush_schedule_mode: str = env_field("FLUSH_SCHEDULE_MODE", default="daily")
    flush_daily_time: str = env_field("FLUSH_DAILY_TIME", default="00:30")
    redis_host: str = env_field("REDIS_HOST", default="localhost")
    redis_port: int = env_field("REDIS_PORT", default=6379)
    redis_db: int = env_field("REDIS_DB", default=0)
    redis_password: Optional[str] = env_field("REDIS_PASSWORD")
    timeseries_db_path: str = env_field("TIMESERIES_DB_PATH", default="data/redis_timeseries.db")
    timeseries_parquet_dir: str = env_field(
        "TIMESERIES_PARQUET_DIR",
        default="data/parquet/timeseries",
    )
    tick_db_path: str = env_field("TICK_DB_PATH", default="data/tick_data.db")
    depth_db_path: str = env_field("DEPTH_DB_PATH", default="data/tick_mbo_data.db")
    tick_parquet_dir: str = env_field("TICK_PARQUET_DIR", default="data/parquet/tick")
    depth_parquet_dir: str = env_field("DEPTH_PARQUET_DIR", default="data/parquet/depth")
    service_control_token: Optional[str] = env_field("SERVICE_CONTROL_TOKEN")

    # Discord bot control
    discord_bot_enabled: bool = env_field("DISCORD_BOT_ENABLED", default=False)

    # TastyTrade DXLink streamer
    tastytrade_stream_enabled: bool = env_field("TASTYTRADE_STREAM_ENABLED", default=False)
    tastytrade_symbols: str = env_field(
        "TASTYTRADE_STREAM_SYMBOLS",
        default="MES,MNQ,NQ,SPY,QQQ,VIX",
    )
    tastytrade_depth_levels: int = env_field("TASTYTRADE_DEPTH_LEVELS", default=40)
    tastytrade_client_id: Optional[str] = env_field("TASTYTRADE_CLIENT_ID")
    tastytrade_client_secret: Optional[str] = env_field("TASTYTRADE_CLIENT_SECRET")
    tastytrade_refresh_token: Optional[str] = env_field("TASTYTRADE_REFRESH_TOKEN")

    # GEXBot poller
    gex_polling_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("GEXBOT_POLLING_ENABLED"),
    )
    gex_poll_interval_seconds: float = env_field(
        "GEXBOT_POLL_INTERVAL_SECONDS",
        default=60.0,
    )
    gex_poll_rth_interval_seconds: float = env_field(
        "GEXBOT_POLL_RTH_INTERVAL_SECONDS",
        default=1.0,
    )
    gex_poll_off_hours_interval_seconds: float = env_field(
        "GEXBOT_POLL_OFF_HOURS_INTERVAL_SECONDS",
        default=300.0,
    )
    gex_poll_dynamic_schedule: bool = env_field(
        "GEXBOT_POLL_DYNAMIC_SCHEDULE",
        default=True,
    )
    gex_poll_symbols: str = Field(
        default="ES_SPX,SPY,QQQ,NDX",
        validation_alias=AliasChoices("GEXBOT_POLL_SYMBOLS"),
    )
    gex_poll_aggregation: str = Field(
        default="zero",
        validation_alias=AliasChoices("GEXBOT_POLL_AGGREGATION"),
    )
    gexbot_api_key: Optional[str] = env_field("GEXBOT_API_KEY")
    gex_nq_polling_enabled: bool = Field(
        default=False,
        # Accept both legacy and canonical env var names; pydantic's validation alias
        # ensures truthy values are coerced correctly from env strings like 'true'.
        validation_alias=AliasChoices("GEXBOT_NQ_POLLING_ENABLED"),
    )
    gex_nq_poll_symbols: str = env_field(
        "GEXBOT_NQ_POLL_SYMBOLS",
        default="NQ_NDX,SPX",
    )
    gex_nq_poll_interval_seconds: float = env_field(
        "GEXBOT_NQ_POLL_INTERVAL_SECONDS",
        default=60.0,
    )
    gex_nq_poll_rth_interval_seconds: float = env_field(
        "GEXBOT_NQ_POLL_RTH_INTERVAL_SECONDS",
        default=1.0,
    )
    gex_nq_poll_off_hours_interval_seconds: float = env_field(
        "GEXBOT_NQ_POLL_OFF_HOURS_INTERVAL_SECONDS",
        default=300.0,
    )
    gex_nq_poll_dynamic_schedule: bool = env_field(
        "GEXBOT_NQ_POLL_DYNAMIC_SCHEDULE",
        default=True,
    )
    gex_nq_poll_aggregation: str = env_field(
        "GEXBOT_NQ_POLL_AGGREGATION",
        default="zero",
    )
    gexbot_nq_poll_symbols: str = env_field(
        "GEXBOT_NQ_POLL_SYMBOLS",
        default="NQ_NDX",
    )

    # Sierra Chart bridge (optional)
    sierra_chart_output_path: Optional[str] = env_field(
        "SIERRA_CHART_OUTPUT_PATH",
        default="/mnt/c/SierraChart/Data/gex_data.json",
    )

    # Schwab streaming
    schwab_enabled: bool = env_field("SCHWAB_ENABLED", default=False)
    schwab_stream_paused: bool = env_field("SCHWAB_STREAM_PAUSED", default=True)
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
    schwab_account_id: Optional[str] = env_field("SCHWAB_ACCOUNT_ID")
    schwab_rest_url: str = env_field("SCHWAB_REST_URL", default="https://api.schwab.com/v1")
    schwab_auth_url: str = env_field(
        "SCHWAB_AUTH_URL",
        default="https://api.schwab.com/oauth2/v1/authorize",
    )
    schwab_token_url: str = env_field(
        "SCHWAB_TOKEN_URL",
        default="https://api.schwab.com/v1/oauth/token",
    )
    schwab_token_auth_method: str = env_field(
        "SCHWAB_TOKEN_AUTH_METHOD",
        default="form",
    )
    schwab_stream_url: str = env_field("SCHWAB_STREAM_URL", default="wss://stream.schwab.com/v1")
    schwab_symbols: str = env_field("SCHWAB_SYMBOLS", default="MNQ,MES,SPY,QQQ,VIX")
    schwab_tick_channel: str = env_field("SCHWAB_TICK_CHANNEL", default="market_data:ticks")
    schwab_level2_channel: str = env_field("SCHWAB_LEVEL2_CHANNEL", default="market_data:level2")
    schwab_heartbeat_seconds: int = env_field("SCHWAB_HEARTBEAT_SECONDS", default=15)
    schwab_redirect_uri: Optional[str] = env_field("SCHWAB_REDIRECT_URI")
    schwab_scope: str = env_field(
        "SCHWAB_SCOPE",
        default="readonly",
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
    def tastytrade_depth_cap(self) -> int:
        """Return the enforced depth level limit; temporarily capped at 5."""
        return max(1, min(self.tastytrade_depth_levels, 5))

    @property
    def gex_symbol_list(self) -> list[str]:
        return [symbol.strip().upper() for symbol in self.gex_poll_symbols.split(",") if symbol.strip()]

    @property
    def gex_nq_poll_symbol_list(self) -> list[str]:
        return [symbol.strip().upper() for symbol in self.gex_nq_poll_symbols.split(",") if symbol.strip()]

    @property
    def gexbot_nq_poll_symbol_list(self) -> list[str]:
        return [symbol.strip().upper() for symbol in self.gexbot_nq_poll_symbols.split(",") if symbol.strip()]

    @property
    def redis_params(self) -> Dict[str, Any]:
        return {
            "host": self.redis_host,
            "port": self.redis_port,
            "db": self.redis_db,
            "password": self.redis_password,
        }

    def ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            self.data_path,
            self.staging_path,
            self.parquet_path,
            Path(self.timeseries_parquet_dir),
            Path(self.tick_parquet_dir),
            Path(self.depth_parquet_dir),
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_directories()
