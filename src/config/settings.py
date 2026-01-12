from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))


class TastyTradeConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    # Discord settings
    discord_token: str = Field(alias="DISCORD_BOT_TOKEN")

    # Tastytrade settings
    tastytrade_client_secret: str = Field(alias="TASTYTRADE_CLIENT_SECRET")
    tastytrade_refresh_token: str = Field(alias="TASTYTRADE_REFRESH_TOKEN")
    tastytrade_account: Optional[str] = Field(None, alias="TASTYTRADE_ACCOUNT")
    tastytrade_use_sandbox: bool = Field(True, alias="TASTYTRADE_USE_SANDBOX")
    tastytrade_allocation_percentage: float = Field(10.0, alias="TASTYTRADE_ALLOCATION_PERCENTAGE")
    tastytrade_prod_client_secret: Optional[str] = Field(
        None, alias="TASTYTRADE_PROD_CLIENT_SECRET"
    )
    tastytrade_prod_refresh_token: Optional[str] = Field(
        None, alias="TASTYTRADE_PROD_REFRESH_TOKEN"
    )

    # Database settings
    redis_url: str = Field("redis://localhost:6379", alias="REDIS_URL")
    database_url: str = Field("data/gex_data.db", alias="DATABASE_URL")

    # Application settings
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    # Allowlist for automated alert trading (comma-separated user/channel ids)
    allowed_users: Optional[str] = Field(None, alias="ALLOWED_USERS")
    allowed_channels: Optional[str] = Field(None, alias="ALLOWED_CHANNELS")

    @field_validator("tastytrade_use_sandbox", mode="before")
    @classmethod
    def _coerce_bool(cls, value):
        """Tolerate loose/typo'd boolean values; default to True (safer sandbox)."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            val = value.strip().lower()
            truthy = {"true", "1", "t", "yes", "y", "on", "sandbox"}
            falsy = {"false", "0", "f", "no", "n", "off", "prod", "production", "live"}
            if val in truthy:
                return True
            if val in falsy:
                return False
            logging.warning(
                "Invalid TASTYTRADE_USE_SANDBOX value '%s'; defaulting to True", value
            )
            return True
        try:
            return bool(value)
        except Exception:
            logging.warning(
                "Invalid TASTYTRADE_USE_SANDBOX value '%s'; defaulting to True", value
            )
            return True

    @property
    def effective_tastytrade_client_secret(self) -> str:
        return (
            self.tastytrade_prod_client_secret
            if not self.tastytrade_use_sandbox
            else self.tastytrade_client_secret
        )

    @property
    def allowed_user_list(self) -> list[str]:
        if not self.allowed_users:
            # Fallback to legacy env variables used by discord-bot
            fallback = os.getenv("ALERT_USERS") or os.getenv("DISCORD_ML_TRADE_USER_ID")
            if fallback:
                return [u.strip() for u in fallback.split(",") if u.strip()]
            return []
        return [u.strip() for u in self.allowed_users.split(",") if u.strip()]

    @property
    def allowed_channel_list(self) -> list[str]:
        if not self.allowed_channels:
            # Fallback to legacy env variables used by discord-bot
            fallback = (
                os.getenv("DISCORD_ALLOWED_CHANNEL_IDS")
                or os.getenv("DISCORD_AUTOMATED_TRADE_IDS")
                or os.getenv("DISCORD_GEX_FEED_CHANNEL_IDS")
            )
            if fallback:
                return [c.strip() for c in fallback.split(",") if c.strip()]
            return []
        return [c.strip() for c in self.allowed_channels.split(",") if c.strip()]

    @property
    def effective_tastytrade_refresh_token(self) -> str:
        return (
            self.tastytrade_prod_refresh_token
            if not self.tastytrade_use_sandbox
            else self.tastytrade_refresh_token
        )


# Global config instance
config = TastyTradeConfig()
