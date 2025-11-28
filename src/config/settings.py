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
    tastytrade_use_sandbox: bool = Field(True, alias="TASTYTRADE_USE_SANDBOX")
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
    def effective_tastytrade_refresh_token(self) -> str:
        return (
            self.tastytrade_prod_refresh_token
            if not self.tastytrade_use_sandbox
            else self.tastytrade_refresh_token
        )


# Global config instance
config = TastyTradeConfig()
