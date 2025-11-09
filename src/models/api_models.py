"""
Pydantic models for GEX Data Pipeline API.

Defines request/response models for GEX payloads, historical data, and webhooks.
Adapted to match legacy database schema.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class GEXStrike(BaseModel):
    """Model for individual GEX strike data (legacy schema)."""

    strike: float = Field(..., description="Strike price")
    gamma_now: Optional[float] = Field(None, alias="gamma", description="Current gamma value")
    vanna: Optional[float] = Field(None, description="Vanna value")
    history: Optional[List[float]] = Field(None, description="History of gamma values")
    oi_gamma: Optional[float] = Field(None, description="OI gamma value")


class GEXPayload(BaseModel):
    """Model for GEX data payload received via /gex endpoint (legacy schema)."""

    timestamp: datetime = Field(..., description="Snapshot timestamp")
    ticker: str = Field(..., min_length=1, description="Financial instrument symbol")
    endpoint: Optional[str] = Field(None, description="Data endpoint")
    spot_price: float = Field(..., alias="spot", description="Current spot price")
    zero_gamma: float = Field(..., description="Zero gamma value")
    net_gex_vol: Optional[float] = Field(None, description="Net GEX volume")
    net_gex_oi: Optional[float] = Field(None, description="Net GEX open interest")
    net_gex: Optional[float] = Field(None, description="Net gamma exposure (combined)")
    min_dte: Optional[int] = Field(None, description="Minimum days to expiration")
    sec_min_dte: Optional[int] = Field(None, description="Secondary minimum days to expiration")
    major_pos_vol: Optional[float] = Field(None, description="Major positive volume strike")
    major_pos_oi: Optional[float] = Field(None, description="Major positive OI strike")
    major_neg_vol: Optional[float] = Field(None, description="Major negative volume strike")
    major_neg_oi: Optional[float] = Field(None, description="Major negative OI strike")
    sum_gex_vol: Optional[float] = Field(None, description="Sum of GEX volume")
    sum_gex_oi: Optional[float] = Field(None, description="Sum of GEX open interest")
    delta_risk_reversal: Optional[float] = Field(None, description="Delta risk reversal value")
    strikes: Optional[List[GEXStrike]] = Field(None, description="Individual strike data")
    max_change: Optional[Dict[str, float]] = Field(None, description="Maximum changes")
    max_priors: Optional[Union[str, List[float]]] = Field(None, description="Prior maximum values")

    class Config:
        """Pydantic config for backward compatibility."""
        populate_by_name = True  # Allow both alias and field name

    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v):
        """Validate ticker format."""
        if not v or not isinstance(v, str):
            raise ValueError("Ticker must be non-empty string")
        return v


class GEXHistoryRequest(BaseModel):
    """Model for historical data import request via /gex_history_url."""

    url: str = Field(..., description="Source URL for historical data download")
    ticker: str = Field(..., min_length=1, description="Associated ticker symbol")
    endpoint: str = Field(..., min_length=1, description="Data endpoint")

    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        """Validate URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        return v


class GEXHistoryResponse(BaseModel):
    """Model for historical data import response."""

    job_id: str = Field(..., description="Import job ID")
    status: str = Field(..., description="Import status")
    message: str = Field(..., description="Status message")


class WebhookPayload(BaseModel):
    """Model for universal webhook payload via /uw endpoint."""

    topic: str = Field(..., min_length=1, description="Webhook topic")
    event_type: Optional[str] = Field(None, description="Event type")
    payload: Dict[str, Any] = Field(..., description="Event-specific data")


class APIResponse(BaseModel):
    """Generic API response model."""

    status: str = Field(..., description="Response status")
    message: Optional[str] = Field(None, description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    timestamp: Optional[str] = Field(None, description="Response timestamp")