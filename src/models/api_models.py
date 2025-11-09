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

    model_config = {"extra": "allow"}

    url: str = Field(..., description="Source URL for historical data download")
    ticker: Optional[str] = Field(None, description="Associated ticker symbol")
    endpoint: Optional[str] = Field(None, description="Data endpoint label (defaults to gex_zero)")
    feed: Optional[str] = Field(None, description="Legacy endpoint alias: feed")
    kind: Optional[str] = Field(None, description="Legacy endpoint alias: kind")

    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        """Validate URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        return v

    @field_validator('ticker', mode='after')
    @classmethod
    def normalize_ticker(cls, v, info):
        """Allow ticker to be inferred from the URL if omitted."""
        if v and v.strip():
            return v.strip()

        data = info.data
        url = data.get('url', '')
        if url:
            import re
            match = re.search(r'/(\d{4}-\d{2}-\d{2})_([^_]+)_classic', url)
            if match:
                inferred = match.group(2)
                return inferred

        raise ValueError("Ticker must be provided or parsable from URL")

    @field_validator('endpoint', mode='after')
    @classmethod
    def normalize_endpoint(cls, v, info):
        """Support legacy payloads that use feed/kind fields."""
        if v:
            return v
        data = info.data  # contains raw input data
        for legacy_field in ('feed', 'kind'):
            legacy_val = data.get(legacy_field)
            if isinstance(legacy_val, str) and legacy_val.strip():
                return legacy_val.strip()
        return "gex_zero"


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
