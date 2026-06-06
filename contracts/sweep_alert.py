"""Sweep alert payload on sweep:alert:{symbol} — keep byte-identical in both repos."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SweepAlertPayload(BaseModel):
    ts_ms: int
    symbol: str
    direction: Literal["up", "down"]
    trigger_price: float
    trigger_ticks: float
    classification: Literal["sweep", "directional"]
    confidence: float = Field(ge=0.0, le=1.0)
    danger_level: int = Field(ge=0, le=3)
    model_version: str
    features: dict[str, Any] = Field(default_factory=dict)