"""Tests for IVR computation service."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import pytest

from src.services.ivr_service import IVRService, IVRServiceSettings


@pytest.fixture
def ivr_service(tmp_path: Path):
    settings = IVRServiceSettings(
        db_path=tmp_path / "ivr.db",
        option_trades_db=tmp_path / "uw_messages.db",
    )
    return IVRService(config=settings, redis_client=MagicMock())


def _seed_option_trades(db_path: Path, rows):
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS option_trades (
            received_at TIMESTAMP,
            ticker VARCHAR,
            implied_volatility DOUBLE
        )
    """)
    for ticker, ts, iv in rows:
        conn.execute(
            "INSERT INTO option_trades VALUES (?, ?, ?)",
            [ts, ticker, iv],
        )
    conn.close()


def test_compute_ivr_returns_none_for_unknown_symbol(ivr_service):
    result = ivr_service.compute_ivr("UNKNOWN")
    assert result["ivr"] is None
    assert result["trade_days"] == 0


def test_compute_ivr_computes_rank(ivr_service):
    # 5 days of IV data: highs at 0.50, lows at 0.20, current at 0.35
    rows = [
        ("SPY", "2026-06-21 10:00:00", 0.35),  # current (most recent)
        ("SPY", "2026-06-20 10:00:00", 0.50),  # high
        ("SPY", "2026-06-19 10:00:00", 0.20),  # low
        ("SPY", "2026-06-18 10:00:00", 0.30),
        ("SPY", "2026-06-17 10:00:00", 0.40),
    ]
    _seed_option_trades(ivr_service.settings.option_trades_db, rows)
    ivr_service.aggregate_daily_iv()

    result = ivr_service.compute_ivr("SPY")

    assert result["ivr"] is not None
    assert result["trade_days"] == 5
    assert result["iv_252_high"] == pytest.approx(0.50, abs=0.01)
    assert result["iv_252_low"] == pytest.approx(0.20, abs=0.01)
    assert result["current_iv"] == pytest.approx(0.35, abs=0.01)
    # IVR = (0.35 - 0.20) / (0.50 - 0.20) * 100 = 50.0
    assert result["ivr"] == pytest.approx(50.0, abs=1.0)


def test_aggregate_daily_iv_returns_zero_without_data(ivr_service):
    # No option_trades table with IV data
    count = ivr_service.aggregate_daily_iv()
    assert count == 0


def test_compute_ivr_batch(ivr_service):
    rows = [
        ("SPY", "2026-06-21 10:00:00", 0.40),
        ("SPY", "2026-06-20 10:00:00", 0.30),
        ("QQQ", "2026-06-21 10:00:00", 0.50),
        ("QQQ", "2026-06-20 10:00:00", 0.25),
    ]
    _seed_option_trades(ivr_service.settings.option_trades_db, rows)
    ivr_service.aggregate_daily_iv()

    batch = ivr_service.compute_ivr_batch()
    assert "SPY" in batch
    assert "QQQ" in batch
    assert batch["SPY"]["ivr"] is not None
    assert batch["QQQ"]["ivr"] is not None
