from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.services.trin_history_recorder import (
    TrinHistoryRecorder,
    TrinHistoryRecorderSettings,
)


@pytest.mark.asyncio
async def test_trin_history_recorder_persists_deduped_daily_files(tmp_path: Path):
    settings = TrinHistoryRecorderSettings(
        db_path=tmp_path / "trin_history.db",
        parquet_dir=tmp_path / "parquet" / "trin",
        flush_interval_seconds=60,
    )
    recorder = TrinHistoryRecorder(settings)

    payload = {
        "symbol": "$TRIN",
        "price": 0.91,
        "size": 1,
        "timestamp": "2026-06-12T14:00:00+00:00",
        "source": "tastytrade",
    }

    recorder.record_trade(payload)
    recorder.record_trade(payload)
    recorder.record_trade({"symbol": "MNQ", "price": 21000.0, "size": 1, "timestamp": payload["timestamp"]})

    written = await recorder.flush()
    assert written == 1

    con = duckdb.connect(str(settings.db_path))
    try:
        count = con.execute(
            "SELECT COUNT(*) FROM trin_trade_history WHERE symbol = ?",
            ["$TRIN"],
        ).fetchone()[0]
    finally:
        con.close()
    assert count == 1

    parquet_file = settings.parquet_dir / "$TRIN" / "20260612.parquet"
    assert parquet_file.exists()

    df = pd.read_parquet(parquet_file)
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "$TRIN"
    assert df.iloc[0]["price"] == pytest.approx(0.91)
