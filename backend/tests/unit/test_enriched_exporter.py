import sys
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.enriched_exporter import EnrichedExporter  # noqa: E402


def test_strike_candidate_ranking():
    con = duckdb.connect()
    try:
        con.execute(
            """
            CREATE TEMP TABLE strikes (
                timestamp BIGINT,
                strike DOUBLE,
                gamma DOUBLE,
                oi_gamma DOUBLE
            );
            INSERT INTO strikes VALUES
            (1000, 24000.0,   5000.0, 10.0),   -- pos 1
            (1000, 24100.0,   3000.0,  9.0),   -- pos 2
            (1000, 24200.0,   2000.0,  8.0),   -- pos 3
            (1000, 23900.0,  -6000.0, 12.0),   -- neg 1
            (1000, 23800.0,  -4000.0, 11.0),   -- neg 2
            (1000, 23700.0,  -1000.0,  5.0);   -- neg 3
            """
        )
        EnrichedExporter._create_strike_candidates(con)
        row = con.execute("SELECT * FROM strike_candidates").fetchone()
    finally:
        con.close()

    (
        ts,
        major_pos_vol,
        major_pos_strike,
        major_pos_vol_2,
        major_pos_strike_2,
        major_pos_vol_3,
        major_pos_strike_3,
        major_neg_vol,
        major_neg_strike,
        major_neg_vol_2,
        major_neg_strike_2,
        major_neg_vol_3,
        major_neg_strike_3,
    ) = row

    assert ts == 1000
    # Positive ranks
    assert major_pos_strike == 24000.0
    assert pytest.approx(major_pos_vol) == 0.005  # millions proxy
    assert major_pos_strike_2 == 24100.0
    assert major_pos_vol_2 == pytest.approx(0.003)
    assert major_pos_strike_3 == 24200.0
    assert major_pos_vol_3 == pytest.approx(0.002)
    # Negative ranks
    assert major_neg_strike == 23900.0
    assert major_neg_vol == pytest.approx(-0.006)
    assert major_neg_strike_2 == 23800.0
    assert major_neg_vol_2 == pytest.approx(-0.004)
    assert major_neg_strike_3 == 23700.0
    assert major_neg_vol_3 == pytest.approx(-0.001)
