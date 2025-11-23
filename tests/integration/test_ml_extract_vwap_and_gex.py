"""
Integration tests for ml.extract: ensure VWAP and derived GEX fields persist
for both --gex-json and --gex-db flows, and for time/volume/dollar bars.
"""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from pathlib import Path
import json
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime

from ml.extract import extract_1s_bars


def _write_tick_parquet(tmp_root: Path, symbol: str, date: datetime, ticks: pd.DataFrame):
    p = tmp_root / 'tick' / symbol
    p.mkdir(parents=True, exist_ok=True)
    file_date = date.strftime('%Y%m%d')
    parquet_path = p / f"{file_date}.parquet"
    ticks.to_parquet(parquet_path, index=False)
    return parquet_path


def test_extract_with_gex_json_and_vwap(tmp_path):
    # Build tick data for a single second
    symbol = 'MNQ'
    dt = datetime(2025, 11, 11, 10, 15, 0)
    # 3 ticks in same second
    ticks = pd.DataFrame({
        'timestamp': [dt, dt, dt],
        'price': [100.0, 102.0, 98.0],
        'volume': [1, 2, 3]
    })
    # expected vwap = (100*1 + 102*2 + 98*3) / (1+2+3) = (100 + 204 + 294)/6 = 598/6 = 99.666...
    expected_vwap = (100*1 + 102*2 + 98*3) / 6

    parquet_root = tmp_path
    parquet_path = _write_tick_parquet(parquet_root, symbol, dt, ticks)

    # Create a gex json file aligned with the bar second
    snapshot = {
        'ticker': 'NQ_NDX',
        'timestamp_ms': int(pd.Timestamp(dt).timestamp() * 1000),
        'spot_price': 25568.31,
        'zero_gamma': 25608.61,
        'major_pos_vol': 102.0,
        'major_neg_vol': 100.0,
        'max_priors': [[100, -50], [101, -100], [102, -300]],
        'strikes': [[100, 10, 1.5], [102, 12, 3.2]]
    }
    json_path = tmp_path / 'gex.json'
    json_path.write_text(json.dumps([snapshot]))

    # Redirect output
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    # monkeypatch the OUT_DIR in ml.extract module
    import ml.extract as extract_mod
    extract_mod.OUT_DIR = out_dir

    out_path = extract_1s_bars(symbol, dt.strftime('%Y-%m-%d'), tick_parquet_root=str(parquet_path.parent.parent), gex_db=None, gex_json=str(json_path), gex_ticker='NQ_NDX', bar_type='time', bar_size=1)
    # Read written parquet and assert vwap and derived fields exist
    df = pd.read_parquet(out_path)
    assert 'vwap' in df.columns
    assert np.isclose(df['vwap'].iloc[0], expected_vwap)
    # Derived columns from max_priors mapping
    assert 'max_current' in df.columns
    assert 'max_1m' in df.columns
    # Since max_priors = [[100,-50],[101,-100],[102,-300]], reversed mapping -> max_current = -300
    assert df['max_current'].iloc[0] == -300
    assert df['max_1m'].iloc[0] == -100
    # top_prior and top_strike fields removed
    assert 'top_prior_pos' not in df.columns
    assert 'top_prior_neg' not in df.columns
    assert 'top_strike_gamma' not in df.columns
    assert 'top_strike_oi' not in df.columns
    # ticker column present and equals symbol; gex_ticker present
    assert 'ticker' in df.columns
    assert df['ticker'].iloc[0] == symbol
    assert 'gex_ticker' in df.columns
    assert df['gex_ticker'].iloc[0] == 'NQ_NDX'
    # candidate fields: major_pos_candidates, major_neg_candidates (json strings)
    assert 'major_pos_candidates' in df.columns
    assert 'major_neg_candidates' in df.columns
    # numeric columns for top 3 candidates (short names)
    assert 'major_pos_can1' in df.columns
    assert 'major_neg_can1' in df.columns
    # top pos candidate should be 102 (exact match)
    assert df['major_pos_can1'].iloc[0] == 102.0
    assert df['major_neg_can1'].iloc[0] == 100.0


def test_extract_with_gex_db_and_vwap(tmp_path):
    symbol = 'MNQ'
    dt = datetime(2025, 11, 11, 10, 15, 0)
    ticks = pd.DataFrame({
        'timestamp': [dt, dt, dt],
        'price': [100.0, 102.0, 98.0],
        'volume': [1, 2, 3]
    })
    parquet_root = tmp_path
    parquet_path = _write_tick_parquet(parquet_root, symbol, dt, ticks)
    # Create a duckdb gex database with a single snapshot row
    db_path = tmp_path / 'gex_data.db'
    con = duckdb.connect(str(db_path))
    # Create minimal gex_snapshots table
    con.execute("""
    CREATE TABLE IF NOT EXISTS gex_snapshots (
        id INTEGER PRIMARY KEY,
        timestamp BIGINT,
        ticker VARCHAR,
        spot_price DOUBLE,
        zero_gamma DOUBLE,
        major_pos_vol DOUBLE,
        major_neg_vol DOUBLE,
        max_priors VARCHAR,
        strikes VARCHAR
    )
    """)
    # insert single row
    timestamp_ms = int(pd.Timestamp(dt).timestamp() * 1000)
    con.execute("INSERT INTO gex_snapshots (id, timestamp, ticker, spot_price, zero_gamma, major_pos_vol, major_neg_vol, max_priors, strikes) VALUES (?,?,?,?,?,?,?,?,?)",
                [1, timestamp_ms, 'NQ_NDX', 25568.31, 25608.61, 102.0, 100.0, json.dumps([[100, -50], [101, -100], [102, -300]]), json.dumps([[100, 10, 1.5], [102, 12, 3.2]])])
    con.close()

    # Redirect output
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    import ml.extract as extract_mod
    extract_mod.OUT_DIR = out_dir

    out_path = extract_1s_bars(symbol, dt.strftime('%Y-%m-%d'), tick_parquet_root=str(parquet_path.parent.parent), gex_db=str(db_path), gex_ticker='NQ_NDX', gex_json=None, bar_type='time', bar_size=1)
    df = pd.read_parquet(out_path)
    assert 'vwap' in df.columns
    assert 'max_current' in df.columns
    # values should match the DB snapshot inserted
    assert df['max_current'].iloc[0] == -300
    assert 'top_strike_gamma' not in df.columns
    assert 'ticker' in df.columns
    assert df['ticker'].iloc[0] == symbol
    assert 'gex_ticker' in df.columns
    assert df['gex_ticker'].iloc[0] == 'NQ_NDX'
    assert 'major_pos_candidates' in df.columns
    assert df['major_pos_can1'].iloc[0] == 102.0
