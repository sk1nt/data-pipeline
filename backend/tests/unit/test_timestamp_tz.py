import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import duckdb
import polars as pl


def test_convert_parquet_to_ts_ms_with_tz(tmp_path):
    p = tmp_path / "test.parquet"
    # Create a small DataFrame with naive timestamps representing America/New_York midnight
    dt = datetime(2025, 11, 11, 5, 0, 0)
    df = pl.DataFrame({"timestamp": [dt, dt.replace(second=1)]})
    df.write_parquet(p)

    # Run the conversion script with timestamp-tz America/New_York
    out = tmp_path / "out.parquet"
    cmd = [
        sys.executable,
        "scripts/convert_parquet_to_ts_ms.py",
        str(p),
        "--out",
        str(out),
        "--atomic",
        "--timestamp-tz",
        "America/New_York",
    ]
    proc = subprocess.Popen(cmd)
    rc = proc.wait()
    assert rc == 0

    con = duckdb.connect()
    # Verify ts_ms computed by DuckDB matches a localized epoch
    rows = con.execute(
        f"SELECT ts_ms, CAST(EXTRACT(epoch FROM timestamp AT TIME ZONE 'America/New_York') * 1000 AS BIGINT) FROM read_parquet('{out}')"
    ).fetchall()
    for ts_ms, expected in rows:
        assert ts_ms == expected


def test_verify_timestamps_script_no_mismatch(tmp_path):
    # Re-use previous test setup
    p = tmp_path / "test2.parquet"
    dt = datetime(2025, 11, 11, 5, 0, 0)
    dt_utc = dt.replace(tzinfo=ZoneInfo("UTC"))
    # Use timezone-aware timestamp so DuckDB's AT TIME ZONE handling is unambiguous
    df = pl.DataFrame(
        {"timestamp": [dt_utc], "ts_ms": [int(dt_utc.timestamp() * 1000)]}
    )
    df.write_parquet(p)

    cmd = [
        sys.executable,
        "scripts/verify_timestamps.py",
        "--start",
        "2025-11-11",
        "--end",
        "2025-11-11",
        "--symbol",
        "MNQ",
        "--parquet-root",
        str(tmp_path.parent),
        "--kind",
        "depth",
        "--timestamp-tz",
        "UTC",
    ]
    # The script expects the files at data/parquet/depth/<symbol>/YYYYMMDD.parquet,
    # so create the expected folder structure and copy our file there.
    import shutil

    dest_dir = tmp_path / "data" / "parquet" / "depth" / "MNQ"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "20251111.parquet"
    shutil.copy(p, dest)
    # Run the verifier against tmp_path as the root
    cmd = [
        sys.executable,
        "scripts/verify_timestamps.py",
        "--start",
        "2025-11-11",
        "--end",
        "2025-11-11",
        "--symbol",
        "MNQ",
        "--parquet-root",
        str(tmp_path / "data" / "parquet"),
        "--kind",
        "depth",
        "--timestamp-tz",
        "UTC",
    ]
    proc = subprocess.Popen(cmd)
    rc = proc.wait()
    # Script should exit 0 for no mismatch
    assert rc == 0
