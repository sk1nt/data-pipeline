#!/usr/bin/env python3
"""Convert Parquet files to include ts_ms column (epoch milliseconds).

Usage:
  python scripts/convert_parquet_to_ts_ms.py <input.parquet> [--out <output.parquet>] [--compression zstd] [--atomic]

This script reads a parquet file, inspects the 'timestamp' column type, and
adds a 'ts_ms' BIGINT column with epoch milliseconds, handling both TIMESTAMP
and numeric epoch seconds/ms fields.

It writes to a temp file and atomically renames to the destination.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import duckdb
import sys


def run_single(
    input_path: Path,
    output_path: Path,
    compression: str = "zstd",
    atomic: bool = True,
    timestamp_tz: str = "UTC",
):
    con = duckdb.connect()
    # Check timestamp type
    try:
        t = con.execute(
            f"SELECT typeof(timestamp) AS t FROM read_parquet('{input_path}') LIMIT 1"
        ).fetchone()[0]
    except Exception:
        # If timestamp column does not exist, try to detect 'ts_ms' or 'timestamp_ms'
        raise

    print(f"Detected timestamp type: {t}")
    if t.upper().startswith("TIMESTAMP"):
        # Read the input schema to enumerate columns and exclude ts_ms if present
        cols = (
            con.execute(f"SELECT * FROM read_parquet('{input_path}') LIMIT 1")
            .fetchdf()
            .columns.tolist()
        )
        select_cols = ",".join([c for c in cols if c != "ts_ms"])
        # Use AT TIME ZONE if a timezone is supplied, ensuring DuckDB computes
        # epoch from the intended timezone rather than the server/system default.
        tz_expr = (
            f"timestamp AT TIME ZONE '{timestamp_tz}'" if timestamp_tz else "timestamp"
        )
        query = (
            f"SELECT {select_cols}, CAST(EXTRACT(epoch FROM {tz_expr}) * 1000 AS BIGINT) as ts_ms "
            f"FROM read_parquet('{input_path}')"
        )
    elif t.upper() in (
        "INT64",
        "BIGINT",
        "INTEGER",
        "INT",
    ):  # numeric epoch seconds or ms
        # Choose safe conversion: if values are large > 1e12 assume ms, else multiply by 1000
        cols = (
            con.execute(f"SELECT * FROM read_parquet('{input_path}') LIMIT 1")
            .fetchdf()
            .columns.tolist()
        )
        select_cols = ",".join([c for c in cols if c != "ts_ms"])
        query = (
            f"SELECT {select_cols}, CASE WHEN timestamp >= 1000000000000 THEN CAST(timestamp AS BIGINT) ELSE CAST(timestamp * 1000 AS BIGINT) END AS ts_ms "
            f"FROM read_parquet('{input_path}')"
        )
    else:
        # Unknown type - attempt to cast
        cols = (
            con.execute(f"SELECT * FROM read_parquet('{input_path}') LIMIT 1")
            .fetchdf()
            .columns.tolist()
        )
        select_cols = ",".join([c for c in cols if c != "ts_ms"])
        tz_expr = (
            f"timestamp AT TIME ZONE '{timestamp_tz}'" if timestamp_tz else "timestamp"
        )
        query = (
            f"SELECT {select_cols}, CAST(EXTRACT(epoch FROM {tz_expr}) * 1000 AS BIGINT) as ts_ms "
            f"FROM read_parquet('{input_path}')"
        )

    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    print("Running transformation, writing to", tmp)
    con.execute(f"COPY ({query}) TO '{tmp}' (FORMAT PARQUET)")
    con.close()
    if atomic:
        tmp.replace(output_path)
        print("Wrote", output_path)
    else:
        print("Wrote", tmp, " (not renamed)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Input parquet file (path)")
    p.add_argument("--out", help="Output parquet file", default=None)
    p.add_argument("--compression", default="zstd")
    p.add_argument("--atomic", action="store_true", default=True)
    p.add_argument(
        "--timestamp-tz",
        default="UTC",
        help="Timezone to use for timestamp->epoch conversion when timestamp is a TIMESTAMP column",
    )
    args = p.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print("Input not found:", input_path)
        sys.exit(2)
    out = Path(args.out) if args.out else input_path
    run_single(
        input_path,
        out,
        compression=args.compression,
        atomic=args.atomic,
        timestamp_tz=args.timestamp_tz,
    )


if __name__ == "__main__":
    main()
