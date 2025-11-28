#!/usr/bin/env python3
"""Purge GEX data older than a cutoff date from DuckDB and Parquet."""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path
import duckdb
from zoneinfo import ZoneInfo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove GEX data before cutoff date (UTC)."
    )
    parser.add_argument(
        "--cutoff",
        required=True,
        help="ISO date (YYYY-MM-DD). Rows strictly earlier than this date are deleted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned deletions without modifying data.",
    )
    return parser.parse_args()


def delete_from_gex_data(cutoff_epoch_ms: int, dry_run: bool) -> None:
    conn = duckdb.connect("data/gex_data.db")
    try:
        for table in ["gex_snapshots", "gex_strikes"]:
            stmt = f"DELETE FROM {table} WHERE timestamp < ?"
            if dry_run:
                count = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE timestamp < ?",
                    [cutoff_epoch_ms],
                ).fetchone()[0]
                print(f"[dry-run] {table}: would delete {count:,} rows")
            else:
                conn.execute(stmt, [cutoff_epoch_ms])
                print(f"{table}: deleted rows before epoch_ms={cutoff_epoch_ms}")
    finally:
        conn.close()


def delete_from_gex_history(epoch_cutoff: int, dry_run: bool) -> None:
    # Legacy history tables have been removed; nothing to purge.
    print("No history tables to purge (gex_bridge_* removed). Skipping.")


def prune_parquet(root: Path, cutoff: dt.date, dry_run: bool) -> None:
    if not root.exists():
        return
    pattern = re.compile(r"(\d{8})\.strikes\.parquet$")
    cutoff_epoch = int(cutoff.strftime("%Y%m%d"))
    for ticker_dir in root.iterdir():
        if not ticker_dir.is_dir():
            continue
        for endpoint_dir in ticker_dir.iterdir():
            if not endpoint_dir.is_dir():
                continue
            for parquet_file in list(endpoint_dir.glob("*.strikes.parquet")):
                match = pattern.match(parquet_file.name)
                if not match:
                    continue
                file_day = int(match.group(1))
                if file_day >= cutoff_epoch:
                    continue
                if dry_run:
                    print(f"[dry-run] would remove {parquet_file}")
                else:
                    parquet_file.unlink()
            if not any(endpoint_dir.iterdir()):
                if dry_run:
                    print(f"[dry-run] would remove empty {endpoint_dir}")
                else:
                    endpoint_dir.rmdir()
        if not any(ticker_dir.iterdir()):
            if dry_run:
                print(f"[dry-run] would remove empty {ticker_dir}")
            else:
                ticker_dir.rmdir()


def main() -> None:
    args = parse_args()
    cutoff_date = dt.datetime.fromisoformat(args.cutoff).date()
    ny_tz = ZoneInfo("America/New_York")
    cutoff_dt = dt.datetime.combine(cutoff_date, dt.time.min, tzinfo=ny_tz)
    cutoff_epoch_ms = int(cutoff_dt.astimezone(dt.timezone.utc).timestamp() * 1000)
    epoch_cutoff = int(cutoff_dt.timestamp())

    delete_from_gex_data(cutoff_epoch_ms, args.dry_run)
    delete_from_gex_history(epoch_cutoff, args.dry_run)
    prune_parquet(Path("data/parquet/gexbot"), cutoff_date, args.dry_run)


if __name__ == "__main__":
    main()
