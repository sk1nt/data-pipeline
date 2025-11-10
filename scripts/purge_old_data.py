#!/usr/bin/env python3
"""Purge GEX data older than a cutoff date from DuckDB and Parquet."""

from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path
import duckdb
import sqlite3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove GEX data before cutoff date (UTC).")
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


def delete_from_gex_data(cutoff_ts: dt.datetime, dry_run: bool) -> None:
    conn = duckdb.connect("data/gex_data.db")
    try:
        for table in ["gex_snapshots", "gex_strikes"]:
            stmt = f"DELETE FROM {table} WHERE timestamp < ?"
            if dry_run:
                count = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE timestamp < ?", [cutoff_ts]).fetchone()[0]
                print(f"[dry-run] {table}: would delete {count:,} rows")
            else:
                conn.execute(stmt, [cutoff_ts])
                print(f"{table}: deleted rows before {cutoff_ts.date()}")
    finally:
        conn.close()


def delete_from_gex_history(epoch_cutoff: int, dry_run: bool) -> None:
    tables = ["gex_bridge_snapshots", "gex_bridge_strikes"]
    for table in tables:
        retries = 0
        while True:
            conn = sqlite3.connect("data/gex_history.db", timeout=10.0)
            try:
                stmt = f"DELETE FROM {table} WHERE timestamp < ?"
                if dry_run:
                    count = conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE timestamp < ?",
                        [epoch_cutoff],
                    ).fetchone()[0]
                    print(f"[dry-run] {table}: would delete {count:,} rows")
                else:
                    conn.execute(stmt, [epoch_cutoff])
                    conn.commit()
                    print(f"{table}: deleted rows before epoch {epoch_cutoff}")
                break
            except sqlite3.OperationalError as exc:
                if "database is locked" in str(exc).lower() and retries < 5:
                    retries += 1
                    time.sleep(0.5 * retries)
                    continue
                raise
            finally:
                conn.close()


def prune_parquet(root: Path, cutoff: dt.date, dry_run: bool) -> None:
    if not root.exists():
        return
    for year_dir in root.iterdir():
        if not year_dir.is_dir():
            continue
        try:
            year = int(year_dir.name)
        except ValueError:
            continue
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir():
                continue
            try:
                month = int(month_dir.name)
            except ValueError:
                continue
            dir_date = dt.date(year, month, 1)
            if dir_date >= dt.date(cutoff.year, cutoff.month, 1):
                continue
            if dry_run:
                print(f"[dry-run] would remove {month_dir}")
            else:
                for child in month_dir.rglob("*"):
                    if child.is_file():
                        child.unlink()
                for child in sorted(month_dir.rglob("*"), reverse=True):
                    if child.exists() and child.is_dir():
                        child.rmdir()
                month_dir.rmdir()
                print(f"Removed {month_dir}")
        if not any(year_dir.iterdir()):
            if dry_run:
                print(f"[dry-run] would remove empty {year_dir}")
            else:
                year_dir.rmdir()


def main() -> None:
    args = parse_args()
    cutoff_date = dt.datetime.fromisoformat(args.cutoff).date()
    cutoff_ts = dt.datetime.combine(cutoff_date, dt.time.min)
    epoch_cutoff = int(cutoff_ts.timestamp())

    delete_from_gex_data(cutoff_ts, args.dry_run)
    delete_from_gex_history(epoch_cutoff, args.dry_run)
    prune_parquet(Path("data/parquet/gex"), cutoff_date, args.dry_run)


if __name__ == "__main__":
    main()
