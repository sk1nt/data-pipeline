#!/usr/bin/env python3
"""
Bulk loader: read per-day tick Parquet files and bulk-load into DuckDB in a single writer step.

Usage:
  python3 scripts/load_ticks_to_db.py --ticks-dir data/parquet/ticks --db data/tick_mbo_data.db

Assumptions:
- Tick Parquet files are under `ticks-dir` in subfolders `YYYYMMDD/mnq_ticks.parquet` or similar.
- Schema is compatible with `mnq_ticks` table.
"""

import argparse
from pathlib import Path
import duckdb


def find_parquet_files(ticks_dir):
    p = Path(ticks_dir)
    files = list(p.rglob('*.parquet'))
    files.sort()
    return files


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ticks-dir', required=True)
    p.add_argument('--db', default='data/tick_mbo_data.db')
    p.add_argument('--table', default='mnq_ticks')
    return p.parse_args()


def main():
    args = parse_args()
    files = find_parquet_files(args.ticks_dir)
    if not files:
        print('No parquet files found in', args.ticks_dir)
        return

    print(f'Found {len(files)} parquet files. Loading into {args.db} table {args.table}')
    con = duckdb.connect(args.db)
    # We'll use a single transaction / writer
    try:
        for f in files:
            print('Loading', f)
            con.execute("BEGIN TRANSACTION")
            # Use DuckDB's efficient parquet scan + insert
            con.execute(f"INSERT INTO {args.table} SELECT * FROM read_parquet('{f}')")
            con.execute("COMMIT")
        print('All files loaded')
    finally:
        con.close()

if __name__ == '__main__':
    main()
