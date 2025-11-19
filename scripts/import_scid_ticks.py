#!/usr/bin/env python3
"""
Backfill MNQ/NQ tick data into data/tick_data.db by parsing Sierra Chart SCID files.

Usage:
    python scripts/import_scid_ticks.py \
        --scid-dir /mnt/c/SierraChart/Data \
        --start-date 2025-09-01 \
        --end-date 2025-11-07 \
        --symbols MNQ NQ
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import List

import duckdb

from src.lib.scid_parser import (
    is_bundled_trade,
    is_tick_record,
    parse_scid_file_backwards_generator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import SCID ticks into tick_data.db")
    parser.add_argument("--scid-dir", required=True, help="Directory containing *.scid files")
    parser.add_argument(
        "--start-date",
        required=True,
        help="UTC start date (YYYY-MM-DD) inclusive",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="UTC end date (YYYY-MM-DD) inclusive",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["MNQ", "NQ"],
        help="Base futures symbols to import (default: MNQ NQ)",
    )
    parser.add_argument(
        "--db-path",
        default="data/tick_data.db",
        help="DuckDB file to update (default: data/tick_data.db)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of rows per INSERT batch (default: 5000)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional cap on records read per SCID file (0 = no cap)",
    )
    return parser.parse_args()


def normalize_base_symbol(filename: str) -> str:
    """
    Normalize SCID filename to its base futures symbol.

    Examples:
        MNQM25-CME.scid -> MNQ
        NQZ25_FUT_CME.scid -> NQ
    """
    import re

    name = filename.upper()
    match = re.match(r"^([A-Z]+?)(?=[A-Z]\d|$)", name)
    if match:
        return match.group(1)
    match = re.match(r"^([A-Z]{2,4})", name)
    if match:
        return match.group(1)
    return name.replace(".SCID", "").split("-")[0].split("_")[0]


def iter_scid_files(scid_dir: Path, base_symbol: str) -> List[Path]:
    candidates: List[Path] = []
    for path in scid_dir.glob(f"{base_symbol}*.scid"):
        if normalize_base_symbol(path.name) == base_symbol:
            candidates.append(path)
    # Include root contract file (e.g., MNQ.scid) if it exists
    direct = scid_dir / f"{base_symbol}.scid"
    if direct.exists() and direct not in candidates:
        candidates.append(direct)
    return sorted(candidates)


def main() -> None:
    args = parse_args()
    scid_dir = Path(args.scid_dir).expanduser().resolve()
    if not scid_dir.exists():
        raise SystemExit(f"SCID directory not found: {scid_dir}")

    start_dt = dt.datetime.strptime(args.start_date, "%Y-%m-%d")
    end_dt = dt.datetime.strptime(args.end_date, "%Y-%m-%d") + dt.timedelta(days=1) - dt.timedelta(microseconds=1)
    max_records = args.max_records if args.max_records > 0 else None

    conn = duckdb.connect(args.db_path)
    conn.execute("PRAGMA threads=4")

    next_id = int(conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM tick_data").fetchone()[0])

    for symbol in args.symbols:
        symbol = symbol.upper()
        files = iter_scid_files(scid_dir, symbol)
        if not files:
            print(f"⚠️  No SCID files found for {symbol} under {scid_dir}")
            continue

        print(f"Processing {symbol}: {len(files)} SCID file(s)")
        conn.execute(
            "DELETE FROM tick_data WHERE symbol = ? AND timestamp BETWEEN ? AND ?",
            [symbol, start_dt, end_dt],
        )

        total_inserted = 0
        batch: List[tuple] = []

        for scid_path in files:
            inserted, next_id = import_file(
                scid_path,
                symbol,
                start_dt,
                end_dt,
                conn,
                batch,
                args.batch_size,
                max_records,
                next_id,
            )
            total_inserted += inserted
            print(f"  • {scid_path.name}: {inserted} ticks")

        if batch:
            conn.executemany(
                "INSERT INTO tick_data (id, symbol, timestamp, price, volume, tick_type, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            total_inserted += len(batch)
            batch.clear()

        print(f"✅ {symbol}: inserted {total_inserted} ticks\n")

    conn.close()


def import_file(
    scid_path: Path,
    target_symbol: str,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    conn: duckdb.DuckDBPyConnection,
    batch: List[tuple],
    batch_size: int,
    max_records: int | None,
    next_id: int,
) -> tuple[int, int]:
    inserted = 0
    for record in parse_scid_file_backwards_generator(str(scid_path), max_records=max_records):
        ts: dt.datetime = record["timestamp"]
        if ts > end_dt:
            continue
        if ts < start_dt:
            break
        if not is_tick_record(record):
            continue

        price = float(record["close"]) / 100.0  # SCID stores prices scaled by 100
        if price <= 0:
            continue
        volume = int(record["total_volume"])
        tick_type = "bundle" if is_bundled_trade(record) else "trade"
        batch.append((next_id, target_symbol, ts, price, volume, tick_type, "scid"))
        next_id += 1
        if len(batch) >= batch_size:
            conn.executemany(
                "INSERT INTO tick_data (id, symbol, timestamp, price, volume, tick_type, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            inserted += len(batch)
            batch.clear()

    return inserted, next_id


if __name__ == "__main__":
    main()
