#!/usr/bin/env python3
"""
Export MNQ/NQ SCID tick data into daily Parquet files.

Each symbol/date pair becomes `data/parquet/ticks/<SYMBOL>/<YYYY-MM-DD>.parquet`.

Roll schedule (CME equity index futures) follows the quarterly pattern:

| Quarter | Contract | Third Friday | Sunday Globex roll start (used here) |
|---------|----------|--------------|--------------------------------------|
| Q1 2025 | H25      | 2025-03-21   | 2025-03-09 22:00 UTC                 |
| Q2 2025 | M25      | 2025-06-20   | 2025-06-08 22:00 UTC                 |
| Q3 2025 | U25      | 2025-09-19   | 2025-09-14 22:00 UTC                 |
| Q4 2025 | Z25      | 2025-12-19   | 2025-12-07 22:00 UTC                 |

We begin reading each new contract file at the **Sunday 22:00 UTC** Globex
session that starts CME’s roll week (third Friday − 12 days), so all trading
in that week belongs to the incoming contract.
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import polars as pl

from src.lib.scid_parser import (
    parse_scid_file_backwards_generator,
    is_tick_record,
    is_bundled_trade,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SCID ticks to Parquet")
    parser.add_argument(
        "--scid-dir", required=True, help="Directory containing *.scid files"
    )
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
        help="Base futures symbols to export (default: MNQ NQ)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/parquet/ticks",
        help="Root directory for parquet output (default: data/parquet/ticks)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional cap on records read per SCID file (0 = no cap)",
    )
    return parser.parse_args()


MONTH_CODES = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}


def normalize_symbol(filename: str) -> str:
    import re

    name = filename.upper()
    match = re.match(r"^([A-Z]+?)(?=[A-Z]\d|$)", name)
    if match:
        return match.group(1)
    match = re.match(r"^([A-Z]{2,4})", name)
    if match:
        return match.group(1)
    return name.replace(".SCID", "").split("-")[0].split("_")[0]


def find_scid_files(scid_dir: Path, symbol: str) -> List[Path]:
    files = []
    for path in scid_dir.glob(f"{symbol}*.scid"):
        if normalize_symbol(path.name) == symbol:
            files.append(path)
    base = scid_dir / f"{symbol}.scid"
    if base.exists() and base not in files:
        files.append(base)
    return sorted(files)


def parse_contract(filename: str) -> Tuple[str, str, int] | None:
    """
    Extract base symbol, month letter, and 2-digit year from filename.
    """
    import re

    name = filename.upper()
    match = re.search(r"([A-Z]+?)([FGHJKMNQUVXZ])(\d{2})", name)
    if not match:
        return None
    base, month_letter, year_suffix = match.groups()
    year = 2000 + int(year_suffix)
    return base, month_letter, year


def third_friday(year: int, month: int) -> dt.date:
    date = dt.date(year, month, 15)
    while date.weekday() != 4:  # 4=Friday
        date += dt.timedelta(days=1)
    return date


def roll_start_datetime(exp_year: int, exp_month: int) -> dt.datetime:
    """
    Compute the Sunday 22:00 UTC Globex open that starts CME roll week.
    """
    # Quarter start is 3 months earlier
    start_month = exp_month - 3
    start_year = exp_year
    if start_month <= 0:
        start_month += 12
        start_year -= 1

    fr = third_friday(start_year, start_month)
    roll_sunday = fr - dt.timedelta(days=12)
    return dt.datetime.combine(roll_sunday, dt.time(22, 0))


def build_contract_windows(
    scid_files: List[Path],
) -> Dict[Path, Tuple[dt.datetime, dt.datetime]]:
    """
    For each SCID file determine [start, end) timestamps that should map to that contract
    based on CME quarterly roll schedule.
    """
    contract_entries = []
    for path in scid_files:
        meta = parse_contract(path.name)
        if not meta:
            continue
        base, month_letter, year = meta
        month_num = MONTH_CODES[month_letter]
        start_dt = roll_start_datetime(year, month_num)
        contract_entries.append((path, start_dt, (base, month_letter, year)))

    # Sort by contract start
    contract_entries.sort(key=lambda item: item[1])

    windows: Dict[Path, Tuple[dt.datetime, dt.datetime]] = {}
    for idx, (path, start_dt, _) in enumerate(contract_entries):
        if idx + 1 < len(contract_entries):
            end_dt = contract_entries[idx + 1][1]
        else:
            end_dt = dt.datetime.max
        windows[path] = (start_dt, end_dt)
    return windows


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_parquet(
    symbol: str, date_key: dt.date, rows: List[dict], out_root: Path
) -> None:
    if not rows:
        return
    df = (
        pl.DataFrame(rows)
        .sort("timestamp")
        .with_columns(pl.lit(symbol).alias("symbol"))
    )
    out_path = out_root / symbol / f"{date_key.isoformat()}.parquet"
    ensure_dir(out_path)
    df.write_parquet(out_path, compression="zstd")
    print(f"  → wrote {len(rows):6d} rows to {out_path}")


def export_symbol(
    symbol: str,
    scid_files: List[Path],
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    out_root: Path,
    max_records: int | None,
) -> None:
    if not scid_files:
        print(f"⚠️  No SCID files for {symbol}")
        return

    contract_windows = build_contract_windows(scid_files)
    buckets: Dict[dt.date, List[dict]] = defaultdict(list)
    seen: Set[Tuple[dt.datetime, int]] = set()

    print(f"Processing {symbol}: {len(scid_files)} file(s)")
    for scid_path in scid_files:
        count = 0
        interval = contract_windows.get(scid_path)
        for record in parse_scid_file_backwards_generator(
            str(scid_path), max_records=max_records
        ):
            ts: dt.datetime = record["timestamp"]
            if ts > end_dt:
                continue
            if ts < start_dt:
                break
            if interval:
                start_window, end_window = interval
                if ts < start_window or ts >= end_window:
                    continue
            if not is_tick_record(record):
                continue

            key = (ts, record["counter"])
            if key in seen:
                continue
            seen.add(key)

            date_key = ts.date()
            row = {
                "timestamp": ts,
                "price": float(record["close"]) / 100.0,
                "volume": int(record["total_volume"]),
                "tick_type": "bundle" if is_bundled_trade(record) else "trade",
                "source": "scid",
                "counter": record["counter"],
            }
            buckets[date_key].append(row)
            count += 1
        print(f"  • {scid_path.name}: collected {count} ticks")

    for date_key, rows in sorted(buckets.items()):
        if not rows:
            continue
        write_parquet(symbol, date_key, rows, out_root)


def main() -> None:
    args = parse_args()
    scid_dir = Path(args.scid_dir).expanduser().resolve()
    if not scid_dir.exists():
        raise SystemExit(f"SCID directory not found: {scid_dir}")

    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    start_dt = dt.datetime.strptime(args.start_date, "%Y-%m-%d")
    # inclusive end-of-day
    end_dt = (
        dt.datetime.strptime(args.end_date, "%Y-%m-%d")
        + dt.timedelta(days=1)
        - dt.timedelta(microseconds=1)
    )
    max_records = args.max_records if args.max_records > 0 else None

    for symbol in args.symbols:
        base_symbol = symbol.upper()
        files = find_scid_files(scid_dir, base_symbol)
        export_symbol(base_symbol, files, start_dt, end_dt, out_root, max_records)


if __name__ == "__main__":
    main()
