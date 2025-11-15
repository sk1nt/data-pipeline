#!/usr/bin/env python3
"""
Split aggregated tick Parquet files into daily trading sessions (6pm ET → 5pm ET).
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import polars as pl
import pytz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split tick Parquet into session days.")
    parser.add_argument("--input", required=True, help="Source Parquet file (aggregated).")
    parser.add_argument("--symbol", required=True, help="Symbol label (e.g., MNQ).")
    parser.add_argument(
        "--output-month",
        required=True,
        help="Destination root in format data/parquet/ticks/YYYYMM/<symbol>/",
    )
    return parser.parse_args()


def session_bounds(timestamp: dt.datetime) -> tuple[dt.datetime, dt.datetime, dt.date]:
    eastern = pytz.timezone("US/Eastern")
    ts_local = timestamp.replace(tzinfo=dt.timezone.utc).astimezone(eastern)
    # Trading day boundary: 6pm local (18:00)
    day_start_local = ts_local.replace(hour=18, minute=0, second=0, microsecond=0)
    if ts_local < day_start_local:
        day_start_local -= dt.timedelta(days=1)
    day_end_local = day_start_local + dt.timedelta(days=1, hours=-1)
    day_start_utc = day_start_local.astimezone(dt.timezone.utc).replace(tzinfo=None)
    day_end_utc = day_end_local.astimezone(dt.timezone.utc).replace(tzinfo=None)
    session_date = day_end_local.date()
    return day_start_utc, day_end_utc, session_date


def split_sessions(df: pl.DataFrame, symbol: str, out_root: Path) -> None:
    if df.is_empty():
        return
    df = df.sort("timestamp")

    eastern = pytz.timezone("US/Eastern")
    session_dates = []
    for ts in df["timestamp"]:
        ts_local = ts.replace(tzinfo=dt.timezone.utc).astimezone(eastern)
        day_start = ts_local.replace(hour=18, minute=0, second=0, microsecond=0)
        if ts_local < day_start:
            day_start -= dt.timedelta(days=1)
        session_dates.append(day_start.date() + dt.timedelta(days=1))

    df = df.with_columns(pl.Series("session_date", session_dates))

    for group in df.partition_by("session_date", maintain_order=True):
        session_date = group["session_date"][0]
        day_str = session_date.strftime("%Y%m%d")
        out_dir = out_root / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{day_str}-{symbol}.parquet"
        (
            group.drop("session_date")
            .write_parquet(out_path, compression="zstd")
        )
        print(f"Wrote {out_path}")


def main() -> None:
    args = parse_args()
    in_path = Path(args.input).expanduser().resolve()
    df = pl.read_parquet(in_path)
    split_sessions(df, args.symbol, Path(args.output_month).expanduser().resolve())


if __name__ == "__main__":
    main()
