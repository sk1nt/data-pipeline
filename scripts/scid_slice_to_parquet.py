#!/usr/bin/env python3
"""
Slice a SCID file between two timestamps and write ticks to Parquet.
"""

from __future__ import annotations

import argparse
import datetime as dt
import struct
from pathlib import Path
from typing import Optional

import polars as pl

from src.lib.scid_parser import is_tick_record, is_bundled_trade

HEADER_SIZE = 56
RECORD_SIZE = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SCID ticks into Parquet.")
    parser.add_argument("--scid-file", required=True, help="Path to .scid file")
    parser.add_argument(
        "--start", required=False, help="Start timestamp (YYYY-MM-DD or ISO)."
    )
    parser.add_argument(
        "--end", required=False, help="End timestamp (YYYY-MM-DD or ISO)."
    )
    parser.add_argument(
        "--symbol", required=True, help="Symbol label to embed in output."
    )
    parser.add_argument("--output", required=True, help="Parquet file to write.")
    return parser.parse_args()


def parse_timestamp(
    value: Optional[str], default: Optional[dt.datetime]
) -> Optional[dt.datetime]:
    if value is None:
        return default
    try:
        if "T" in value:
            return dt.datetime.fromisoformat(value)
        return dt.datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise SystemExit(f"Invalid timestamp format: {value}")


def parse_record_bytes(data: bytes) -> Optional[dict]:
    if len(data) != RECORD_SIZE:
        return None
    fields = struct.unpack("<Q4f4I", data)
    timestamp_ms = fields[0]
    epoch = dt.datetime(1899, 12, 30)
    timestamp = epoch + dt.timedelta(milliseconds=int(str(timestamp_ms)[:-3]))
    counter = int(str(timestamp_ms)[-3:])
    return {
        "timestamp": timestamp,
        "counter": counter,
        "open": fields[1],
        "high": fields[2],
        "low": fields[3],
        "close": fields[4],
        "num_trades": fields[5],
        "total_volume": fields[6],
        "bid_volume": fields[7],
        "ask_volume": fields[8],
    }


def read_record_at(f, index: int) -> Optional[dict]:
    f.seek(HEADER_SIZE + index * RECORD_SIZE)
    data = f.read(RECORD_SIZE)
    return parse_record_bytes(data)


def locate_index(f, total_records: int, target: dt.datetime, find_first: bool) -> int:
    low, high = 0, total_records
    result = total_records if find_first else -1
    while low < high if find_first else low <= high:
        mid = (low + high) // 2
        rec = read_record_at(f, mid)
        if rec is None:
            break
        ts = rec["timestamp"]
        if find_first:
            if ts >= target:
                result = mid
                high = mid
            else:
                low = mid + 1
        else:
            if ts <= target:
                result = mid
                low = mid + 1
            else:
                high = mid - 1
    return max(0, min(result, total_records - 1))


def load_slice(
    path: Path, start_ts: Optional[dt.datetime], end_ts: Optional[dt.datetime]
) -> pl.DataFrame:
    with open(path, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        data_size = file_size - HEADER_SIZE
        total_records = data_size // RECORD_SIZE
        start_idx = 0
        end_idx = total_records - 1
        if start_ts:
            start_idx = locate_index(f, total_records, start_ts, True)
        if end_ts:
            end_idx = locate_index(f, total_records, end_ts, False)
        if end_idx < start_idx:
            return pl.DataFrame(schema={"timestamp": pl.Datetime, "price": pl.Float64})

        f.seek(HEADER_SIZE + start_idx * RECORD_SIZE)
        timestamps = []
        prices = []
        volumes = []
        tick_types = []
        sources = []
        counters = []
        for idx in range(start_idx, end_idx + 1):
            data = f.read(RECORD_SIZE)
            rec = parse_record_bytes(data)
            if rec is None:
                break
            ts = rec["timestamp"]
            if start_ts and ts < start_ts:
                continue
            if end_ts and ts > end_ts:
                break
            if not is_tick_record(rec):
                continue
            timestamps.append(ts)
            prices.append(float(rec["close"]) / 100.0)
            volumes.append(int(rec["total_volume"]))
            tick_types.append("bundle" if is_bundled_trade(rec) else "trade")
            sources.append("scid")
            counters.append(rec["counter"])

    if not timestamps:
        return pl.DataFrame(schema={"timestamp": pl.Datetime, "price": pl.Float64})

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,
            "tick_type": tick_types,
            "source": sources,
            "counter": counters,
        }
    ).sort("timestamp")


def main() -> None:
    args = parse_args()
    scid_path = Path(args.scid_file).expanduser().resolve()
    if not scid_path.exists():
        raise SystemExit(f"SCID file not found: {scid_path}")
    start_ts = parse_timestamp(args.start, None)
    end_ts = parse_timestamp(args.end, None)

    df = load_slice(scid_path, start_ts, end_ts)
    if df.is_empty():
        print("No ticks found in the requested window.")
        return

    df = df.with_columns(pl.lit(args.symbol).alias("symbol"))
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
