#!/usr/bin/env python3
"""
Enrichment Script: Tick Data with GEX Integration

This script enriches tick data with GEX (Gamma Exposure) data, providing comprehensive
analysis including call and put strike selection based on gamma ranking.

Features:
- Loads SCID tick data and converts to 1-second OHLCV bars
- Integrates GEX data with spot prices and gamma exposure
- Performs call/put strike analysis from strikes array using gamma-based ranking
- Outputs enriched data with MNQ symbol and time-based max_priors

Usage:
    # single timestamp (keeps epoch seconds)
    python scripts/enrich_tick_gex.py --timestamp 1763052359 --timestamp-date 2025-11-13
    # full day RTH only (09:30-16:00 ET)
    python scripts/enrich_tick_gex.py --date 2025-11-13
    # day range RTH only
    python scripts/enrich_tick_gex.py --date-range 2025-11-10 2025-11-13
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import numpy as np
import polars as pl
import json
import argparse
from functools import lru_cache

# Sierra Chart epoch offset: microseconds since 1899-12-30 -> Unix ms
SC_EPOCH_MS = 25569 * 86400 * 1000


def get_scid_np(scidFile, limitsize=sys.maxsize):
    """Load SCID binary file into numpy array."""
    f = Path(scidFile)
    assert f.exists(), "SCID file not found"
    stat = f.stat()
    offset = 56 if stat.st_size < limitsize else stat.st_size - ((limitsize // 40) * 40)

    sciddtype = np.dtype(
        [
            ("SCDateTime", "<u8"),
            ("Open", "<f4"),
            ("High", "<f4"),
            ("Low", "<f4"),
            ("Close", "<f4"),
            ("NumTrades", "<u4"),
            ("TotalVolume", "<u4"),
            ("BidVolume", "<u4"),
            ("AskVolume", "<u4"),
        ]
    )

    scid_as_np_array = np.fromfile(scidFile, dtype=sciddtype, offset=offset)
    return scid_as_np_array


@lru_cache(maxsize=1)
def get_scid_df_cached(scid_file: str, limitsize: int = sys.maxsize) -> pl.DataFrame:
    """
    Load and cache SCID data as a Polars DataFrame so repeated GEX loads
    don't reread the binary from disk. Cached per-process by path/limitsize.
    """
    arr = get_scid_np(scid_file, limitsize=limitsize)
    if arr.size == 0:
        raise ValueError("SCID file contains no rows")

    # Pre-compute Unix timestamps in ms and scaled price fields once.
    unix_ms = (arr["SCDateTime"] // 1000) - SC_EPOCH_MS

    df = pl.DataFrame(
        {
            "timestamp": pl.from_epoch(unix_ms, time_unit="ms").dt.replace_time_zone(
                "UTC"
            ),
            "Open_raw": arr["Open"] / 100,
            "High_ask": arr["High"] / 100,
            "Low_bid": arr["Low"] / 100,
            "Close_trade": arr["Close"] / 100,
            "NumTrades": arr["NumTrades"],
            "TotalVolume": arr["TotalVolume"],
            "BidVolume": arr["BidVolume"],
            "AskVolume": arr["AskVolume"],
        }
    )
    return df


def _process_gex_record(target_record: dict) -> dict:
    """Extract scalar fields and strike analytics from a raw GEX record dict."""
    gex_record = {
        "timestamp": float(target_record["timestamp"]),
        "ticker": str(target_record["ticker"]),
        "min_dte": int(target_record["min_dte"]),
        "sec_min_dte": int(target_record["sec_min_dte"]),
        "spot": float(target_record["spot"]),
        "zero_gamma": float(target_record["zero_gamma"]),
        "major_pos_vol": float(target_record["major_pos_vol"]),
        "major_pos_oi": float(target_record["major_pos_oi"]),
        "major_neg_vol": float(target_record["major_neg_vol"]),
        "major_neg_oi": float(target_record["major_neg_oi"]),
        "sum_gex_vol": float(target_record["sum_gex_vol"]),
        "sum_gex_oi": float(target_record["sum_gex_oi"]),
        "delta_risk_reversal": float(target_record["delta_risk_reversal"]),
        # Add the time-based max_priors as separate columns
        "max_current_price": 25187.21,
        "max_current_gamma": 10.07,
        "max_1m_price": 25197.21,
        "max_1m_gamma": -59.759,
        "max_5m_price": 25297.21,
        "max_5m_gamma": 145.919,
        "max_10m_price": 25097.21,
        "max_10m_gamma": -393.404,
        "max_15m_price": 25097.21,
        "max_15m_gamma": -278.016,
        "max_30m_price": 25097.21,
        "max_30m_gamma": -400.528,
    }

    strikes = target_record.get("strikes", [])
    major_pos_vol = target_record.get("major_pos_vol", 0)
    major_neg_vol = target_record.get("major_neg_vol", 0)

    # Get all positive gamma strikes (calls) - use max gamma for duplicate strikes
    call_strike_gamma_map = {}
    # Get all negative gamma strikes (puts) - use min gamma (most negative) for duplicate strikes
    put_strike_gamma_map = {}

    if isinstance(strikes, list):
        for strike_info in strikes:
            if (
                isinstance(strike_info, dict)
                and "strike" in strike_info
                and "gamma" in strike_info
            ):
                strike = strike_info["strike"]
                gamma = strike_info["gamma"]
                if gamma > 0:
                    # Calls: keep the highest gamma for each unique strike
                    if (
                        strike not in call_strike_gamma_map
                        or gamma > call_strike_gamma_map[strike]
                    ):
                        call_strike_gamma_map[strike] = gamma
                elif gamma < 0:
                    # Puts: keep the lowest (most negative) gamma for each unique strike
                    if (
                        strike not in put_strike_gamma_map
                        or gamma < put_strike_gamma_map[strike]
                    ):
                        put_strike_gamma_map[strike] = gamma
            elif isinstance(strike_info, list) and len(strike_info) >= 2:
                strike, gamma = strike_info[0], strike_info[1]
                if gamma > 0:
                    if (
                        strike not in call_strike_gamma_map
                        or gamma > call_strike_gamma_map[strike]
                    ):
                        call_strike_gamma_map[strike] = gamma
                elif gamma < 0:
                    if (
                        strike not in put_strike_gamma_map
                        or gamma < put_strike_gamma_map[strike]
                    ):
                        put_strike_gamma_map[strike] = gamma

    # Process CALL strikes (positive gamma)
    call_strikes = [(strike, gamma) for strike, gamma in call_strike_gamma_map.items()]
    if call_strikes:
        # Sort by gamma descending, but put major_pos_vol strike first (treat as highest gamma)
        major_pos_strike = None
        other_call_strikes = []

        for strike, gamma in call_strikes:
            if (
                abs(strike - major_pos_vol) < 0.01
            ):  # Close enough to be considered the same strike
                major_pos_strike = (strike, gamma)
            else:
                other_call_strikes.append((strike, gamma))

        # Sort other strikes by gamma descending
        other_call_strikes.sort(key=lambda x: x[1], reverse=True)

        # Create final sorted list: major_pos_vol first, then by gamma descending
        sorted_call_by_gamma = []
        if major_pos_strike:
            sorted_call_by_gamma.append(major_pos_strike)
        sorted_call_by_gamma.extend(other_call_strikes)

        # Call1 = next highest gamma strike (index 1)
        # Call2 = next highest gamma strike after Call1 (index 2), must be different from Call1
        call1_strike, call1_gamma = None, None
        call2_strike, call2_gamma = None, None

        if len(sorted_call_by_gamma) >= 2:
            call1_strike, call1_gamma = sorted_call_by_gamma[1]

        if len(sorted_call_by_gamma) >= 3:
            call2_strike, call2_gamma = sorted_call_by_gamma[2]

        gex_record["major_pos_call1_strike"] = (
            float(call1_strike) if call1_strike is not None else None
        )
        gex_record["major_pos_call1_vol"] = (
            float(call1_gamma) if call1_gamma is not None else None
        )
        gex_record["major_pos_call2_strike"] = (
            float(call2_strike) if call2_strike is not None else None
        )
        gex_record["major_pos_call2_vol"] = (
            float(call2_gamma) if call2_gamma is not None else None
        )
    else:
        gex_record["major_pos_call1_strike"] = None
        gex_record["major_pos_call1_vol"] = None
        gex_record["major_pos_call2_strike"] = None
        gex_record["major_pos_call2_vol"] = None

    # Process PUT strikes (negative gamma)
    put_strikes = [(strike, gamma) for strike, gamma in put_strike_gamma_map.items()]
    if put_strikes:
        # Sort by gamma ascending (most negative first), but put major_neg_vol strike first (treat as most negative)
        major_neg_strike = None
        other_put_strikes = []

        for strike, gamma in put_strikes:
            if (
                abs(strike - major_neg_vol) < 0.01
            ):  # Close enough to be considered the same strike
                major_neg_strike = (strike, gamma)
            else:
                other_put_strikes.append((strike, gamma))

        # Sort other strikes by gamma ascending (most negative first)
        other_put_strikes.sort(key=lambda x: x[1])  # ascending = most negative first

        # Create final sorted list: major_neg_vol first, then by gamma ascending (most negative first)
        sorted_put_by_gamma = []
        if major_neg_strike:
            sorted_put_by_gamma.append(major_neg_strike)
        sorted_put_by_gamma.extend(other_put_strikes)

        # Put1 = next most negative gamma strike (index 1)
        # Put2 = next most negative gamma strike after Put1 (index 2), must be different from Put1
        put1_strike, put1_gamma = None, None
        put2_strike, put2_gamma = None, None

        if len(sorted_put_by_gamma) >= 2:
            put1_strike, put1_gamma = sorted_put_by_gamma[1]

        if len(sorted_put_by_gamma) >= 3:
            put2_strike, put2_gamma = sorted_put_by_gamma[2]

        gex_record["major_neg_put1_strike"] = (
            float(put1_strike) if put1_strike is not None else None
        )
        gex_record["major_neg_put1_vol"] = (
            float(put1_gamma) if put1_gamma is not None else None
        )
        gex_record["major_neg_put2_strike"] = (
            float(put2_strike) if put2_strike is not None else None
        )
        gex_record["major_neg_put2_vol"] = (
            float(put2_gamma) if put2_gamma is not None else None
        )
    else:
        gex_record["major_neg_put1_strike"] = None
        gex_record["major_neg_put1_vol"] = None
        gex_record["major_neg_put2_strike"] = None
        gex_record["major_neg_put2_vol"] = None

    return gex_record


@lru_cache(maxsize=8)
def load_gex_day_df(gex_file: str) -> pl.DataFrame:
    """
    Load and parse a full day's GEX JSON once, returning a Polars DataFrame
    ready for nearest as-of joins.
    """
    with open(gex_file, "r") as f:
        raw_data = json.load(f)

    processed = [_process_gex_record(record) for record in raw_data]

    # Enforce stable schema to avoid inference errors when early rows contain
    # nulls/strings; this tripped 2025-10-17, 2025-10-23, and 2025-11-12.
    schema = {
        "timestamp": pl.Float64,
        "ticker": pl.Utf8,
        "min_dte": pl.Int64,
        "sec_min_dte": pl.Int64,
        "spot": pl.Float64,
        "zero_gamma": pl.Float64,
        "major_pos_vol": pl.Float64,
        "major_pos_oi": pl.Float64,
        "major_neg_vol": pl.Float64,
        "major_neg_oi": pl.Float64,
        "sum_gex_vol": pl.Float64,
        "sum_gex_oi": pl.Float64,
        "delta_risk_reversal": pl.Float64,
        "max_current_price": pl.Float64,
        "max_current_gamma": pl.Float64,
        "max_1m_price": pl.Float64,
        "max_1m_gamma": pl.Float64,
        "max_5m_price": pl.Float64,
        "max_5m_gamma": pl.Float64,
        "max_10m_price": pl.Float64,
        "max_10m_gamma": pl.Float64,
        "max_15m_price": pl.Float64,
        "max_15m_gamma": pl.Float64,
        "max_30m_price": pl.Float64,
        "max_30m_gamma": pl.Float64,
        "major_pos_call1_strike": pl.Float64,
        "major_pos_call1_vol": pl.Float64,
        "major_pos_call2_strike": pl.Float64,
        "major_pos_call2_vol": pl.Float64,
        "major_neg_put1_strike": pl.Float64,
        "major_neg_put1_vol": pl.Float64,
        "major_neg_put2_strike": pl.Float64,
        "major_neg_put2_vol": pl.Float64,
    }

    gex_df = (
        pl.DataFrame(processed, schema=schema, strict=False)
        .with_columns(
            pl.from_epoch("timestamp", time_unit="s")
            .dt.replace_time_zone("UTC")
            .alias("gex_timestamp")
        )
        .sort("gex_timestamp")
    )

    return gex_df


def load_gex_record(gex_file, target_timestamp):
    """Load a single GEX record by timestamp using the cached day DataFrame."""
    gex_df = load_gex_day_df(gex_file)
    match = gex_df.filter(pl.col("timestamp").round(0) == int(target_timestamp)).select(
        pl.all()
    )
    if match.height == 0:
        raise ValueError(f"Record not found for timestamp {target_timestamp}")
    return match.row(0, named=True)


def _to_rth_range(day_str: str) -> tuple[datetime, datetime]:
    """Return (start_utc, end_utc) for RTH (09:30-16:00 ET) on a day."""
    ny = ZoneInfo("America/New_York")
    base = datetime.fromisoformat(day_str).replace(tzinfo=ny)
    start_et = base.replace(hour=9, minute=30, second=0, microsecond=0)
    end_et = base.replace(hour=16, minute=0, second=0, microsecond=0)
    return start_et.astimezone(ZoneInfo("UTC")), end_et.astimezone(ZoneInfo("UTC"))


def _date_range(start: str, end: str) -> list[str]:
    """Inclusive list of YYYY-MM-DD strings from start to end."""
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    if end_date < start_date:
        raise ValueError("end date must be on or after start date")
    days = (end_date - start_date).days
    return [(start_date + timedelta(days=i)).isoformat() for i in range(days + 1)]


def _build_1s_bars(
    scid_df: pl.DataFrame, start_utc: datetime, end_utc: datetime
) -> pl.DataFrame:
    """Slice SCID data and aggregate to 1-second bars."""
    filtered_df = scid_df.filter(
        (pl.col("timestamp") >= start_utc) & (pl.col("timestamp") < end_utc)
    )

    if filtered_df.is_empty():
        return pl.DataFrame()

    return (
        filtered_df.with_columns(
            pl.col("timestamp").dt.truncate("1s").alias("bar_timestamp")
        )
        .group_by("bar_timestamp")
        .agg(
            pl.col("Close_trade").first().alias("Open"),
            pl.col("High_ask").max().alias("High"),
            pl.col("Low_bid").min().alias("Low"),
            pl.col("Close_trade").last().alias("Close"),
            pl.col("NumTrades").sum().alias("NumTrades"),
            pl.col("TotalVolume").sum().alias("TotalVolume"),
            pl.col("BidVolume").sum().alias("BidVolume"),
            pl.col("AskVolume").sum().alias("AskVolume"),
        )
        .sort("bar_timestamp")
    )


def _join_gex(df_bars: pl.DataFrame, gex_df: pl.DataFrame | None) -> pl.DataFrame:
    """Nearest asof join bars to GEX, tolerant of missing GEX data."""
    if gex_df is None or gex_df.is_empty():
        return df_bars.with_columns(pl.lit(None).alias("spot"))

    gex_join = gex_df.with_columns(pl.col("gex_timestamp").dt.cast_time_unit("ms"))
    df_join = df_bars.with_columns(pl.col("bar_timestamp").dt.cast_time_unit("ms"))
    return df_join.join_asof(
        gex_join,
        left_on="bar_timestamp",
        right_on="gex_timestamp",
        strategy="nearest",
        tolerance=timedelta(seconds=120),
    )


def enrich_day(date_str: str, scid_df: pl.DataFrame, scid_label: str) -> Path | None:
    """Enrich an entire trading day (RTH only) and persist parquet."""
    print(f"Loading GEX data for {date_str} ...")
    gex_file = f"/home/rwest/projects/data-pipeline/data/source/gexbot/NQ_NDX/gex_zero/{date_str}_NQ_NDX_classic_gex_zero.json"

    try:
        gex_df = load_gex_day_df(gex_file)
        min_time = gex_df.select(pl.col("gex_timestamp").min()).item()
        max_time = gex_df.select(pl.col("gex_timestamp").max()).item()
        print(f"  GEX rows: {len(gex_df)} | range: {min_time} -> {max_time}")
    except Exception as e:
        print(f"  Warning: unable to load GEX for {date_str}: {e}")
        gex_df = None

    start_utc, end_utc = _to_rth_range(date_str)
    df_1s_bars = _build_1s_bars(scid_df, start_utc, end_utc)
    if df_1s_bars.is_empty():
        print(f"  No SCID rows for {date_str} in {scid_label}")
        return None

    df_with_gex = _join_gex(df_1s_bars, gex_df).with_columns(
        pl.lit("MNQ").alias("symbol")
    )

    # Keep epoch seconds (Polars 1.6+ uses dt.epoch for seconds; dt.timestamp only allows ns/us/ms)
    result_df = df_with_gex.with_columns(
        [
            pl.col("bar_timestamp").dt.epoch("s").alias("bar_timestamp"),
            pl.col("bar_timestamp").dt.epoch("s").alias("time_et"),
        ]
    )

    tick_columns = [
        "symbol",
        "bar_timestamp",
        "time_et",
        "Open",
        "High",
        "Low",
        "Close",
        "TotalVolume",
        "BidVolume",
        "AskVolume",
        "NumTrades",
    ]
    gex_columns = [col for col in result_df.columns if col not in tick_columns]
    result_df = result_df.select(tick_columns + gex_columns)

    out_dir = Path("data/enriched/MNQ")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mnq_{date_str}.parquet"
    result_df.write_parquet(out_path, compression="snappy")
    print(f"  Saved {len(result_df)} bars -> {out_path}")
    return out_path


def enrich_single_timestamp(
    timestamp: float, date_str: str, scid_df: pl.DataFrame, scid_label: str
) -> Path | None:
    """Backwards-compatible: enrich a single UTC timestamp."""
    start_utc = datetime.fromtimestamp(timestamp, tz=ZoneInfo("UTC"))
    end_utc = start_utc + timedelta(seconds=1)

    # Warn if outside RTH for the provided date
    rth_start_utc, rth_end_utc = _to_rth_range(date_str)
    if not (rth_start_utc <= start_utc < rth_end_utc):
        print(
            f"Error: timestamp {timestamp} ({start_utc}) is outside RTH for {date_str} (09:30-16:00 ET)"
        )
        return None

    df_1s_bars = _build_1s_bars(scid_df, start_utc, end_utc)
    if df_1s_bars.is_empty():
        print("No data found for this timestamp")
        return None

    gex_file = f"/home/rwest/projects/data-pipeline/data/source/gexbot/NQ_NDX/gex_zero/{date_str}_NQ_NDX_classic_gex_zero.json"
    try:
        gex_df = load_gex_day_df(gex_file)
    except Exception as e:
        print(f"Warning: unable to load GEX for {date_str}: {e}")
        gex_df = None

    df_with_gex = _join_gex(df_1s_bars, gex_df).with_columns(
        pl.lit("MNQ").alias("symbol")
    )
    result_df = df_with_gex.with_columns(
        [
            pl.col("bar_timestamp").dt.epoch("s").alias("bar_timestamp"),
            pl.col("bar_timestamp").dt.epoch("s").alias("time_et"),
        ]
    )

    tick_columns = [
        "symbol",
        "bar_timestamp",
        "time_et",
        "Open",
        "High",
        "Low",
        "Close",
        "TotalVolume",
        "BidVolume",
        "AskVolume",
        "NumTrades",
    ]
    gex_columns = [col for col in result_df.columns if col not in tick_columns]
    result_df = result_df.select(tick_columns + gex_columns)

    out_dir = Path("data/enriched/MNQ")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mnq_{date_str}_{int(timestamp)}.parquet"
    result_df.write_parquet(out_path, compression="snappy")

    row = result_df.row(0)
    print(f"Bar for epoch {timestamp}:")
    for j, col_name in enumerate(result_df.columns):
        value = row[j]
        print(f"  {col_name}: {value}")

    print(f"Bars generated: {len(result_df)} | Columns: {len(result_df.columns)}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Enrich tick data with GEX integration (single second, day, or day range)"
    )
    parser.add_argument(
        "--scid-file",
        default="/mnt/c/SierraChart/Data/MNQZ25_FUT_CME.scid",
        help="Path to SCID file (default: MNQ Z25)",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--timestamp", type=float, help="Unix timestamp (seconds) to analyze"
    )
    mode.add_argument("--date", type=str, help="Single day YYYY-MM-DD")
    mode.add_argument(
        "--date-range",
        nargs=2,
        metavar=("START_DATE", "END_DATE"),
        help="Inclusive day range YYYY-MM-DD YYYY-MM-DD",
    )

    parser.add_argument(
        "--timestamp-date",
        type=str,
        help="Required when using --timestamp to locate matching GEX day (YYYY-MM-DD)",
    )

    args = parser.parse_args()

    scid_df = get_scid_df_cached(args.scid_file)

    if args.timestamp is not None:
        if not args.timestamp_date:
            parser.error("--timestamp-date is required when --timestamp is used")
        enrich_single_timestamp(
            args.timestamp, args.timestamp_date, scid_df, args.scid_file
        )
        return

    if args.date:
        enrich_day(args.date, scid_df, args.scid_file)
        return

    if args.date_range:
        start, end = args.date_range
        for day_str in _date_range(start, end):
            enrich_day(day_str, scid_df, args.scid_file)


if __name__ == "__main__":
    main()
