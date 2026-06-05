#!/usr/bin/env python3
"""
Fetch 1-minute OHLCV candles from TastyTrade/DXFeed for the last N trading
days and write them to Parquet files.

Output: data/parquet/candles/1m/<SYMBOL>/<YYYYMMDD>.parquet
Schema: symbol, timestamp, timestamp_ms, open, high, low, close, volume,
        bid_volume, ask_volume, vwap

Usage:
    python scripts/fetch_tt_candles.py
    python scripts/fetch_tt_candles.py --symbols MNQ NQ MES --days 2
    python scripts/fetch_tt_candles.py --symbols MNQ --days 5 --out data/parquet/candles/1m
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

from tastytrade import Session
from tastytrade.dxfeed.candle import Candle
from tastytrade.order import InstrumentType
from tastytrade.streamer import DXLinkStreamer
from tastytrade.instruments import Future


# ── Config ────────────────────────────────────────────────────────────────────

SYMBOLS = ["MNQ", "NQ", "MES"]   # product codes; front month resolved automatically

# How many seconds to wait after start_time before assuming all history arrived
DRAIN_SECONDS = 15


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trading_days_back(n: int) -> datetime:
    """Return UTC midnight of n trading days ago (skips weekends)."""
    day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    counted = 0
    while counted < n:
        day -= timedelta(days=1)
        if day.weekday() < 5:   # Mon–Fri
            counted += 1
    return day


def _resolve_front_month(session: Session, product_code: str) -> str:
    """Return the nearest tradeable front-month streamer symbol."""
    try:
        futures = Future.get(session, symbols=None, product_codes=[product_code])
        now = datetime.now(timezone.utc)
        roll_cutoff = (now + timedelta(days=5)).date()
        candidates = sorted(
            [f for f in futures if getattr(f, "is_tradeable", True)],
            key=lambda f: getattr(f, "expiration_date", datetime.max)
        )
        for f in candidates:
            exp = getattr(f, "expiration_date", None)
            if exp is None:
                continue
            if hasattr(exp, "date"):
                exp = exp.date()
            if exp >= roll_cutoff:
                sym = getattr(f, "streamer_symbol", None) or getattr(f, "symbol", "")
                return sym if sym.startswith("/") else f"/{sym}"
        # fallback: last candidate
        if candidates:
            sym = getattr(candidates[-1], "streamer_symbol", None) or getattr(candidates[-1], "symbol", "")
            return sym if sym.startswith("/") else f"/{sym}"
    except Exception as exc:
        print(f"  Warning: could not resolve front month for {product_code}: {exc}")
    # Heuristic fallback
    from datetime import date
    now = date.today()
    q = [("H", 3), ("M", 6), ("U", 9), ("Z", 12)]
    for code, month in q:
        if now.month <= month:
            return f"/{product_code}{code}{str(now.year)[-1]}"
    return f"/{product_code}H{str(now.year + 1)[-1]}"


# ── Main async fetch ──────────────────────────────────────────────────────────

async def fetch_candles(
    session: Session,
    streamer_symbols: dict[str, str],   # product_code -> streamer_symbol
    start_time: datetime,
    interval: str = "1m",
) -> dict[str, list[Candle]]:
    """Subscribe to candles and drain historical data."""
    results: dict[str, list[Candle]] = {s: [] for s in streamer_symbols}
    # Map streamer_symbol -> product_code for reverse lookup
    rev = {v: k for k, v in streamer_symbols.items()}

    async with DXLinkStreamer(session) as streamer:
        syms = list(streamer_symbols.values())
        print(f"  Subscribing to {syms} from {start_time.date()}...")
        await streamer.subscribe_candle(
            symbols=syms,
            interval=interval,
            start_time=start_time,
            extended_trading_hours=False,
        )

        # Collect for up to 45 seconds; all history arrives well within that
        hard_deadline = asyncio.get_event_loop().time() + 45
        while asyncio.get_event_loop().time() < hard_deadline:
            remaining = hard_deadline - asyncio.get_event_loop().time()
            try:
                candle = await asyncio.wait_for(
                    streamer.get_event(Candle), timeout=min(3.0, remaining)
                )
                raw_sym = (candle.event_symbol or "").split("{")[0]
                product = rev.get(raw_sym)
                if product is None:
                    for prod, ssym in streamer_symbols.items():
                        if ssym.split("{")[0] == raw_sym:
                            product = prod
                            break
                if product:
                    results[product].append(candle)
            except asyncio.TimeoutError:
                break
            except Exception as exc:
                print(f"  Warning: {exc}")
                break

    total = sum(len(v) for v in results.values())
    print(f"  Received {total} total candles")
    return results


# ── Parquet write ─────────────────────────────────────────────────────────────

def write_parquets(
    results: dict[str, list[Candle]],
    out_dir: Path,
    start_date: datetime,
) -> None:
    for product, candles in results.items():
        if not candles:
            print(f"  {product}: no candles received")
            continue

        rows = []
        for c in candles:
            ts_ms = c.time
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            if ts < start_date:
                continue
            rows.append({
                "symbol": product,
                "timestamp": ts,
                "timestamp_ms": ts_ms,
                "open": float(c.open or 0),
                "high": float(c.high or 0),
                "low": float(c.low or 0),
                "close": float(c.close or 0),
                "volume": float(c.volume or 0) if c.volume else 0.0,
                "bid_volume": float(c.bid_volume or 0) if c.bid_volume else 0.0,
                "ask_volume": float(c.ask_volume or 0) if c.ask_volume else 0.0,
                "vwap": float(c.vwap or 0) if c.vwap else None,
            })

        if not rows:
            print(f"  {product}: all candles filtered (before start_date)")
            continue

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["day"] = df["timestamp"].dt.strftime("%Y%m%d")

        for day, group in df.groupby("day"):
            day_dir = out_dir / product
            day_dir.mkdir(parents=True, exist_ok=True)
            path = day_dir / f"{day}.parquet"
            group.drop(columns=["day"]).to_parquet(path, index=False)
            print(f"  {product}/{day}.parquet  ({len(group)} bars)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch TastyTrade 1-min candles to Parquet")
    p.add_argument("--symbols", nargs="+", default=SYMBOLS, help="Product codes (e.g. MNQ NQ MES)")
    p.add_argument("--days", type=int, default=2, help="Number of trading days to fetch (default 2)")
    p.add_argument("--interval", default="1m", help="Candle interval (default 1m)")
    p.add_argument("--out", default=str(PROJECT_ROOT / "data/parquet/candles/1m"), help="Output directory")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)

    client_secret = os.getenv("TASTYTRADE_CLIENT_SECRET", "")
    refresh_token = os.getenv("TASTYTRADE_REFRESH_TOKEN", "")
    use_sandbox = os.getenv("TASTYTRADE_USE_SANDBOX", "true").lower() == "true"

    if not client_secret or not refresh_token:
        print("ERROR: TASTYTRADE_CLIENT_SECRET and TASTYTRADE_REFRESH_TOKEN must be set in .env")
        sys.exit(1)

    start_time = _trading_days_back(args.days)
    print(f"Fetching {args.interval} candles from {start_time.date()} for: {args.symbols}")

    print("Creating TastyTrade session...")
    session = Session(
        provider_secret=client_secret,
        refresh_token=refresh_token,
        is_test=use_sandbox,
    )

    print("Resolving front-month symbols...")
    streamer_symbols = {}
    for product in args.symbols:
        sym = _resolve_front_month(session, product)
        streamer_symbols[product] = sym
        print(f"  {product} → {sym}")

    candles = await fetch_candles(session, streamer_symbols, start_time, args.interval)
    write_parquets(candles, out_dir, start_time)
    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
