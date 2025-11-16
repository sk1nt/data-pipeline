#!/usr/bin/env python3
"""Per-day worker that processes both depth (.depth) and tick (.scid) files.

The worker ingests an entire trading day, emitting Parquet slices under the
canonical layout:

- Depth:  data/parquet/depth/<SYMBOL>/<YYYYMMDD>.parquet
- Ticks:  data/parquet/tick/<SYMBOL>/<YYYYMMDD>.parquet

Each invocation is independent, making it suitable for parallel execution via
``scripts/orchestrator.py``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def setup_sys_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a single trading day of MNQ depth + ticks")
    parser.add_argument("--date", required=True, help="Trading date YYYY-MM-DD")
    parser.add_argument("--symbol", default="MNQ", help="Canonical symbol (used in output paths)")
    parser.add_argument("--scid-file", required=True, help="Path to SCID file covering the date")
    parser.add_argument("--depth-dir", required=True, help="Directory containing .depth files")
    parser.add_argument("--depth-prefix", default="MNQZ25_FUT_CME", help="Depth filename prefix before .YYYY-MM-DD.depth")
    parser.add_argument("--depth-parquet-dir", default="data/parquet/depth", help="Depth parquet root")
    parser.add_argument("--tick-parquet-dir", default="data/parquet/tick", help="Tick parquet root")
    parser.add_argument("--top-k", type=int, default=80, help="Number of bid/ask levels to retain")
    parser.add_argument("--depth-chunk-size", type=int, default=100_000, help="Snapshots per depth parquet chunk flush")
    parser.add_argument("--tick-chunk-size", type=int, default=50_000, help="Tick rows per parquet chunk flush")
    parser.add_argument("--max-per-day", type=int, default=0, help="Optional tick cap per day (0 = unlimited)")
    parser.add_argument("--emit-tick-parquet", action="store_true", help="Write tick parquet slices (always on)")
    parser.add_argument("--log-dir", default="logs", help="Directory for per-day worker logs")
    return parser.parse_args()


def setup_logger(date_str: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"worker_{date_str}.log"
    logger = logging.getLogger(f"worker_{date_str}")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(console)
    return logger


def _depth_file(depth_dir: Path, prefix: str, date_str: str) -> Path:
    direct = depth_dir / f"{prefix}.{date_str}.depth"
    if direct.exists():
        return direct
    matches = sorted(depth_dir.glob(f"{prefix.split('_')[0]}*.{date_str}.depth"))
    if matches:
        return matches[0]
    return direct


def write_depth_parquet(
    depth_path: Path,
    symbol: str,
    date_str: str,
    out_root: Path,
    top_k: int,
    chunk_size: int,
    logger: logging.Logger,
) -> int:
    from src.lib.depth_parser import SierraChartDepthParser
    import polars as pl
    import pyarrow.parquet as pq

    if not depth_path.exists():
        logger.warning("Depth file not found: %s", depth_path)
        return 0

    out_file = out_root / symbol / f"{date_str.replace('-', '')}.parquet"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    parser = SierraChartDepthParser()
    rows: List[dict] = []
    writer = None
    snapshots = 0
    start = time.time()
    for snapshot in parser.parse_file(str(depth_path)):
        if snapshot.timestamp.strftime("%Y-%m-%d") != date_str:
            continue
        record = {"timestamp": snapshot.timestamp}
        for idx in range(top_k):
            if idx < len(snapshot.bids):
                level = snapshot.bids[idx]
                record[f"bid_price_{idx+1}"] = level.price / 100.0
                record[f"bid_size_{idx+1}"] = level.quantity
            else:
                record[f"bid_price_{idx+1}"] = None
                record[f"bid_size_{idx+1}"] = None
        for idx in range(top_k):
            if idx < len(snapshot.asks):
                level = snapshot.asks[idx]
                record[f"ask_price_{idx+1}"] = level.price / 100.0
                record[f"ask_size_{idx+1}"] = level.quantity
            else:
                record[f"ask_price_{idx+1}"] = None
                record[f"ask_size_{idx+1}"] = None
        rows.append(record)
        snapshots += 1
        if len(rows) >= chunk_size:
            tbl = pl.DataFrame(rows).to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(out_file, tbl.schema)
            writer.write_table(tbl)
            rows.clear()
    if rows:
        tbl = pl.DataFrame(rows).to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(out_file, tbl.schema)
        writer.write_table(tbl)
        rows.clear()
    if writer is not None:
        writer.close()
    duration = time.time() - start
    logger.info("Depth: wrote %d snapshots to %s in %.2fs", snapshots, out_file, duration)
    return snapshots


def write_tick_parquet(
    scid_file: Path,
    symbol: str,
    date_str: str,
    out_root: Path,
    chunk_size: int,
    max_per_day: int,
    logger: logging.Logger,
) -> int:
    from src.lib.scid_parser import parse_scid_file_backwards_generator
    import polars as pl
    import pyarrow.parquet as pq

    if not scid_file.exists():
        logger.error("SCID file missing: %s", scid_file)
        return 0

    out_file = out_root / symbol / f"{date_str.replace('-', '')}.parquet"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    writer = None
    ticks = 0
    start = time.time()
    date_filter = datetime.strptime(date_str, "%Y-%m-%d")
    limit = max_per_day if max_per_day > 0 else None
    for record in parse_scid_file_backwards_generator(str(scid_file), date_filter=date_filter, max_records=limit):
        ts = record["timestamp"]
        if ts.strftime("%Y-%m-%d") != date_str:
            continue
        rows.append(
            {
                "timestamp": ts,
                "price": float(record["close"]),
                "volume": int(record.get("total_volume", 0)),
            }
        )
        ticks += 1
        if len(rows) >= chunk_size:
            tbl = pl.DataFrame(rows).to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(out_file, tbl.schema)
            writer.write_table(tbl)
            rows.clear()
    if rows:
        tbl = pl.DataFrame(rows).to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(out_file, tbl.schema)
        writer.write_table(tbl)
        rows.clear()
    if writer is not None:
        writer.close()
    duration = time.time() - start
    logger.info("Ticks: wrote %d rows to %s in %.2fs", ticks, out_file, duration)
    return ticks


def main() -> None:
    setup_sys_path()
    args = parse_args()

    date_str = args.date
    symbol = args.symbol.upper()
    log_dir = Path(args.log_dir)
    logger = setup_logger(date_str, log_dir)
    logger.info(
        "Worker start symbol=%s scid=%s depth_dir=%s",
        symbol,
        args.scid_file,
        args.depth_dir,
    )

    depth_dir = Path(args.depth_dir)
    depth_path = _depth_file(depth_dir, args.depth_prefix, date_str)
    depth_written = write_depth_parquet(
        depth_path=depth_path,
        symbol=symbol,
        date_str=date_str,
        out_root=Path(args.depth_parquet_dir),
        top_k=args.top_k,
        chunk_size=args.depth_chunk_size,
        logger=logger,
    )

    tick_written = 0
    if args.emit_tick_parquet:
        tick_written = write_tick_parquet(
            scid_file=Path(args.scid_file),
            symbol=symbol,
            date_str=date_str,
            out_root=Path(args.tick_parquet_dir),
            chunk_size=args.tick_chunk_size,
            max_per_day=args.max_per_day,
            logger=logger,
        )

    logger.info(
        "Worker complete %s: depth_rows=%d, tick_rows=%d",
        date_str,
        depth_written,
        tick_written,
    )


if __name__ == "__main__":
    main()
