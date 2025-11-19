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
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List


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
    parser.add_argument("--threads", type=int, default=0, help="Internal thread count (0 = auto)")
    parser.add_argument("--parquet-compression", default="zstd", help="Parquet compression codec (snappy,zstd,gzip)")
    parser.add_argument("--parquet-compression-level", type=int, default=1, help="Parquet compression level (if supported by codec)")
    parser.add_argument("--parquet-row-group-size", type=int, default=0, help="Target parquet row group size in bytes (0 = default chunking)")
    parser.add_argument("--max-memory-mb", type=int, default=0, help="Optional: maximum memory in MB to allow before flushing (0 = disabled)")
    parser.add_argument("--convert-timestamp-to-ms", action="store_true", help="Add ts_ms epoch milliseconds column (recommended) and keep timestamp")
    parser.add_argument("--timestamp-tz", default="UTC", help="Timezone name for naive timestamps (e.g. UTC, America/New_York). If timestamp has tzinfo, it will be respected.")
    parser.add_argument("--atomic-writes", action="store_true", help="Write to a temporary file and then atomically rename to final path on success")
    parser.add_argument("--skip-existing", action="store_true", help="Skip writing if the target parquet file already exists")
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
    parquet_compression: str = "zstd",
    parquet_compression_level: int = 1,
    atomic_writes: bool = False,
    skip_existing: bool = False,
    convert_timestamp_to_ms: bool = False,
    max_memory_mb: int = 0,
    timestamp_tz: str = "UTC",
) -> int:
    from src.lib.depth_parser import SierraChartDepthParser
    import polars as pl
    import pyarrow.parquet as pq

    if not depth_path.exists():
        logger.warning("Depth file not found: %s", depth_path)
        return 0

    out_file = out_root / symbol / f"{date_str.replace('-', '')}.parquet"
    if skip_existing and out_file.exists():
        logger.info("Depth: skipping existing file %s", out_file)
        return 0
    tmp_file = out_file.with_suffix(".parquet.tmp")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    parser = SierraChartDepthParser()
    rows: List[dict] = []
    writer = None
    snapshots = 0
    start = time.time()
    # Safety cap on chunk size to avoid excessive memory usage.
    MAX_DEPTH_CHUNK = 200_000
    if chunk_size > MAX_DEPTH_CHUNK:
        logger.warning(
            "Depth chunk_size %d is greater than MAX %d; capping to %d",
            chunk_size,
            MAX_DEPTH_CHUNK,
            MAX_DEPTH_CHUNK,
        )
        chunk_size = MAX_DEPTH_CHUNK
    # Memory-based safety: if available memory is below threshold, flush rows.
    try:
        import psutil  # optional dependency for memory checks
    except Exception:
        psutil = None
    if convert_timestamp_to_ms:
        logger.info("Depth: convert timestamps to ms using tz=%s", timestamp_tz)
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
        if convert_timestamp_to_ms:
            # Robust conversion: support datetime or numeric epoch (seconds or ms)
            val = snapshot.timestamp
            if isinstance(val, (int, float)):
                v = int(val)
                if v >= 10**12:
                    record["ts_ms"] = v
                elif v >= 10**9:
                    # value in seconds
                    record["ts_ms"] = int(v * 1000)
                else:
                    # small number, assume seconds
                    record["ts_ms"] = int(v * 1000)
            elif isinstance(val, datetime):
                # If the datetime is naive (no tzinfo), localize it with the supplied
                # timestamp_tz. Then convert to epoch ms. If it's timezone aware,
                # Python's timestamp() will return epoch seconds in UTC.
                if val.tzinfo is None:
                    try:
                        from zoneinfo import ZoneInfo

                        tz = ZoneInfo(timestamp_tz)
                        val = val.replace(tzinfo=tz)
                    except Exception:
                        # If ZoneInfo is not available or the tz name invalid,
                        # fallback to naive timestamp behavior (UTC assumption).
                        pass
                record["ts_ms"] = int(val.timestamp() * 1000)
        rows.append(record)
        snapshots += 1
        # Throttle / flush when too many rows accumulated or memory low.
        mem_low = False
        if psutil and max_memory_mb and max_memory_mb > 0:
            try:
                avail_mb = psutil.virtual_memory().available // (1024 * 1024)
                if avail_mb < max_memory_mb:
                    mem_low = True
            except Exception:
                mem_low = False
        if len(rows) >= chunk_size or mem_low:
            tbl = pl.DataFrame(rows)
            arrow_tbl = tbl.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(tmp_file, arrow_tbl.schema, compression=parquet_compression)
            writer.write_table(arrow_tbl)
            rows.clear()
            if snapshots % 100000 == 0:
                logger.info("Depth: processed %d snapshots", snapshots)
    if rows:
        tbl = pl.DataFrame(rows)
        arrow_tbl = tbl.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(tmp_file, arrow_tbl.schema, compression=parquet_compression)
        writer.write_table(arrow_tbl)
        rows.clear()
    if writer is not None:
        writer.close()
    if atomic_writes and tmp_file.exists():
        tmp_file.replace(out_file)
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
    parquet_compression: str = "zstd",
    parquet_compression_level: int = 1,
    atomic_writes: bool = False,
    skip_existing: bool = False,
    convert_timestamp_to_ms: bool = False,
    max_memory_mb: int = 0,
    timestamp_tz: str = "UTC",
) -> int:
    from src.lib.scid_parser import parse_scid_file_backwards_generator
    import polars as pl
    import pyarrow.parquet as pq

    if not scid_file.exists():
        logger.error("SCID file missing: %s", scid_file)
        return 0

    out_file = out_root / symbol / f"{date_str.replace('-', '')}.parquet"
    if skip_existing and out_file.exists():
        logger.info("Ticks: skipping existing file %s", out_file)
        return 0
    tmp_file = out_file.with_suffix(".parquet.tmp")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    writer = None
    ticks = 0
    start = time.time()
    # Safety cap on chunk size to avoid excessive memory usage.
    MAX_TICK_CHUNK = 200_000
    if chunk_size > MAX_TICK_CHUNK:
        logger.warning(
            "Tick chunk_size %d is greater than MAX %d; capping to %d",
            chunk_size,
            MAX_TICK_CHUNK,
            MAX_TICK_CHUNK,
        )
        chunk_size = MAX_TICK_CHUNK
    try:
        import psutil  # optional dependency for memory checks
    except Exception:
        psutil = None
    date_filter = datetime.strptime(date_str, "%Y-%m-%d")
    limit = max_per_day if max_per_day > 0 else None
    if convert_timestamp_to_ms:
        logger.info("Ticks: convert timestamps to ms using tz=%s", timestamp_tz)
    for record in parse_scid_file_backwards_generator(str(scid_file), date_filter=date_filter, max_records=limit):
        ts = record["timestamp"]
        if ts.strftime("%Y-%m-%d") != date_str:
            continue
        record_row = {
            "timestamp": ts,
            "price": float(record["close"]),
            "volume": int(record.get("total_volume", 0)),
        }
        if convert_timestamp_to_ms:
            # ts may be datetime or numeric
            val = ts
            if isinstance(val, (int, float)):
                v = int(val)
                if v >= 10**12:
                    record_row["ts_ms"] = v
                elif v >= 10**9:
                    record_row["ts_ms"] = int(v * 1000)
                else:
                    record_row["ts_ms"] = int(v * 1000)
            elif isinstance(val, datetime):
                if val.tzinfo is None:
                    try:
                        from zoneinfo import ZoneInfo

                        tz = ZoneInfo(timestamp_tz)
                        val = val.replace(tzinfo=tz)
                    except Exception:
                        pass
                record_row["ts_ms"] = int(val.timestamp() * 1000)
        rows.append(record_row)
        ticks += 1
        if ticks == 1:
            logger.debug("Ticks sample first: %s", record_row)
        mem_low = False
        if psutil and max_memory_mb and max_memory_mb > 0:
            try:
                avail_mb = psutil.virtual_memory().available // (1024 * 1024)
                if avail_mb < max_memory_mb:
                    mem_low = True
            except Exception:
                mem_low = False
        if len(rows) >= chunk_size or mem_low:
            tbl = pl.DataFrame(rows)
            arrow_tbl = tbl.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(tmp_file, arrow_tbl.schema, compression=parquet_compression)
            writer.write_table(arrow_tbl)
            rows.clear()
            if ticks % 100000 == 0:
                logger.info("Ticks: processed %d ticks", ticks)
    if rows:
        tbl = pl.DataFrame(rows)
        arrow_tbl = tbl.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(tmp_file, arrow_tbl.schema, compression=parquet_compression)
        writer.write_table(arrow_tbl)
        rows.clear()
    if writer is not None:
        writer.close()
    if atomic_writes and tmp_file.exists():
        tmp_file.replace(out_file)
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
    # Log whether tick emission is enabled for diagnostic purposes
    logger.info("emit_tick_parquet=%s", args.emit_tick_parquet)

    depth_dir = Path(args.depth_dir)
    depth_path = _depth_file(depth_dir, args.depth_prefix, date_str)
    results = {}

    def _run_depth():
        results["depth"] = write_depth_parquet(
            depth_path=depth_path,
            symbol=symbol,
            date_str=date_str,
            out_root=Path(args.depth_parquet_dir),
            top_k=args.top_k,
            chunk_size=args.depth_chunk_size,
            logger=logger,
            parquet_compression=args.parquet_compression,
            parquet_compression_level=args.parquet_compression_level,
            atomic_writes=args.atomic_writes,
            skip_existing=args.skip_existing,
            convert_timestamp_to_ms=args.convert_timestamp_to_ms,
            max_memory_mb=args.max_memory_mb,
            timestamp_tz=args.timestamp_tz,
        )

    def _run_ticks():
        if not args.emit_tick_parquet:
            results["ticks"] = 0
            return
        results["ticks"] = write_tick_parquet(
            scid_file=Path(args.scid_file),
            symbol=symbol,
            date_str=date_str,
            out_root=Path(args.tick_parquet_dir),
            chunk_size=args.tick_chunk_size,
            max_per_day=args.max_per_day,
            logger=logger,
            parquet_compression=args.parquet_compression,
            parquet_compression_level=args.parquet_compression_level,
            atomic_writes=args.atomic_writes,
            skip_existing=args.skip_existing,
            convert_timestamp_to_ms=args.convert_timestamp_to_ms,
            max_memory_mb=args.max_memory_mb,
            timestamp_tz=args.timestamp_tz,
        )

    max_workers = 2 if args.emit_tick_parquet else 1
    threads = args.threads if args.threads > 0 else max_workers
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(_run_depth)]
        if args.emit_tick_parquet:
            futures.append(executor.submit(_run_ticks))
        for fut in futures:
            fut.result()

    depth_written = results.get("depth", 0)
    tick_written = results.get("ticks", 0)

    logger.info(
        "Worker complete %s: depth_rows=%d, tick_rows=%d",
        date_str,
        depth_written,
        tick_written,
    )


if __name__ == "__main__":
    main()
