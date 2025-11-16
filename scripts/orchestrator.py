#!/usr/bin/env python3
"""Parallel driver for `worker_day.py` across an explicit date range."""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import os
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Roll schedule derived from the CME quarterly contract table documented in
# scripts/export_scid_ticks_to_parquet.py (see module docstring). MNQ rolls
# to the next contract at the Sunday 22:00 UTC session prior to the third
# Friday of March/June/September/December. For the historical window we're
# backfilling, MNQU25 covers through 2025-09-17, then MNQZ25 starts at the
# Globex session beginning 2025-09-18.
SCID_CONTRACT_WINDOWS = [
    (date(2025, 9, 2), date(2025, 9, 18), "MNQU25_FUT_CME.scid"),
    (date(2025, 9, 18), date(9999, 12, 31), "MNQZ25_FUT_CME.scid"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run worker_day in parallel over a date range")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() // 2))
    parser.add_argument("--python", default="python3", help="Python interpreter path")
    parser.add_argument("--symbol", default="MNQ", help="Canonical symbol")
    parser.add_argument("--scid-file", default=None, help="Fallback SCID file (used if --scid-dir not provided)")
    parser.add_argument("--scid-dir", default=None, help="Directory containing contract-specific SCID files")
    parser.add_argument("--depth-dir", required=True, help="Directory with .depth files")
    parser.add_argument("--depth-prefix", default="MNQZ25_FUT_CME", help="Depth filename prefix")
    parser.add_argument("--depth-parquet-dir", default="data/parquet/depth", help="Depth parquet root")
    parser.add_argument("--tick-parquet-dir", default="data/parquet/tick", help="Tick parquet root")
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--depth-chunk-size", type=int, default=100_000)
    parser.add_argument("--tick-chunk-size", type=int, default=50_000)
    parser.add_argument("--max-per-day", type=int, default=0)
    parser.add_argument("--emit-tick-parquet", action="store_true")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--worker-threads", type=int, default=2, help="Internal threads per worker_day")
    parser.add_argument("--convert-existing-parquet", action="store_true", help="Convert existing parquet files to normalized schema (add ts_ms)")
    parser.add_argument("--force-convert", action="store_true", help="Force conversion even when ts_ms column exists")
    parser.add_argument("--recreate-corrupt", action="store_true", help="If parquet is corrupted, re-run the worker to re-source that day")
    parser.add_argument("--convert-parquet-compression", default="zstd", help="Compression codec to use when rewriting parquet files")
    parser.add_argument("--parquet-compression", default="zstd", help="Parquet compression codec (snappy,zstd,gzip)")
    parser.add_argument("--parquet-compression-level", type=int, default=1, help="Parquet compression level (zstd levels 1-22)")
    parser.add_argument("--parquet-row-group-size", type=int, default=67108864, help="Parquet row group size in bytes (used to tune chunking)")
    parser.add_argument("--convert-timestamp-to-ms", action="store_true", help="Emit ts_ms epoch milliseconds column in Parquet instead of or in addition to timestamp")
    parser.add_argument("--atomic-writes", action="store_true", help="Write to temp file and atomically rename to destination once complete")
    parser.add_argument("--skip-existing", action="store_true", help="Skip processing if output file already exists")
    return parser.parse_args()


def daterange(start: datetime, end: datetime) -> List[str]:
    cur = start
    results: List[str] = []
    while cur <= end:
        results.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return results


def resolve_scid_for_date(target: date, args: argparse.Namespace) -> Path:
    """Select the correct SCID contract file for the requested date."""
    if args.scid_dir:
        root = Path(args.scid_dir)
        for start, end, filename in SCID_CONTRACT_WINDOWS:
            if start <= target < end:
                candidate = root / filename
                if candidate.exists():
                    return candidate
                raise FileNotFoundError(f"SCID file not found: {candidate}")
        raise ValueError(f"No SCID contract window covers {target}")
    if args.scid_file:
        return Path(args.scid_file)
    raise ValueError("Either --scid-dir or --scid-file must be provided")


def run_day(python: str, date_str: str, args: argparse.Namespace) -> int:
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    scid_path = resolve_scid_for_date(target_date, args)
    cmd = [
        python,
        "scripts/worker_day.py",
        "--date",
        date_str,
        "--symbol",
        args.symbol,
        "--scid-file",
        str(scid_path),
        "--depth-dir",
        args.depth_dir,
        "--depth-prefix",
        args.depth_prefix,
        "--depth-parquet-dir",
        args.depth_parquet_dir,
        "--tick-parquet-dir",
        args.tick_parquet_dir,
        "--top-k",
        str(args.top_k),
        "--depth-chunk-size",
        str(args.depth_chunk_size),
        "--tick-chunk-size",
        str(args.tick_chunk_size),
        "--max-per-day",
        str(args.max_per_day),
        "--log-dir",
        args.log_dir,
        "--threads",
        str(args.worker_threads),
    ]
    if args.emit_tick_parquet:
        cmd.append("--emit-tick-parquet")
    if args.parquet_compression:
        cmd.extend(["--parquet-compression", args.parquet_compression])
    if args.parquet_compression_level:
        cmd.extend(["--parquet-compression-level", str(args.parquet_compression_level)])
    if args.parquet_row_group_size:
        cmd.extend(["--parquet-row-group-size", str(args.parquet_row_group_size)])
    if args.convert_timestamp_to_ms:
        cmd.append("--convert-timestamp-to-ms")
    if args.atomic_writes:
        cmd.append("--atomic-writes")
    if args.skip_existing:
        cmd.append("--skip-existing")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(".").resolve())

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"worker_{date_str}.stdout.log"
    print("[orchestrator] starting", date_str)

    with open(log_file, "a") as fh:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
                print(f"[{date_str}] {line.rstrip()}")
        except Exception as exc:  # pragma: no cover - defensive
            fh.write(f"streaming error: {exc}\n")
            raise
        return proc.wait()


def _validate_parquet(path: Path) -> (bool, bool, str):
    """Validate parquet file with DuckDB.

    Returns tuple: (is_valid, has_ts_ms, error_message)
    """
    try:
        import duckdb
        con = duckdb.connect()
        # simple read to confirm file is readable
        rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]
        cols = con.execute(f"SELECT * FROM read_parquet('{path}') LIMIT 1").fetchdf().columns.tolist()
        has_ts_ms = 'ts_ms' in cols
        return True, has_ts_ms, ''
    except Exception as exc:
        return False, False, str(exc)


def run_convert_for_day(python: str, date_str: str, args: argparse.Namespace) -> int:
    """Convert existing parquet files (depth/tick) for a given day if required.

    If the parquet is corrupted and --recreate-corrupt is set, re-run the worker_day
    for that day to re-source raw files before conversion.
    """
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    # compute expected parquet paths
    out_depth = Path(args.depth_parquet_dir) / args.symbol / f"{date_str.replace('-', '')}.parquet"
    out_tick = Path(args.tick_parquet_dir) / args.symbol / f"{date_str.replace('-', '')}.parquet"
    rc_total = 0
    for p in (out_depth, out_tick):
        if not p.exists():
            print(f"[orchestrator] parquet not found for {p}, skipping.")
            continue
        valid, has_ts_ms, err = _validate_parquet(p)
        if not valid:
            print(f"[orchestrator] parquet corrupt: {p} err={err}")
            if args.recreate_corrupt:
                print(f"[orchestrator] re-running worker_day to re-source {date_str}")
                rc = run_day(python, date_str, args)
                if rc != 0:
                    print(f"[orchestrator] worker_day failed rc={rc} for {date_str}")
                    rc_total = rc_total or rc
                else:
                    # re-validate to detect corruption again
                    valid, has_ts_ms, err = _validate_parquet(p)
        if valid:
            if has_ts_ms and not args.force_convert:
                print(f"[orchestrator] parquet already contains ts_ms, skipping: {p}")
                continue
            # run conversion using our script
            cmd = [python, 'scripts/convert_parquet_to_ts_ms.py', str(p), '--out', str(p)]
            if args.convert_parquet_compression:
                cmd.extend(['--compression', args.convert_parquet_compression])
            if args.atomic_writes:
                cmd.append('--atomic')
            print(f"[orchestrator] converting parquet: {' '.join(cmd)}")
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path('.').resolve())
            if args.skip_existing:
                print(f"[orchestrator] skip-existing set; already handled previously")
            proc = subprocess.Popen(cmd, env=env)
            rc = proc.wait()
            if rc != 0:
                print(f"[orchestrator] conversion failed for {p} rc={rc}")
                rc_total = rc_total or rc
            else:
                print(f"[orchestrator] conversion succeeded for {p}")
    return rc_total


def main() -> None:
    args = parse_args()
    if not args.scid_dir and not args.scid_file:
        raise SystemExit("Provide --scid-dir for contract-aware rollover or --scid-file as a static fallback")
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    dates = daterange(start_dt, end_dt)
    print(f"[orchestrator] scheduling {len(dates)} days with {args.workers} workers")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        if args.convert_existing_parquet:
            # schedule convert tasks for each day
            future_map = {executor.submit(run_convert_for_day, args.python, d, args): d for d in dates}
        else:
            future_map = {executor.submit(run_day, args.python, d, args): d for d in dates}
        for fut in concurrent.futures.as_completed(future_map):
            day = future_map[fut]
            rc = fut.result()
            if rc != 0:
                print(f"[orchestrator] {day} FAILED rc={rc}")
            else:
                print(f"[orchestrator] {day} complete")


if __name__ == "__main__":
    main()
