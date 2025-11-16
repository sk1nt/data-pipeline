#!/usr/bin/env python3
"""Parallel driver for `worker_day.py` across an explicit date range."""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run worker_day in parallel over a date range")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() // 2))
    parser.add_argument("--python", default="python3", help="Python interpreter path")
    parser.add_argument("--symbol", default="MNQ", help="Canonical symbol")
    parser.add_argument("--scid-file", required=True, help="Path to SCID file")
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
    return parser.parse_args()


def daterange(start: datetime, end: datetime) -> List[str]:
    cur = start
    results: List[str] = []
    while cur <= end:
        results.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return results


def run_day(python: str, date_str: str, args: argparse.Namespace) -> int:
    cmd = [
        python,
        "scripts/worker_day.py",
        "--date",
        date_str,
        "--symbol",
        args.symbol,
        "--scid-file",
        args.scid_file,
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
    ]
    if args.emit_tick_parquet:
        cmd.append("--emit-tick-parquet")

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


def main() -> None:
    args = parse_args()
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    dates = daterange(start_dt, end_dt)
    print(f"[orchestrator] scheduling {len(dates)} days with {args.workers} workers")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
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
