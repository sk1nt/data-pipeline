#!/usr/bin/env python3
"""
Orchestrator to run `worker_day.py` in parallel across a date range.

Usage example:
  python3 scripts/orchestrator.py --days 30 --workers 4 --scid-file /mnt/c/.../MNQZ25_FUT_CME.scid --depth-dir /mnt/c/.../MarketDepthData --parquet-dir data/parquet/depth

This will spawn up to `workers` processes concurrently. Each worker invokes the `worker_day.py` script.
"""

import argparse
from datetime import date, timedelta
import subprocess
import os
from pathlib import Path
import multiprocessing


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--days', type=int, default=30)
    p.add_argument('--workers', type=int, default=max(1, multiprocessing.cpu_count() // 2))
    p.add_argument('--scid-file', required=False, default=None)
    p.add_argument('--depth-dir', required=True)
    p.add_argument('--parquet-dir', default='data/parquet/depth')
    p.add_argument('--top-k', type=int, default=80)
    p.add_argument('--chunk-size', type=int, default=10000)
    p.add_argument('--python', default='python3', help='Python executable')
    return p.parse_args()


def run_day(python, date_str, args):
    cmd = [python, 'scripts/worker_day.py', '--date', date_str,
           '--depth-dir', args.depth_dir,
           '--parquet-dir', args.parquet_dir,
           '--top-k', str(args.top_k),
           '--chunk-size', str(args.chunk_size)]
    if args.scid_file:
        cmd.extend(['--scid-file', args.scid_file])

    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path('.').resolve())
    logs_dir = Path('logs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f'worker_{date_str}.log'
    print('Starting:', ' '.join(cmd), '->', log_file)

    # Stream subprocess stdout/stderr to per-day log file and to console
    with open(log_file, 'a') as fh:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
                print(f'[{date_str}]', line.rstrip())
        except Exception as e:
            fh.write(f'ERROR streaming output: {e}\n')
            raise
        rc = proc.wait()
    return rc


def main():
    args = parse_args()
    end = date.today() - timedelta(days=1)  # process up to yesterday
    start = end - timedelta(days=args.days - 1)
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime('%Y-%m-%d'))
        cur += timedelta(days=1)

    print(f'Processing {len(dates)} days using up to {args.workers} workers')

    # Simple worker pool using subprocesses
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(run_day, args.python, d, args): d for d in dates}
        for fut in concurrent.futures.as_completed(futures):
            d = futures[fut]
            rc = fut.result()
            if rc != 0:
                print(f'Day {d} failed with exit code {rc}')
            else:
                print(f'Day {d} completed')

if __name__ == '__main__':
    main()
