#!/usr/bin/env python3
"""Fix ts_ms values for files listed in verification CSV using our conversion script.

Usage:
  python3 scripts/fix_ts_ms_from_report.py verify_timestamps_report.csv --workers 3 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('csv', help='Path to verification CSV')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--python', default='python3')
    p.add_argument('--compression', default='snappy')
    p.add_argument('--atomic', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def run_convert(python, infile, compression='snappy', atomic=True):
    cmd = [python, 'scripts/convert_parquet_to_ts_ms.py', infile, '--out', infile]
    if compression:
        cmd.extend(['--compression', compression])
    if atomic:
        cmd.append('--atomic')
    return subprocess.call(cmd, env={**os.environ, 'PYTHONPATH': str(Path('.').resolve())})


def main():
    args = parse_args()
    to_process = []
    with open(args.csv, 'r', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            try:
                ts_mismatch = int(float(r.get('ts_ms_mismatch_rows', 0)))
            except Exception:
                ts_mismatch = 0
            has_ts_ms = (r.get('has_ts_ms', '').lower() == 'true')
            exists = (r.get('exists', '').lower() == 'true' or r.get('exists') == '1')
            path = r.get('path')
            if not path:
                continue
            if not exists:
                continue
            if ts_mismatch > 0 or not has_ts_ms:
                to_process.append(path)
    print(f'Found {len(to_process)} files to re-convert')
    if args.dry_run:
        for p in to_process[:50]:
            print('DRY:', p)
        return
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run_convert, args.python, p, args.compression, args.atomic) for p in to_process]
        rc = 0
        for f in futures:
            rc |= f.result()
    if rc == 0:
        print('All conversions finished successfully')
    else:
        print('Some conversion failed (rc != 0)')

if __name__ == '__main__':
    main()
