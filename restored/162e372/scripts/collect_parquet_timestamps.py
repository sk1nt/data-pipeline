#!/usr/bin/env python3
"""
Collect and print all unique timestamps from depth parquet files in a directory.
Usage:
  python3 scripts/collect_parquet_timestamps.py --parquet-dir data/parquet/depth
"""
import argparse
from pathlib import Path
import polars as pl

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--parquet-dir', required=True)
    return p.parse_args()

def main():
    args = parse_args()
    pdir = Path(args.parquet_dir)
    files = list(pdir.rglob('mnq_depth.parquet'))
    if not files:
        print('No depth parquet files found in', pdir)
        return
    all_timestamps = set()
    for f in files:
        print(f'Processing {f}...')
        try:
            df = pl.read_parquet(f, columns=['timestamp'])
            ts = df['timestamp'].to_list()
            all_timestamps.update(ts)
            print(f'  Found {len(ts)} timestamps')
        except Exception as e:
            print(f'  Error reading {f}: {e}')
    print(f'Total unique timestamps: {len(all_timestamps)}')
    # Print sorted sample
    sorted_ts = sorted(all_timestamps)
    for t in sorted_ts[:20]:
        print(' ', t)
    if len(sorted_ts) > 20:
        print('...')
        print('Last:', sorted_ts[-1])

if __name__ == '__main__':
    main()
