#!/usr/bin/env python3
"""
Worker to process a single day: parse depth snapshots and write a single parquet
per day containing top-K levels. This worker runs in "depth-only" mode: it
does not scan the SCID ticks. The schema includes `last_tick_*` columns set to
NULL so downstream steps can populate them if desired.

Usage:
  python3 scripts/worker_day.py --date 2025-11-07 --depth-dir /path/to/depth --parquet-dir data/parquet/depth --top-k 80
"""

import argparse
import sys
from pathlib import Path
import time
import os
import logging


def setup_sys_path():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def parse_args():
    p = argparse.ArgumentParser(description='Process one day of depth (depth-only)')
    p.add_argument('--date', required=True, help='Date YYYY-MM-DD')
    p.add_argument('--scid-file', required=False, default=None, help='Optional SCID file (ignored in depth-only mode)')
    p.add_argument('--depth-dir', required=True)
    p.add_argument('--parquet-dir', default='data/parquet/depth')
    p.add_argument('--top-k', type=int, default=80)
    p.add_argument('--chunk-size', type=int, default=10000, help='snapshots per parquet write')
    return p.parse_args()


def setup_logging(date_str: str):
    logs_dir = Path('logs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f'worker_{date_str}.log'
    logger = logging.getLogger(f'worker_{date_str}')
    logger.setLevel(logging.DEBUG)
    # avoid duplicate handlers
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
        logger.addHandler(ch)
    return logger


def main():
    setup_sys_path()
    args = parse_args()

    from src.lib.depth_parser import SierraChartDepthParser
    import polars as pl
    import pyarrow.parquet as pq

    date_str = args.date
    depth_path = os.path.join(args.depth_dir, f'MNQZ25_FUT_CME.{date_str}.depth')
    out_dir = Path(args.parquet_dir) / date_str.replace('-', '')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = str(out_dir / 'mnq_depth.parquet')

    parser = SierraChartDepthParser()
    logger = setup_logging(date_str)
    logger.info('Started worker for %s (top_k=%d, chunk=%d)', date_str, args.top_k, args.chunk_size)

    if not os.path.exists(depth_path):
        logger.warning('depth file not found: %s', depth_path)
        return

    logger.info('Streaming depth file and writing parquet: %s', depth_path)
    writer = None
    rows = []
    written = 0
    chunk = args.chunk_size
    t0 = time.time()
    parsed = 0
    for s in parser.parse_file(depth_path):
        parsed += 1
        if s.timestamp.strftime('%Y-%m-%d') != date_str:
            continue
        row = {'timestamp': s.timestamp}
        for i in range(args.top_k):
            if i < len(s.bids):
                lvl = s.bids[i]
                row[f'bid_price_{i+1}'] = lvl.price / 100.0
                row[f'bid_size_{i+1}'] = lvl.quantity
            else:
                row[f'bid_price_{i+1}'] = None
                row[f'bid_size_{i+1}'] = None
        for i in range(args.top_k):
            if i < len(s.asks):
                lvl = s.asks[i]
                row[f'ask_price_{i+1}'] = lvl.price / 100.0
                row[f'ask_size_{i+1}'] = lvl.quantity
            else:
                row[f'ask_price_{i+1}'] = None
                row[f'ask_size_{i+1}'] = None
        row['last_tick_ts'] = None
        row['last_tick_price'] = None
        row['last_tick_volume'] = None
        rows.append(row)
        if len(rows) >= chunk:
            tbl = pl.DataFrame(rows).to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(out_file, tbl.schema)
            writer.write_table(tbl)
            written += len(rows)
            logger.info('Wrote chunk, total rows written: %d', written)
            rows = []
    # flush
    if rows:
        tbl = pl.DataFrame(rows).to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(out_file, tbl.schema)
        writer.write_table(tbl)
        written += len(rows)
        rows = []
    if writer is not None:
        writer.close()
    dt = time.time() - t0
    logger.info('Finished writing %d rows to %s in %.2fs (parsed %d records)', written, out_file, dt, parsed)


if __name__ == '__main__':
    main()
