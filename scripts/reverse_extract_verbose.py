#!/usr/bin/env python3
"""
Verbose reverse extraction script

Usage (from repo root):
  python3 scripts/reverse_extract_verbose.py --start 2025-11-01 --end 2025-11-07 --clean

This script streams the SCID file backwards per-day, inserts ticks into DuckDB,
parses depth .depth files and writes per-day Parquet files with top-100 levels.
It prints verbose progress, batch insert rates, and per-day summaries.
"""

import argparse
import logging
import sys
import time
import os
from datetime import datetime, timedelta
from pathlib import Path


def setup_sys_path():
    # Ensure repo root is on sys.path so src imports work when run from anywhere
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def parse_args():
    p = argparse.ArgumentParser(description='Verbose reverse extraction for MNQ ticks and depth')
    p.add_argument('--scid-file', default='/mnt/c/SierraChart/Data/MNQZ25_FUT_CME.scid')
    p.add_argument('--depth-dir', default='/mnt/c/SierraChart/Data/MarketDepthData')
    p.add_argument('--start', required=True, help='Start date YYYY-MM-DD')
    p.add_argument('--end', required=True, help='End date YYYY-MM-DD')
    p.add_argument('--batch-size', type=int, default=5000, help='DB insert batch size')
    p.add_argument('--parquet-dir', default='data/depth_parquet', help='Parquet output directory')
    p.add_argument('--clean', action='store_true', help='Remove existing DB tables before running')
    p.add_argument('--max-per-day', type=int, default=0, help='Limit ticks per day (0 = unlimited)')
    p.add_argument('--top-k', type=int, default=80, help='Top K bid/ask levels to keep per snapshot')
    p.add_argument('--depth-chunk-size', type=int, default=100000, help='Number of snapshots per parquet chunk')
    p.add_argument('--emit-tick-parquet', action='store_true', help='Write ticks per-day to parquet instead of inserting into DuckDB')
    p.add_argument('--ticks-parquet-dir', default='data/parquet/ticks', help='Output dir for per-day ticks parquet files')
    p.add_argument('--tick-chunk-size', type=int, default=50000, help='Number of tick rows per parquet write chunk')
    p.add_argument('--debug', action='store_true', help='Enable debug logging')
    return p.parse_args()


def main():
    setup_sys_path()
    args = parse_args()
    # Configure logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger('reverse_extract')

    # Lazy imports (after sys.path setup)
    from src.lib.scid_parser import parse_scid_file_backwards_generator
    from src.lib.depth_parser import SierraChartDepthParser
    from src.lib.database import db as DB
    import polars as pl

    scid_file = args.scid_file
    depth_dir = args.depth_dir
    parquet_dir = Path(args.parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end, '%Y-%m-%d').date()

    conn = DB.connect()

    if args.clean:
        log.info('Cleaning existing mnq_ticks table (DROP)')
        try:
            conn.execute('DROP TABLE IF EXISTS mnq_ticks')
            DB._create_tables()
        except Exception:
            log.exception('Failed to clean DB tables')

    current = end_date
    try:
        while current >= start_date:
            day_str = current.strftime('%Y-%m-%d')
            log.info('Processing day %s', day_str)

            # parse depth snapshots (if present) and stream them into chunked parquet parts
            depth_path = os.path.join(depth_dir, f'MNQZ25_FUT_CME.{day_str}.depth')
            snaps_timestamps = []  # lightweight list of snapshot timestamps (for interleaving)
            day_out_dir = parquet_dir / current.strftime('%Y%m%d')
            day_out_dir.mkdir(parents=True, exist_ok=True)
            part_idx = 0
            rows_chunk = []
            chunk_size = args.depth_chunk_size
            top_k = args.top_k

            if os.path.exists(depth_path):
                log.info('  Streaming depth file into chunks: %s -> %s', depth_path, day_out_dir)
                parser = SierraChartDepthParser()
                t0 = time.time()
                try:
                    import pyarrow.parquet as pq
                    writer = None
                    # single parquet file per day inside the day directory
                    out_file = str(day_out_dir / 'mnq_depth.parquet')
                    for s in parser.parse_file(depth_path):
                        if s.timestamp.date() != current:
                            continue
                        snaps_timestamps.append(s.timestamp)
                        row = {'timestamp': s.timestamp}
                        # only keep top_k levels
                        for i in range(top_k):
                            if i < len(s.bids):
                                lvl = s.bids[i]
                                row[f'bid_price_{i+1}'] = lvl.price / 100.0
                                row[f'bid_size_{i+1}'] = lvl.quantity
                            else:
                                row[f'bid_price_{i+1}'] = None
                                row[f'bid_size_{i+1}'] = None
                        for i in range(top_k):
                            if i < len(s.asks):
                                lvl = s.asks[i]
                                row[f'ask_price_{i+1}'] = lvl.price / 100.0
                                row[f'ask_size_{i+1}'] = lvl.quantity
                            else:
                                row[f'ask_price_{i+1}'] = None
                                row[f'ask_size_{i+1}'] = None
                        rows_chunk.append(row)
                        if len(rows_chunk) >= chunk_size:
                            tbl = pl.DataFrame(rows_chunk).to_arrow()
                            if writer is None:
                                writer = pq.ParquetWriter(out_file, tbl.schema)
                            writer.write_table(tbl)
                            log.info('  Appended chunk to %s (%d rows)', out_file, len(rows_chunk))
                            rows_chunk = []
                            part_idx += 1
                except Exception:
                    log.exception('  Error while streaming depth file')
                # flush remaining
                if rows_chunk:
                    tbl = pl.DataFrame(rows_chunk).to_arrow()
                    if writer is None:
                        writer = pq.ParquetWriter(out_file, tbl.schema)
                    writer.write_table(tbl)
                    log.info('  Appended final chunk to %s (%d rows)', out_file, len(rows_chunk))
                    rows_chunk = []
                    part_idx += 1
                if writer is not None:
                    writer.close()

                parse_time = time.time() - t0
                log.info('  Parsed and wrote %d snapshots for %s (%.2fs) -> %s', len(snaps_timestamps), day_str, parse_time, out_file)
                # prepare timestamps newest -> oldest
                snaps_timestamps.reverse()
            else:
                log.info('  Depth file not found for %s: %s', day_str, depth_path)

            # If no snapshots, simple tick-only processing
            if not snaps_timestamps:
                log.info('  No depth snapshots; processing ticks only for %s', day_str)
                buffer = []
                tick_count = 0
                inserted = 0
                start_t = time.time()
                for record in parse_scid_file_backwards_generator(scid_file, date_filter=current, max_records=(args.max_per_day or None)):
                    ts = record['timestamp']
                    price = float(record['close'])
                    volume = int(record.get('total_volume', 0))
                    buffer.append((ts, price, volume, None, 'MNQ'))
                    tick_count += 1
                    if args.max_per_day and tick_count >= args.max_per_day:
                        break
                    if len(buffer) >= args.batch_size:
                        t_ins = time.time()
                        conn.executemany('INSERT INTO mnq_ticks (timestamp, price, volume, tick_type, ticker) VALUES (?, ?, ?, ?, ?)', buffer)
                        dt = time.time() - t_ins
                        inserted += len(buffer)
                        log.info('  Inserted batch %d rows (%.2f rows/s)', len(buffer), len(buffer) / max(dt, 1e-6))
                        buffer = []
                if buffer:
                    t_ins = time.time()
                    conn.executemany('INSERT INTO mnq_ticks (timestamp, price, volume, tick_type, ticker) VALUES (?, ?, ?, ?, ?)', buffer)
                    dt = time.time() - t_ins
                    inserted += len(buffer)
                    log.info('  Inserted final batch %d rows (%.2f rows/s)', len(buffer), len(buffer) / max(dt, 1e-6))
                else:
                    # Interleave ticks with depth snapshots using lightweight timestamps (newest->oldest)
                    snap_idx = 0
                    buffered = []
                    tick_count = 0
                    inserted = 0
                    start_t = time.time()

                    # snaps_timestamps is newest->oldest
                    for record in parse_scid_file_backwards_generator(scid_file, date_filter=current, max_records=(args.max_per_day or None)):
                        ts = record['timestamp']
                        price = float(record['close'])
                        volume = int(record.get('total_volume', 0))

                        # advance snapshots while the record is older than the current snapshot
                        while snap_idx < len(snaps_timestamps) and ts < snaps_timestamps[snap_idx]:
                            if buffered:
                                t_ins = time.time()
                                conn.executemany('INSERT INTO mnq_ticks (timestamp, price, volume, tick_type, ticker) VALUES (?, ?, ?, ?, ?)', buffered)
                                dt = time.time() - t_ins
                                inserted += len(buffered)
                                log.info('  Inserted %d buffered ticks for snapshot %d (%.2f rows/s)', len(buffered), snap_idx + 1, len(buffered) / max(dt, 1e-6))
                                buffered = []
                            snap_idx += 1

                        buffered.append((ts, price, volume, None, 'MNQ'))
                        tick_count += 1

                        if args.max_per_day and tick_count >= args.max_per_day:
                            break

                        if len(buffered) >= args.batch_size:
                            t_ins = time.time()
                            conn.executemany('INSERT INTO mnq_ticks (timestamp, price, volume, tick_type, ticker) VALUES (?, ?, ?, ?, ?)', buffered)
                            dt = time.time() - t_ins
                            inserted += len(buffered)
                            log.info('  Inserted batch %d rows (%.2f rows/s)', len(buffered), len(buffered) / max(dt, 1e-6))
                            buffered = []

                    if buffered:
                        t_ins = time.time()
                        conn.executemany('INSERT INTO mnq_ticks (timestamp, price, volume, tick_type, ticker) VALUES (?, ?, ?, ?, ?)', buffered)
                        dt = time.time() - t_ins
                        inserted += len(buffered)
                        log.info('  Inserted final buffered %d rows (%.2f rows/s)', len(buffered), len(buffered) / max(dt, 1e-6))

                    elapsed = time.time() - start_t
                    log.info('  Day %s: ticks processed=%d, inserted=%d, time=%.1fs, rate=%.1f rows/s', day_str, tick_count, inserted, elapsed, (inserted / max(elapsed, 1e-6)))

                    # Register depth metadata: path is the directory containing part files
                    if snaps_timestamps:
                        record_count = len(snaps_timestamps)
                        date_range_start = min(snaps_timestamps)
                        date_range_end = max(snaps_timestamps)
                        metadata_path = out_file
                        conn.execute('INSERT OR REPLACE INTO mnq_depth_metadata (file_path, date_range_start, date_range_end, record_count) VALUES (?, ?, ?, ?)', [metadata_path, date_range_start, date_range_end, record_count])
                        log.info('  Registered depth parquet %s rows=%d range=%s->%s', metadata_path, record_count, date_range_start, date_range_end)
                    log.info('  Wrote depth parquet %s (%d rows) range=%s -> %s', out_file, record_count, date_range_start, date_range_end)

            # move to previous day (newest -> oldest)
            current = current - timedelta(days=1)

    except KeyboardInterrupt:
        log.warning('Interrupted by user; exiting gracefully')

    log.info('Finished')


if __name__ == '__main__':
    main()
