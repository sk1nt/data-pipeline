#!/usr/bin/env python3
"""Extract 1-second OHLCV bars from tick Parquet and join to GEX snapshots.

Writes output to `ml/data/{symbol}_{date}_1s.parquet` and a simple CSV with sample counts.
"""
import argparse
from pathlib import Path
import duckdb
import os
from datetime import datetime

OUT_DIR = Path("ml/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_1s_bars(symbol: str, date: str, tick_parquet_root: str = 'data/parquet/tick'):
    # date in YYYY-MM-DD or YYYYMMDD form
    parsed = datetime.fromisoformat(date) if '-' in date else datetime.strptime(date, '%Y%m%d')
    file_date = parsed.strftime('%Y%m%d')
    parquet_path = os.path.join(tick_parquet_root, symbol, f"{file_date}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(parquet_path)

    out_path = OUT_DIR / f"{symbol}_{file_date}_1s.parquet"
    con = duckdb.connect()
    # Read symbol parquet and aggregate to 1-second OHLCV
    q = f"""
    WITH ticks AS (
        SELECT
            CAST(timestamp AS TIMESTAMP) AS ts,
            price,
            volume,
            to_timestamp(floor(extract(epoch from timestamp))) AS ts_s
        FROM read_parquet('{parquet_path}')
    ),
    agg AS (
        SELECT
            to_timestamp(floor(extract(epoch from ts))) AS ts_s,
            min(ts) AS first_ts,
            max(ts) AS last_ts,
            min(price) AS low,
            max(price) AS high,
            sum(volume) AS volume
        FROM ticks
        GROUP BY 1
        ORDER BY 1
    )
    SELECT a.ts_s as timestamp,
           f.price as open,
           a.high,
           a.low,
           l.price as close,
           a.volume
    FROM agg a
    LEFT JOIN ticks f ON a.ts_s = f.ts_s AND a.first_ts = f.ts
    LEFT JOIN ticks l ON a.ts_s = l.ts_s AND a.last_ts = l.ts
    ORDER BY a.ts_s
    """
    df = con.execute(q).fetchdf()
    if df is None or df.empty:
        raise RuntimeError('No ticks found for symbol/date parquet')

    # Write pandas DataFrame to parquet
    df.to_parquet(out_path, index=False)
    con.close()
    print('Wrote:', out_path)
    return out_path

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', required=True)
    p.add_argument('--date', required=True, help='YYYY-MM-DD or YYYYMMDD')
    args = p.parse_args()
    extract_1s_bars(args.symbol, args.date)
