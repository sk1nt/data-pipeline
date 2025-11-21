#!/usr/bin/env python3
"""Extract 1-second OHLCV bars from tick Parquet and join to GEX snapshots.

Writes output to `ml/output/{symbol}_{date}_1s.parquet` and a simple CSV with sample counts.
"""
import argparse
from pathlib import Path
import duckdb
import os
from datetime import datetime

OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_1s_bars(symbol: str, date: str, tick_parquet_root: str = 'data/tick', gex_db: str = None, gex_ticker: str = 'NQ_NDX', bar_type: str = 'time', bar_size: int = 1):
    # date in YYYY-MM-DD or YYYYMMDD form
    parsed = datetime.fromisoformat(date) if '-' in date else datetime.strptime(date, '%Y%m%d')
    file_date = parsed.strftime('%Y%m%d')
    parquet_path = os.path.join(tick_parquet_root, symbol, f"{file_date}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(parquet_path)

    # output filename encodes bar type/size; keep backward compatible default name
    out_name = f"{symbol}_{file_date}_1s.parquet" if bar_type == 'time' and bar_size == 1 else f"{symbol}_{file_date}_{bar_type}{bar_size}.parquet"
    out_path = OUT_DIR / out_name
    con = duckdb.connect()
    if gex_db:
        # attach the GEX DB to the duckdb connection so we can query gex_snapshots
        con.execute(f"ATTACH DATABASE '{gex_db}' AS gexdb")
    # Read symbol parquet and aggregate to 1-second OHLCV
    # Build optional join SQL for gex if requested
    gex_join = ''
    if gex_db:
        gex_join = f"LEFT JOIN (SELECT *, to_timestamp(timestamp/1000) as gts FROM gexdb.gex_snapshots WHERE ticker = '{gex_ticker}') g ON a.ts_s = g.gts"

    gex_select = ''
    if gex_db:
        gex_select = """
              , g.timestamp as gex_timestamp
              , g.ticker as gex_ticker
              , g.spot_price
              , g.zero_gamma
              , g.net_gex
              , g.min_dte
              , g.sec_min_dte
              , g.major_pos_vol
              , g.major_pos_oi
              , g.major_neg_vol
              , g.major_neg_oi
              , g.sum_gex_vol
              , g.sum_gex_oi
              , g.delta_risk_reversal
              , g.max_priors
"""
    # For time bars we can use DuckDB aggregation; for volume/dollar bars, read into pandas and aggregate
    if bar_type == 'time':
        # use formula to group to larger time buckets if bar_size > 1
        ts_floor_expr = f"to_timestamp(floor(extract(epoch from ts) / {bar_size}) * {bar_size}) AS ts_s"
        q = f"""
        WITH ticks AS (
            SELECT
                CAST(timestamp AS TIMESTAMP) AS ts,
                price,
                volume,
                {ts_floor_expr}
            FROM read_parquet('{parquet_path}')
        ),
        agg AS (
            SELECT
                ts_s,
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
        {gex_select}
        FROM agg a
        LEFT JOIN ticks f ON a.ts_s = f.ts_s AND a.first_ts = f.ts
        LEFT JOIN ticks l ON a.ts_s = l.ts_s AND a.last_ts = l.ts
        {gex_join}
        ORDER BY a.ts_s
        """
        df = con.execute(q).fetchdf()
    else:
        # Load into pandas and build volume/dollar bars
        df_ticks = con.execute(f"SELECT CAST(timestamp AS TIMESTAMP) as ts, price, volume FROM read_parquet('{parquet_path}') ORDER BY ts").fetchdf()
        if df_ticks is None or df_ticks.empty:
            raise RuntimeError('No ticks found for symbol/date parquet')
        import numpy as np
        if bar_type == 'volume':
            sizes = df_ticks['volume'].astype(int).values
            cum = sizes.cumsum()
            bar_id = ((cum - 1) // bar_size).astype(int)
        elif bar_type == 'dollar':
            dollars = (df_ticks['price'] * df_ticks['volume']).astype(float).values
            cum = dollars.cumsum()
            bar_id = ((cum - 1) // float(bar_size)).astype(int)
        else:
            raise ValueError(f'Unknown bar_type: {bar_type}')
        df_ticks['bar_id'] = bar_id
        grouped = df_ticks.groupby('bar_id')
        # Build OHLCV per bar
        df = grouped.agg({'ts': 'first', 'price': ['first', 'max', 'min', 'last'], 'volume': 'sum'})
        # Flatten multiindex
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df.reset_index(drop=True)
    if df is None or df.empty:
        raise RuntimeError('No ticks found for symbol/date parquet')

    # If gex values present, forward-fill missing values to align with tick seconds
    if 'gex_zero' in df.columns:
        df['gex_zero'] = df['gex_zero'].ffill()
    if 'nq_spot' in df.columns:
        df['nq_spot'] = df['nq_spot'].ffill()

    # Write pandas DataFrame to parquet
    df.to_parquet(out_path, index=False)
    con.close()
    print('Wrote:', out_path)
    return out_path

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', required=True)
    p.add_argument('--date', required=True, help='YYYY-MM-DD or YYYYMMDD; comma-separated for multiple dates')
    p.add_argument('--gex-db', default='data/gex_data.db', help='Path to data/gex_data.db to join GEX columns')
    p.add_argument('--gex-ticker', default='NQ_NDX', help='Ticker to join from gex snapshots (default NQ_NDX)')
    p.add_argument('--bar-type', default='time', choices=['time', 'volume', 'dollar'], help='Bar type to build (time/volume/dollar)')
    p.add_argument('--bar-size', type=float, default=1, help='Bar-size: seconds for time bars, volume threshold for volume bars, dollar threshold for dollar bars')
    args = p.parse_args()
    # allow comma-separated dates
    date_values = [d.strip() for d in args.date.split(',') if d.strip()]
    for d in date_values:
        try:
            extract_1s_bars(args.symbol, d, tick_parquet_root='data/tick', gex_db=args.gex_db, gex_ticker=args.gex_ticker, bar_type=args.bar_type, bar_size=int(args.bar_size) if args.bar_type in ['time','volume'] else float(args.bar_size))
        except FileNotFoundError:
            print("Missing parquet for", args.symbol, d)
