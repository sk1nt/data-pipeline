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

def extract_1s_bars(symbol: str, date: str, tick_parquet_root: str = 'data/tick', gex_db: str = None, gex_ticker: str = 'NQ_NDX'):
    # date in YYYY-MM-DD or YYYYMMDD form
    parsed = datetime.fromisoformat(date) if '-' in date else datetime.strptime(date, '%Y%m%d')
    file_date = parsed.strftime('%Y%m%d')
    parquet_path = os.path.join(tick_parquet_root, symbol, f"{file_date}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(parquet_path)

    out_path = OUT_DIR / f"{symbol}_{file_date}_1s.parquet"
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
    {gex_select}
    FROM agg a
    LEFT JOIN ticks f ON a.ts_s = f.ts_s AND a.first_ts = f.ts
    LEFT JOIN ticks l ON a.ts_s = l.ts_s AND a.last_ts = l.ts
    {gex_join}
    ORDER BY a.ts_s
    """
    df = con.execute(q).fetchdf()
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
    args = p.parse_args()
    # allow comma-separated dates
    date_values = [d.strip() for d in args.date.split(',') if d.strip()]
    for d in date_values:
        try:
            extract_1s_bars(args.symbol, d, gex_db=args.gex_db, gex_ticker=args.gex_ticker)
        except FileNotFoundError:
            print("Missing parquet for", args.symbol, d)
