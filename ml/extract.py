#!/usr/bin/env python3
"""Extract 1-second OHLCV bars from tick Parquet and join to GEX snapshots.

Writes output to `ml/output/{symbol}_{date}_1s.parquet` and a simple CSV with sample counts.
"""
import argparse
from pathlib import Path
import duckdb
import os
import json as _json
import pandas as pd
from datetime import datetime

OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_1s_bars(symbol: str, date: str, tick_parquet_root: str = 'data/tick', gex_db: str = None, gex_ticker: str = 'NQ_NDX', gex_json: str = None, bar_type: str = 'time', bar_size: int = 1):
    # date in YYYY-MM-DD or YYYYMMDD form
    parsed = datetime.fromisoformat(date) if '-' in date else datetime.strptime(date, '%Y%m%d')
    file_date = parsed.strftime('%Y%m%d')
    parquet_path = os.path.join(tick_parquet_root, symbol, f"{file_date}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(parquet_path)

    # Helper parsers for GEX nested fields
    def _parse_max_priors_map(v):
        names = ['max_current', 'max_1m', 'max_5m', 'max_10m', 'max_15m', 'max_30m']
        mapping = {n: None for n in names}
        if v is None:
            return mapping
        try:
            if isinstance(v, str):
                arr = _json.loads(v)
            else:
                arr = v
            if not isinstance(arr, list):
                return mapping
            rev = list(arr[::-1])
            for i, n in enumerate(names):
                try:
                    mapping[n] = rev[i][1] if len(rev[i]) >= 2 else None
                except Exception:
                    mapping[n] = None
        except Exception:
            pass
        return mapping

    def _derive_top_prior_vals(v):
        try:
            if isinstance(v, str):
                arr = _json.loads(v)
            else:
                arr = v
            vals = [p[1] for p in arr if isinstance(p, (list, tuple)) and len(p) >= 2 and isinstance(p[1], (int, float))]
            pos = max([vv for vv in vals if vv >= 0], default=None) if vals else None
            neg = min([vv for vv in vals if vv <= 0], default=None) if vals else None
            return pos, neg
        except Exception:
            return None, None

    def _derive_top_strike_features(v):
        try:
            if isinstance(v, str):
                arr = _json.loads(v)
            else:
                arr = v
            gammas = []
            ois = []
            for s in arr:
                if not isinstance(s, (list, tuple)):
                    continue
                if len(s) >= 3:
                    try:
                        gammas.append(float(s[2]))
                    except Exception:
                        pass
                if len(s) >= 2:
                    try:
                        ois.append(float(s[1]))
                    except Exception:
                        pass
            top_gamma = max([abs(g) for g in gammas], default=None) if gammas else None
            top_oi = max(ois, default=None) if ois else None
            return top_gamma, top_oi
        except Exception:
            return None, None
    # derived forward-fill columns for both gex_json and gex_db
    gex_derived_forward = ['max_current', 'max_1m', 'max_5m', 'max_10m', 'max_15m', 'max_30m']
    # helper to get closest strike candidates to target
    def _closest_strike_candidates(v, target, n=3, by='oi'):
        try:
            if isinstance(v, str):
                arr = _json.loads(v)
            else:
                arr = v
            if not arr:
                return []
            # build (abs distance, strike, oi, gamma)
            cands = []
            for s in arr:
                try:
                    strike_price = float(s[0])
                except Exception:
                    continue
                # choose metric by oi (default) or price
                metric = None
                if by == 'oi' and len(s) >= 2:
                    try:
                        metric = float(s[1])
                    except Exception:
                        metric = None
                if metric is None:
                    metric = strike_price
                try:
                    comp = float(target)
                except Exception:
                    comp = None
                if comp is None:
                    continue
                cands.append((abs(metric - comp), strike_price, s[1] if len(s) >= 2 else None, s[2] if len(s) >= 3 else None))
            cands.sort(key=lambda x: x[0])
            return cands[:n]
        except Exception:
            return []

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
    gex_select = ''
    if gex_db:
        # Discover available columns in gex_snapshots and build a safe select fragment
        try:
            df_gex_header = con.execute("SELECT * FROM gexdb.gex_snapshots LIMIT 0").fetchdf()
            gex_cols = set(df_gex_header.columns)
        except Exception:
            gex_cols = set()

        gex_select_parts = []
        if 'timestamp' in gex_cols:
            gex_select_parts.append('g.timestamp as gex_timestamp')
        if 'ticker' in gex_cols:
            # select gex ticker as `gex_ticker` to avoid conflating with the bar symbol
            gex_select_parts.append('g.ticker as gex_ticker')
        for c in ['spot_price', 'zero_gamma', 'net_gex', 'min_dte', 'sec_min_dte', 'major_pos_vol', 'major_pos_oi', 'major_neg_vol', 'major_neg_oi', 'sum_gex_vol', 'sum_gex_oi', 'delta_risk_reversal', 'max_priors', 'strikes']:
            if c in gex_cols:
                gex_select_parts.append(f'g.{c}')
        if gex_select_parts:
            gex_select = ',\n             ' + ',\n             '.join(gex_select_parts)
        gex_join = f"LEFT JOIN (SELECT *, to_timestamp(timestamp/1000) as gts FROM gexdb.gex_snapshots WHERE ticker = '{gex_ticker}') g ON a.ts_s = g.gts"
    # Determine which column holds volume in the parquet (some files use `size` instead of `volume`)
    try:
        df0 = con.execute(f"SELECT * FROM read_parquet('{parquet_path}') LIMIT 0").fetchdf()
        parquet_cols = set(df0.columns)
    except Exception:
        # fallback to reading with pyarrow if duckdb cannot fetch columns
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(parquet_path)
            parquet_cols = set(pf.schema.names)
        except Exception:
            parquet_cols = set()

    if 'volume' in parquet_cols:
        volume_col = 'volume'
    elif 'size' in parquet_cols:
        volume_col = 'size'
    else:
        raise RuntimeError(f'Parquet {parquet_path} missing volume/size column; found: {parquet_cols}')

    # For time bars we can use DuckDB aggregation; for volume/dollar bars, read into pandas and aggregate
    try:
        from ml.bar_builder import build_bars
    except Exception:
        # fall back to relative import when running in ml/ cwd
        from bar_builder import build_bars
    if bar_type == 'time':
        # use formula to group to larger time buckets if bar_size > 1
        ts_floor_expr = f"to_timestamp(floor(extract(epoch from ts) / {bar_size}) * {bar_size}) AS ts_s"
        q = f"""
        WITH ticks AS (
            SELECT
                CAST(timestamp AS TIMESTAMP) AS ts,
                price,
                {volume_col} as vol,
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
                sum(vol) AS volume,
                CASE WHEN sum(vol) = 0 THEN NULL ELSE sum(price * vol) / sum(vol) END AS vwap
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
             , a.vwap
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
        df_ticks = con.execute(f"SELECT CAST(timestamp AS TIMESTAMP) as timestamp, price, {volume_col} AS volume FROM read_parquet('{parquet_path}') ORDER BY timestamp").fetchdf()
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
        # Calculate VWAP per grouped bar
        df = grouped.agg({'timestamp': 'first', 'price': ['first', 'max', 'min', 'last'], 'volume': 'sum', 'price': lambda s: (s * df_ticks.loc[s.index, 'volume']).sum()})
        # The previous aggregation gives 'price' aggregated with custom; compute vwap by dividing by volume
        # Flatten multiindex
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df.reset_index(drop=True)
    if df is None or df.empty:
        raise RuntimeError('No ticks found for symbol/date parquet')

    # If the user requested volume or dollar bars, we can call the helper
    if bar_type in ('volume', 'dollar'):
        # Expect a `timestamp` (datetime), `price`, and `volume` columns
        # Build bars from raw ticks (df_ticks) rather than re-aggregating an already-aggregated dataframe
        if 'df_ticks' in locals():
            df = build_bars(df_ticks, bar_type, bar_size)
        else:
            df = build_bars(df, bar_type, bar_size)

    # Ensure the bar `ticker` (symbol for the parquet) is set on all outputs
    df['ticker'] = symbol

    # If gex values present, forward-fill missing values to align with tick seconds
    # Backfill/forward-fill and normalize GEX column names for downstream components.
    # If gex_json passed, merge raw snapshots into the pandas dataframe instead of using gex_db join
    if gex_json:
        try:
            from ml.gex_json_reader import load_gex_json
        except Exception:
            from gex_json_reader import load_gex_json
        gex_df = load_gex_json(gex_json, ticker=gex_ticker)
        if not gex_df.empty:
            # Align datetimes between bars and gex snaps; use naive UTC timestamps for match
            gex_df['gts'] = pd.to_datetime(gex_df['timestamp_ms'], unit='ms')
            # Ensure `gts` has the same timezone-awareness as bars to permit a safe merge
            if 'timestamp' in df.columns:
                bars_tz = getattr(df['timestamp'].dt, 'tz', None)
                if bars_tz is not None:
                    # bars are tz-aware: localize gts to UTC then convert to bars tz
                    if gex_df['gts'].dt.tz is None:
                        gex_df['gts'] = gex_df['gts'].dt.tz_localize('UTC').dt.tz_convert(bars_tz)
                    else:
                        gex_df['gts'] = gex_df['gts'].dt.tz_convert(bars_tz)
                else:
                    # bars are naive: ensure gts is naive
                    if gex_df['gts'].dt.tz is not None:
                        gex_df['gts'] = gex_df['gts'].dt.tz_convert('UTC').dt.tz_localize(None)
            bars_ts = pd.to_datetime(df['timestamp'])
            # Use left join on exact timestamps; then forward-fill GEX values
            # rename gex 'ticker' to 'gex_ticker' to avoid overwriting the bar symbol ticker
            if 'ticker' in gex_df.columns:
                gex_df = gex_df.rename(columns={'ticker': 'gex_ticker'})
            df = df.merge(gex_df.drop(columns=['timestamp_ms']), left_on='timestamp', right_on='gts', how='left')
            # Drop helper join column
            if 'gts' in df.columns:
                df.drop(columns=['gts'], inplace=True)
            # If json loader provided derived columns (top_prior_*, top_strike_*), keep them.
            # Also ensure max_priors/strikes are JSON-string values as expected downstream
            if 'max_priors' in df.columns:
                def _ensure_json_str(v):
                    if v is None:
                        return None
                    if isinstance(v, str):
                        return v
                    try:
                        return _json.dumps(v)
                    except Exception:
                        return str(v)
                df['max_priors'] = df['max_priors'].apply(_ensure_json_str)
            if 'strikes' in df.columns:
                df['strikes'] = df['strikes'].apply(_ensure_json_str)
            if 'max_priors' in df.columns:
                map_df = df['max_priors'].apply(_parse_max_priors_map).apply(pd.Series)
                for col in map_df.columns:
                    if col not in df.columns:
                        df[col] = map_df[col]
                    else:
                        df[col] = df[col].fillna(map_df[col])
            # forward fill any derived columns if present
            for c in gex_derived_forward:
                if c in df.columns:
                    df[c] = df[c].ffill()
            # Keep gex_ticker separate; the bar `ticker` is the input symbol (set below)
            # Compute candidate strikes for JSON join based on `strikes` list (closest to major_pos/major_neg)
            if 'strikes' in df.columns:
                # ensure 'strikes' is json string
                df['strikes'] = df['strikes'].apply(lambda v: v if isinstance(v, str) else _json.dumps(v) if v is not None else None)
                def _candidates_json_from_row(row, which):
                    try:
                        strikes_val = row.get('strikes')
                        if not strikes_val:
                            return _json.dumps([])
                        strikes_arr = _json.loads(strikes_val)
                        target = row.get(which)
                        if target is None:
                            return _json.dumps([])
                        by = 'oi' if which.endswith('_oi') else 'price'
                        c = _closest_strike_candidates(strikes_arr, target, by=by)
                        return _json.dumps([{'strike': x[1], 'oi': x[2], 'gamma': x[3]} for x in c])
                    except Exception:
                        return _json.dumps([])
                if 'major_pos_oi' in df.columns or 'major_pos_vol' in df.columns:
                    use_oi_pos = ('major_pos_oi' in df.columns and df['major_pos_oi'].notna().any())
                    which_pos = 'major_pos_oi' if use_oi_pos else 'major_pos_vol'
                    df['major_pos_candidates'] = df.apply(lambda r: _candidates_json_from_row(r, which_pos), axis=1)
                if 'major_neg_oi' in df.columns or 'major_neg_vol' in df.columns:
                    use_oi_neg = ('major_neg_oi' in df.columns and df['major_neg_oi'].notna().any())
                    which_neg = 'major_neg_oi' if use_oi_neg else 'major_neg_vol'
                    df['major_neg_candidates'] = df.apply(lambda r: _candidates_json_from_row(r, which_neg), axis=1)
                # generate numeric columns for top 3 candidates
                for i in range(3):
                    cpos = f'major_pos_candidate_{i+1}'
                    cneg = f'major_neg_candidate_{i+1}'
                    # Shorter names for compatibility: major_pos_can1 / major_neg_can1
                    cpos_short = f'major_pos_can{i+1}'
                    cneg_short = f'major_neg_can{i+1}'
                    if cpos not in df.columns:
                        df[cpos] = df['major_pos_candidates'].apply(lambda v: _json.loads(v)[i]['strike'] if v and len(_json.loads(v))>i else None)
                    if cpos_short not in df.columns:
                        df[cpos_short] = df[cpos]
                    if cneg not in df.columns:
                        df[cneg] = df['major_neg_candidates'].apply(lambda v: _json.loads(v)[i]['strike'] if v and len(_json.loads(v))>i else None)
                    if cneg_short not in df.columns:
                        df[cneg_short] = df[cneg]
    if gex_db:
        # the gex DB will have raw nested columns serialized as text; we may need to derive fields from them
        gex_fields = [
            'spot_price', 'zero_gamma', 'net_gex', 'min_dte', 'sec_min_dte',
            'major_pos_vol', 'major_pos_oi', 'major_neg_vol', 'major_neg_oi',
            'sum_gex_vol', 'sum_gex_oi', 'delta_risk_reversal', 'max_priors',
            'gex_timestamp', 'gex_ticker'
        ]
        # include derived columns for forward-fill if they exist
        # Forward fill relevant numeric columns; keep JSON/nested fields as-is (stringify if necessary)
        for c in gex_fields:
            if c in df.columns and c not in ('max_priors',):
                df[c] = df[c].ffill()
        for c in gex_derived_forward:
            if c in df.columns:
                df[c] = df[c].ffill()
        # Normalize aliases expected by preprocess/backtest
        if 'zero_gamma' in df.columns and 'gex_zero' not in df.columns:
            df['gex_zero'] = df['zero_gamma']
        if 'spot_price' in df.columns and 'nq_spot' not in df.columns:
            df['nq_spot'] = df['spot_price']
        # Ensure nested fields (max_priors) are JSON-string if not already - easier downstream
        if 'max_priors' in df.columns:
            def _to_json_or_none(v):
                if v is None:
                    return None
                if isinstance(v, (str, bytes)):
                    return v
                try:
                    return _json.dumps(v)
                except Exception:
                    return str(v)
            df['max_priors'] = df['max_priors'].apply(_to_json_or_none)
        # ensure `ticker` present and drop legacy alias if present
        # Set bar ticker to the input symbol (parquet file identifier) regardless of gex ticker
        if 'ticker' not in df.columns:
            df['ticker'] = symbol
        # Keep `gex_ticker` in place and do not override the bar `ticker` that we set above
        # Derive candidate strikes for DB join
        if 'strikes' in df.columns:
            df['strikes'] = df['strikes'].apply(_to_json_or_none)
            def _candidates_json_from_row_db(row, which):
                try:
                    strikes_val = row.get('strikes')
                    if not strikes_val:
                        return _json.dumps([])
                    strikes_arr = _json.loads(strikes_val) if isinstance(strikes_val, str) else strikes_val
                    target = row.get(which)
                    if target is None:
                        return _json.dumps([])
                    by = 'oi' if which.endswith('_oi') else 'price'
                    c = _closest_strike_candidates(strikes_arr, target, by=by)
                    return _json.dumps([{'strike': x[1], 'oi': x[2], 'gamma': x[3]} for x in c])
                except Exception:
                    return _json.dumps([])
            if 'major_pos_oi' in df.columns or 'major_pos_vol' in df.columns:
                use_oi_pos = ('major_pos_oi' in df.columns and df['major_pos_oi'].notna().any())
                which_pos = 'major_pos_oi' if use_oi_pos else 'major_pos_vol'
                df['major_pos_candidates'] = df.apply(lambda r: _candidates_json_from_row_db(r, which_pos), axis=1)
            if 'major_neg_oi' in df.columns or 'major_neg_vol' in df.columns:
                use_oi_neg = ('major_neg_oi' in df.columns and df['major_neg_oi'].notna().any())
                which_neg = 'major_neg_oi' if use_oi_neg else 'major_neg_vol'
                df['major_neg_candidates'] = df.apply(lambda r: _candidates_json_from_row_db(r, which_neg), axis=1)
            for i in range(3):
                cpos = f'major_pos_candidate_{i+1}'
                cneg = f'major_neg_candidate_{i+1}'
                cpos_short = f'major_pos_can{i+1}'
                cneg_short = f'major_neg_can{i+1}'
                if cpos not in df.columns:
                    df[cpos] = df['major_pos_candidates'].apply(lambda v: _json.loads(v)[i]['strike'] if v and len(_json.loads(v))>i else None)
                if cpos_short not in df.columns:
                    df[cpos_short] = df[cpos]
                if cneg not in df.columns:
                    df[cneg] = df['major_neg_candidates'].apply(lambda v: _json.loads(v)[i]['strike'] if v and len(_json.loads(v))>i else None)
                if cneg_short not in df.columns:
                    df[cneg_short] = df[cneg]
        # Derived helpers for max_priors and strikes are defined above and reused

        # Compute derived columns if present
        if 'max_priors' in df.columns:
            map_df = df['max_priors'].apply(_parse_max_priors_map).apply(pd.Series)
            for col in map_df.columns:
                if col not in df.columns:
                    df[col] = map_df[col]
                else:
                    df[col] = df[col].fillna(map_df[col])
        # top_prior_pos/top_prior_neg removed per rename/field removal requests
        # top_strike_gamma/top_strike_oi removed per rename/field removal requests

    # Drop raw nested `max_priors` field from final output; keep mapped max_* cols
    if 'max_priors' in df.columns:
        df = df.drop(columns=['max_priors'])
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
    p.add_argument('--gex-json', default=None, help='Path to raw GEX JSON file to read and join (optional)')
    p.add_argument('--bar-type', default='time', choices=['time', 'volume', 'dollar'], help='Bar type to build (time/volume/dollar)')
    p.add_argument('--bar-size', type=float, default=1, help='Bar-size: seconds for time bars, volume threshold for volume bars, dollar threshold for dollar bars')
    args = p.parse_args()
    # allow comma-separated dates
    date_values = [d.strip() for d in args.date.split(',') if d.strip()]
    for d in date_values:
        try:
            extract_1s_bars(args.symbol, d, tick_parquet_root='data/tick', gex_db=args.gex_db, gex_ticker=args.gex_ticker, gex_json=args.gex_json, bar_type=args.bar_type, bar_size=int(args.bar_size) if args.bar_type in ['time','volume'] else float(args.bar_size))
        except FileNotFoundError:
            print("Missing parquet for", args.symbol, d)
