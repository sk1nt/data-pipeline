#!/usr/bin/env python3
"""Preprocess 1s bars into sliding-window dataset for supervised learning.

Inputs: `output/{symbol}_{date}_1s.parquet`
Outputs: `output/{symbol}_{date}_windows.npz` containing numpy arrays of X and y
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from ml.path_utils import resolve_cli_path
except Exception:
    from path_utils import resolve_cli_path

DATA_DIR = resolve_cli_path('output')
OUT_DIR = resolve_cli_path('output')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sliding_windows(df: pd.DataFrame, features: list, window: int, stride: int = 1, horizon: int = 1):
    arr = df[features].values
    n = arr.shape[0]
    # number of windows that have a future label at 'horizon' steps ahead
    num = n - window - horizon + 1
    if num <= 0:
        return np.empty((0, window, len(features)), dtype=np.float32)
    X = []
    for start in range(0, num, stride):
        X.append(arr[start:start + window])
    return np.stack(X, axis=0)


def labels_from_close(df: pd.Series, horizon: int = 1):
    # regression: next-horizon return; classification: sign
    future = df.shift(-horizon)
    returns = (future - df) / df
    return returns[:-horizon]


def williams_r(high, low, close, period=14):
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll)


def rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(close, fast=12, slow=26, signal=9):
    fast_ema = close.ewm(span=fast).mean()
    slow_ema = close.ewm(span=slow).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line


def bollinger_bands(close, period=20, std=2):
    sma = close.rolling(period).mean()
    std_dev = close.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True, help='Path(s) to 1s-parquet file(s); comma-separated or glob')
    p.add_argument('--window', default=60, type=int)
    p.add_argument('--stride', default=1, type=int)
    p.add_argument('--horizon', default=1, type=int)
    p.add_argument('--bar-type', default='time', choices=['time', 'volume', 'dollar'], help='Type of bars used in inputs (time/volume/dollar). If not time, run `ml/extract.py` to produce bar datasets. Preprocess currently expects 1s or bar parquet files.')
    p.add_argument('--bar-size', default=1, type=float, help='Bar size parameter (seconds for time; volume threshold for volume, dollar threshold for dollar bars).')
    p.add_argument('--features', default='open,high,low,close,volume,gex_zero,nq_spot,williams_r,rsi,macd,macd_signal,bb_upper,bb_lower,vwap,bb3_upper,bb3_lower')
    p.add_argument('--label-source', default='close', choices=['close','nq_spot','gex_zero'], help='Use this column to build labels instead of close')
    args = p.parse_args()
    # Resolve inputs - can be comma-separated list or glob
    import glob
    input_patterns = [p.strip() for p in args.inputs.split(',')]
    files = []
    for pat in input_patterns:
        if '*' in pat or '?' in pat or '[' in pat:
            files.extend(glob.glob(pat))
        else:
            files.append(pat)
    # Read and concat in timestamp order
    df_list = []
    # Resolve input files in a repo-root-aware manner
    try:
        from ml.path_utils import resolve_cli_path
    except Exception:
        # Running from within `ml/` (cwd='ml') - import as local module
        from path_utils import resolve_cli_path
    for f in files:
        p = resolve_cli_path(f)
        print(f"[preprocess] resolve_cli_path: '{f}' -> '{p}' (exists={p.exists()})")
        if not p.exists():
            raise FileNotFoundError(f)
        # If inputs point to raw tick parquet and user requested non-time bars, remind to run extract first
        if args.bar_type != 'time' and 'tick' in str(p):
            print('[preprocess] Warning: inputs appear to be raw tick parquet. For volume/dollar bars please run `ml/extract.py --bar-type', args.bar_type, '--bar-size', args.bar_size, '` first.')
        df_list.append(pd.read_parquet(p))
    df = pd.concat(df_list, ignore_index=True)
    # Normalize GEX/spot aliases for downstream processing
    # Some extraction paths write 'zero_gamma'/'spot_price' while tests expect 'gex_zero'/'nq_spot'
    if 'zero_gamma' in df.columns and 'gex_zero' not in df.columns:
        df['gex_zero'] = df['zero_gamma']
    if 'spot_price' in df.columns and 'nq_spot' not in df.columns:
        df['nq_spot'] = df['spot_price']
    # Compute technical indicators
    df['williams_r'] = williams_r(df['high'], df['low'], df['close'])
    df['rsi'] = rsi(df['close'])
    df['macd'], df['macd_signal'] = macd(df['close'])
    df['bb_upper'], df['bb_lower'] = bollinger_bands(df['close'])
    # Add 3-sigma bollinger bands
    df['bb3_upper'], df['bb3_lower'] = bollinger_bands(df['close'], std=3)
    # Ensure vwap exists if not provided by extractor; compute a simple cumulative vwap as fallback
    if 'vwap' not in df.columns:
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    features = [f.strip() for f in args.features.split(',') if f.strip()]
    df = df.sort_values('timestamp').reset_index(drop=True)
    # Keep only features that exist in the dataframe
    existing_features = [f for f in features if f in df.columns]
    missing = set(features) - set(existing_features)
    if missing:
        print('Warning: missing features in parquet and will be skipped:', missing)
    features = existing_features
    # Drop rows where requested features or the label source have NaNs to avoid NaNs in windows
    subset_cols = features[:]  # feature subset for dropna
    if args.label_source in df.columns:
        subset_cols.append(args.label_source)
    else:
        # allow label-source aliases mapping
        if args.label_source == 'nq_spot' and 'spot_price' in df.columns:
            subset_cols.append('spot_price')
        if args.label_source == 'gex_zero' and 'zero_gamma' in df.columns:
            subset_cols.append('zero_gamma')
    df = df.dropna(subset=subset_cols, how='any')
    if len(features) == 0:
        raise RuntimeError('No requested features are present in inputs')

    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    X = sliding_windows(df, features, args.window, args.stride, horizon=args.horizon)
    # Build labels from label-source
    if args.label_source == 'close':
        returns = labels_from_close(df['close'], horizon=args.horizon)
    else:
        # labels from arbitrary numeric series: future - current normalized by current
        # map aliases
        label_col = args.label_source
        if args.label_source == 'nq_spot' and 'nq_spot' not in df.columns and 'spot_price' in df.columns:
            label_col = 'spot_price'
        if args.label_source == 'gex_zero' and 'gex_zero' not in df.columns and 'zero_gamma' in df.columns:
            label_col = 'zero_gamma'
        if label_col not in df.columns:
            raise RuntimeError(f'label_source {args.label_source} not found in input data')
        future = df[label_col].shift(-args.horizon)
        returns = ((future - df[label_col]) / df[label_col])[:-args.horizon]
    # Build labels aligned with windows: y[i] corresponds to returns at index (i + window - 1)
    if X.shape[0] == 0:
        raise RuntimeError('Not enough data to create any windows with given window/horizon.')
    y = returns[args.window - 1: args.window - 1 + X.shape[0]]
    # Filter NaN labels and corresponding windows
    import numpy as np
    y_arr = np.asarray(y)
    valid_mask = ~np.isnan(y_arr)
    X = X[valid_mask]
    y = y_arr[valid_mask]

    # Name output based on first/last file stem
    base_name = Path(files[0]).stem if files else Path(args.inputs).stem
    out_file = OUT_DIR / (base_name + f'_w{args.window}s_h{args.horizon}.npz')
    np.savez_compressed(out_file, X=X.astype(np.float32), y=np.asarray(y).astype(np.float32))
    print('Saved dataset:', out_file, 'X shape:', X.shape, 'y shape:', y.shape)
