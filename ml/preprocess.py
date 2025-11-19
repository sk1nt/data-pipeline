#!/usr/bin/env python3
"""Preprocess 1s bars into sliding-window dataset for supervised learning.

Inputs: `output/{symbol}_{date}_1s.parquet`
Outputs: `output/{symbol}_{date}_windows.npz` containing numpy arrays of X and y
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path('output')
OUT_DIR = Path('output')
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


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True, help='Path(s) to 1s-parquet file(s); comma-separated or glob')
    p.add_argument('--window', default=60, type=int)
    p.add_argument('--stride', default=1, type=int)
    p.add_argument('--horizon', default=1, type=int)
    p.add_argument('--features', default='open,high,low,close,volume,gex_zero,nq_spot')
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
    for f in files:
        if not Path(f).exists():
            raise FileNotFoundError(f)
        df_list.append(pd.read_parquet(f))
    df = pd.concat(df_list, ignore_index=True)
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
    df = df.dropna(subset=subset_cols, how='any')
    if len(features) == 0:
        raise RuntimeError('No requested features are present in inputs')

    X = sliding_windows(df, features, args.window, args.stride, horizon=args.horizon)
    # Build labels from label-source
    if args.label_source == 'close':
        returns = labels_from_close(df['close'], horizon=args.horizon)
    else:
        # labels from arbitrary numeric series: future - current normalized by current
        if args.label_source not in df.columns:
            raise RuntimeError(f'label_source {args.label_source} not found in input data')
        future = df[args.label_source].shift(-args.horizon)
        returns = ((future - df[args.label_source]) / df[args.label_source])[:-args.horizon]
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
