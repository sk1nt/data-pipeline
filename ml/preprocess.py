#!/usr/bin/env python3
"""Preprocess 1s bars into sliding-window dataset for supervised learning.

Inputs: `ml/data/{symbol}_{date}_1s.parquet`
Outputs: `ml/data/{symbol}_{date}_windows.npz` containing numpy arrays of X and y
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path('ml/data')
OUT_DIR = Path('ml/output')
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
    p.add_argument('--input', required=True, help='Path to 1s-parquet file')
    p.add_argument('--window', default=60, type=int)
    p.add_argument('--stride', default=1, type=int)
    p.add_argument('--horizon', default=1, type=int)
    p.add_argument('--features', default='open,high,low,close,volume')
    args = p.parse_args()

    df = pd.read_parquet(args.input)
    df = df.sort_values('timestamp').reset_index(drop=True)
    features = args.features.split(',')

    X = sliding_windows(df, features, args.window, args.stride, horizon=args.horizon)
    returns = labels_from_close(df['close'], horizon=args.horizon)
    # Build labels aligned with windows: y[i] corresponds to returns at index (i + window - 1)
    if X.shape[0] == 0:
        raise RuntimeError('Not enough data to create any windows with given window/horizon.')
    y = returns[args.window - 1: args.window - 1 + X.shape[0]]

    out_file = OUT_DIR / (Path(args.input).stem + f'_w{args.window}s_h{args.horizon}.npz')
    np.savez_compressed(out_file, X=X.astype(np.float32), y=y.to_numpy().astype(np.float32))
    print('Saved dataset:', out_file, 'X shape:', X.shape, 'y shape:', y.shape)
