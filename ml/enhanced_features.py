#!/usr/bin/env python3
"""Enhanced feature engineering and data expansion for improved trading models.

Adds advanced technical indicators, market microstructure features, and expands dataset.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('output')
OUT_DIR = Path('output')

def add_advanced_technical_indicators(df):
    """Add advanced technical indicators beyond basic ones."""
    try:
        # Basic price and volume
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_ = df['open'].values
        volume = df['volume'].values

        # Trend indicators
        df['adx'] = talib.ADX(high, low, close, timeperiod=14)
        df['di_plus'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        df['di_minus'] = talib.MINUS_DI(high, low, close, timeperiod=14)

        # Momentum indicators
        df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close,
                                                   fastk_period=14, slowk_period=3, slowd_period=3)
        df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)

        # Volatility indicators
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        df['natr'] = talib.NATR(high, low, close, timeperiod=14)
        df['trange'] = talib.TRANGE(high, low, close)

        # Volume indicators
        df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        df['obv'] = talib.OBV(close, volume)

        # Simple moving averages
        df['sma_5'] = talib.SMA(close, timeperiod=5)
        df['sma_10'] = talib.SMA(close, timeperiod=10)
        df['sma_20'] = talib.SMA(close, timeperiod=20)

        # RSI
        df['rsi'] = talib.RSI(close, timeperiod=14)

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)

    except Exception as e:
        print(f"Warning: Error adding TA-Lib indicators: {e}")
        # Fallback to basic indicators
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi'] = 50  # Neutral RSI

    # Statistical features (always work)
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Rolling statistics
    for window in [5, 10, 20]:
        df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
        df[f'close_std_{window}'] = df['close'].rolling(window).std()
        df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
        if 'returns' in df.columns:
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()

    return df

def add_market_microstructure_features(df):
    """Add market microstructure features."""
    # Spread and liquidity features
    df['spread'] = df['high'] - df['low']
    df['spread_pct'] = df['spread'] / df['close']
    df['mid_price'] = (df['high'] + df['low']) / 2

    # Volume-weighted features
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['vwap_distance'] = (df['close'] - df['vwap']) / df['vwap']

    # Order flow indicators
    df['price_volume_trend'] = talib.AD(high=df['high'], low=df['low'],
                                       close=df['close'], volume=df['volume'])

    # Tick-based features (if available)
    if 'tick_type' in df.columns:
        df['trade_ratio'] = df.groupby(df['timestamp'].dt.date)['tick_type'].transform(
            lambda x: (x == 'trade').rolling(10).mean()
        )

    return df

def add_inter_market_features(df):
    """Add inter-market and external features."""
    # NQ correlation features (if nq_spot available)
    if 'nq_spot' in df.columns:
        df['nq_returns'] = df['nq_spot'].pct_change()
        df['nq_momentum'] = df['nq_spot'] - df['nq_spot'].shift(10)
        df['cross_momentum'] = df['returns'] - df['nq_returns']

    # GEX features (if available)
    if 'gex_zero' in df.columns:
        df['gex_change'] = df['gex_zero'].diff()
        df['gex_momentum'] = df['gex_zero'] - df['gex_zero'].shift(10)

    return df

def add_temporal_features(df):
    """Add time-based and seasonal features."""
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)

    # Intraday seasonality
    df['minutes_since_open'] = ((df['hour'] - 9) * 60 + df['minute']).clip(0, 390)

    # Sine/cosine transformations for cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df

def create_enhanced_features(df):
    """Create all enhanced features."""
    print("Adding advanced technical indicators...")
    df = add_advanced_technical_indicators(df)

    print("Adding market microstructure features...")
    df = add_market_microstructure_features(df)

    print("Adding inter-market features...")
    df = add_inter_market_features(df)

    print("Adding temporal features...")
    df = add_temporal_features(df)

    return df

def sliding_windows_enhanced(df: pd.DataFrame, features: list, window: int, stride: int = 1, horizon: int = 1):
    """Enhanced sliding window creation with better NaN handling."""
    arr = df[features].values
    n = arr.shape[0]

    # number of windows that have a future label at 'horizon' steps ahead
    num = n - window - horizon + 1
    if num <= 0:
        return np.empty((0, window, len(features)), dtype=np.float32)

    X = []
    valid_windows = 0

    for start in range(0, num, stride):
        window_data = arr[start:start + window]

        # Check for NaN values in window
        if not np.any(np.isnan(window_data)):
            X.append(window_data)
            valid_windows += 1

    print(f"Created {valid_windows} valid windows out of {num} possible")
    return np.stack(X, axis=0) if X else np.empty((0, window, len(features)), dtype=np.float32)

def process_enhanced_dataset(input_file, window=60, horizon=1, stride=1):
    """Process a single dataset with enhanced features."""
    print(f"\nProcessing {input_file}...")

    # Load data
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    # Add enhanced features
    df = create_enhanced_features(df)

    # Drop rows with NaN values
    initial_rows = len(df)
    df = df.dropna()
    print(f"Dropped {initial_rows - len(df)} rows with NaN values")

    if len(df) < window + horizon:
        print(f"Insufficient data after cleaning: {len(df)} rows")
        return None

    # Select features (exclude timestamp and non-numeric)
    exclude_cols = ['timestamp', 'tick_type', 'source', 'counter']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]

    print(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")

    # Scale features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Create sliding windows
    X = sliding_windows_enhanced(df, feature_cols, window, stride, horizon)

    if X.shape[0] == 0:
        print("No valid windows created")
        return None

    # Create labels
    returns = df['close'].pct_change(horizon).shift(-horizon)
    y = returns[window-1: window-1 + X.shape[0]]

    # Filter out NaN labels
    valid_mask = ~np.isnan(y.values)
    X = X[valid_mask]
    y = y.values[valid_mask]

    print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} timesteps, {X.shape[2]} features")

    return X, y, feature_cols

def expand_dataset_with_more_days():
    """Process additional days of data."""
    # Check available data files
    data_files = list(DATA_DIR.glob("MNQ_2025-11-*.parquet"))
    print(f"Found {len(data_files)} data files")

    all_X = []
    all_y = []
    feature_cols = None

    for file_path in sorted(data_files):
        result = process_enhanced_dataset(file_path)
        if result is not None:
            X, y, cols = result
            all_X.append(X)
            all_y.append(y)
            if feature_cols is None:
                feature_cols = cols

    if not all_X:
        print("No valid datasets created")
        return None

    # Combine all datasets
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    print(f"\nCombined dataset: {X_combined.shape[0]} samples, {X_combined.shape[2]} features")

    return X_combined, y_combined, feature_cols

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='output/MNQ_2025-11-11_1s.parquet',
                       help='Input parquet file')
    parser.add_argument('--output', default='output/MNQ_enhanced.npz',
                       help='Output NPZ file')
    parser.add_argument('--window', type=int, default=60, help='Window size')
    parser.add_argument('--horizon', type=int, default=1, help='Prediction horizon')
    parser.add_argument('--stride', type=int, default=1, help='Stride for windows')
    parser.add_argument('--expand_data', action='store_true',
                       help='Process all available days')
    args = parser.parse_args()

    if args.expand_data:
        print("Expanding dataset with all available days...")
        result = expand_dataset_with_more_days()
        if result is None:
            return
        X, y, feature_cols = result
        output_file = 'output/MNQ_multi_day_enhanced.npz'
    else:
        result = process_enhanced_dataset(args.input, args.window, args.horizon, args.stride)
        if result is None:
            return
        X, y, feature_cols = result
        output_file = args.output

    # Save enhanced dataset
    np.savez_compressed(output_file, X=X.astype(np.float32), y=y.astype(np.float32))
    print(f"Saved enhanced dataset to {output_file}")
    print(f"Shape: X={X.shape}, y={y.shape}")

    # Save feature list for reference
    with open(output_file.replace('.npz', '_features.txt'), 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"Feature list saved to {output_file.replace('.npz', '_features.txt')}")

    # Print feature statistics
    print("\nFeature Statistics:")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Categories:")
    categories = {}
    for col in feature_cols:
        category = col.split('_')[0] if '_' in col else 'other'
        categories[category] = categories.get(category, 0) + 1

    for cat, count in sorted(categories.items()):
        print(f"    {cat}: {count} features")

if __name__ == '__main__':
    main()