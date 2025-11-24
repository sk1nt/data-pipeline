#!/usr/bin/env python3
"""Optimize prediction thresholds and position sizing for maximum P&L.

Tests different thresholds and position sizing strategies on existing models.
"""
import argparse
import numpy as np
import torch
import joblib
from pathlib import Path
import mlflow
import mlflow.pytorch
import mlflow.lightgbm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from datetime import datetime
import polars as pl

# Model paths
LSTM_MODEL = Path('ml/models/lstm_large_dropout.pt')
LSTM_CV_MODELS = [Path(f'ml/models/lstm_cv_fold{i}.pt') for i in range(5)]
LGB_MODEL = Path('ml/models/lightgbm_tuned.pkl')

# Dataset - can be .npz or directory with .parquet files
DATASET = 'ml/output'  # Directory with parquet files

def load_enhanced_gex_data():
    """Load enhanced GEX data for alignment."""
    try:
        # Try to load from the data directory
        gex_files = list(Path('data').glob('*gex*.parquet')) + list(Path('data').glob('*GEX*.parquet'))
        if gex_files:
            df = pd.read_parquet(gex_files[0])
            df = df.sort_values('timestamp').drop_duplicates('timestamp')
            df = df.set_index('timestamp', drop=False)
            return df
        
        # Fallback: create empty dataframe with expected columns
        print("Warning: No GEX data found, using empty alignment")
        empty_df = pd.DataFrame(columns=['spot_price', 'zero_gamma', 'net_gex', 'major_pos_vol', 'major_neg_vol', 
                                       'sum_gex_vol', 'delta_risk_reversal', 'max_priors_current', 'max_priors_1m', 'max_priors_5m'])
        return empty_df
        
    except Exception as e:
        print(f"Warning: Could not load enhanced GEX data: {e}")
        empty_df = pd.DataFrame(columns=['spot_price', 'zero_gamma', 'net_gex', 'major_pos_vol', 'major_neg_vol', 
                                       'sum_gex_vol', 'delta_risk_reversal', 'max_priors_current', 'max_priors_1m', 'max_priors_5m'])
        return empty_df

def create_enhanced_features(df):
    """Create enhanced technical features for the model."""
    # Basic returns and volatility
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Moving averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    
    # Exponential moving averages
    df['ema_5'] = df['close'].ewm(span=5).mean()
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Stochastic Oscillator
    low_min = df['low'].rolling(14).min()
    high_max = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # Williams %R
    df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)
    
    # Average True Range (ATR)
    tr = np.maximum(df['high'] - df['low'], 
                   np.maximum(abs(df['high'] - df['close'].shift(1)),
                             abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()
    
    # Commodity Channel Index (CCI)
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(20).mean()
    mad_tp = (tp - sma_tp).abs().rolling(20).mean()
    df['cci'] = (tp - sma_tp) / (0.015 * mad_tp)
    
    # Momentum
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    
    # Rate of Change
    df['roc_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['roc_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    # Volume indicators
    df['volume_sma_5'] = df['volume'].rolling(5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_5']
    
    # Volatility
    df['volatility_5'] = df['returns'].rolling(5).std()
    df['volatility_10'] = df['returns'].rolling(10).std()
    df['volatility_20'] = df['returns'].rolling(20).std()
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(0)
    
    return df

def align_with_enhanced_gex(df_tick):
    """Align tick data with enhanced GEX data."""
    try:
        # Load enhanced GEX data (cached to avoid reloading)
        if not hasattr(align_with_enhanced_gex, '_gex_cache'):
            align_with_enhanced_gex._gex_cache = load_enhanced_gex_data()
        
        df_gex = align_with_enhanced_gex._gex_cache
        
        # Ensure tick data timestamps are tz-naive UTC for alignment
        if hasattr(df_tick.index, 'tz') and df_tick.index.tz is not None:
            tick_timestamps_utc = df_tick.index.tz_convert('UTC').tz_localize(None)
        else:
            tick_timestamps_utc = df_tick.index
        
        # Forward fill GEX data to align with tick timestamps
        gex_cols = ['spot_price', 'zero_gamma', 'net_gex', 'major_pos_vol', 'major_neg_vol', 
                   'sum_gex_vol', 'delta_risk_reversal', 'max_priors_current', 'max_priors_1m', 'max_priors_5m']
        
        df_gex_aligned = df_gex[gex_cols].reindex(tick_timestamps_utc, method='ffill')
        
        # Merge with tick data
        for col in gex_cols:
            df_tick[col] = df_gex_aligned[col].values
            
        # Fill any remaining NaN with 0
        df_tick[gex_cols] = df_tick[gex_cols].fillna(0)
        
        return df_tick
        
    except Exception as e:
        print(f"Warning: Could not align GEX data: {e}")
        # Add default GEX columns
        gex_cols = ['spot_price', 'zero_gamma', 'net_gex', 'major_pos_vol', 'major_neg_vol', 
                   'sum_gex_vol', 'delta_risk_reversal', 'max_priors_current', 'max_priors_1m', 'max_priors_5m']
        for col in gex_cols:
            if col not in df_tick.columns:
                df_tick[col] = 0.0
        return df_tick

def preprocess_day_data(parquet_file, scaler, sequence_length=60, required_feature_count=None, stride=1, max_samples=None, use_enhanced_features=True, model_type='sklearn'):
    """Preprocess a single day's data for prediction."""
    # Load data with pandas (since GEX alignment functions expect pandas)
    df = pd.read_parquet(parquet_file)
    
    # Handle different data formats
    if 'price' in df.columns and 'timestamp' in df.columns:
        # Tick data format - resample to 1-second OHLCV
        df = df.set_index('timestamp', drop=False)
        df = df.resample('1s').agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df = df.dropna()
        # Ensure 'timestamp' column exists for enhanced feature functions
        if 'timestamp' not in df.columns and df.index.name == 'timestamp':
            df['timestamp'] = df.index
        
        # Add missing columns with default values
        df['gex_zero'] = 0.0
        df['nq_spot'] = 0.0
        
        # Try to align with enhanced GEX data
        df = align_with_enhanced_gex(df)
        
    elif not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'TotalVolume']):
        print(f"Warning: Missing required OHLCV columns in {parquet_file}")
        return None, None
    else:
        print("Processing as processed data format")
        # Processed data format - rename columns to lowercase
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'TotalVolume': 'volume',
            'spot': 'spot_price'
        })
        print(f"After renaming: {list(df.columns[:10])}")
        
        # For 38-feature models, skip GEX alignment since processed data already has GEX features
        if not use_enhanced_features:
            # Processed data already has GEX features, just ensure proper column names
            gex_col_mapping = {
                'zero_gamma': 'zero_gamma',
                'net_gex': 'net_gex', 
                'major_pos_vol': 'major_pos_vol',
                'major_neg_vol': 'major_neg_vol',
                'sum_gex_vol': 'sum_gex_vol',
                'delta_risk_reversal': 'delta_risk_reversal',
                'max_priors_current': 'max_priors_current',
                'max_priors_1m': 'max_priors_1m', 
                'max_priors_5m': 'max_priors_5m'
            }
            for new_col, existing_col in gex_col_mapping.items():
                if existing_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[existing_col]
                elif new_col not in df.columns:
                    df[new_col] = 0.0
        else:
            # If data is already in 1s OHLCV form, still align with enhanced GEX features
            # to ensure features like max_priors_current, max_priors_1m, max_priors_5m are present
            try:
                df = align_with_enhanced_gex(df)
            except Exception:
                pass
        # If index contains timestamp but 'timestamp' column was not present, add it
        if 'timestamp' not in df.columns and df.index.name == 'timestamp':
            df['timestamp'] = df.index

    # Basic feature engineering (same as training)
    print(f"Columns before feature engineering: {list(df.columns)}")
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    # Add enhanced features if available (to match enhanced model training)
    if create_enhanced_features is not None and use_enhanced_features:
        try:
            df = create_enhanced_features(df)
        except Exception as e:
            print(f"Warning: create_enhanced_features failed: {e}")

    # Technical indicators (same as training)
    # RSI
    df['rsi'] = 100 - (100 / (1 + df['close'].diff(1).clip(lower=0).rolling(14).mean() /
                              df['close'].diff(1).clip(upper=0).abs().rolling(14).mean()))

    # ADX and DIs (simplified calculation)
    df['tr'] = np.maximum(df['high'] - df['low'],
                         np.maximum(abs(df['high'] - df['close'].shift(1)),
                                   abs(df['low'] - df['close'].shift(1))))

    df['plus_dm'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                            np.maximum(df['high'] - df['high'].shift(1), 0), 0)
    df['minus_dm'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                             np.maximum(df['low'].shift(1) - df['low'], 0), 0)

    # Smoothed values
    df['atr'] = df['tr'].rolling(14).mean()
    df['plus_di'] = 100 * (df['plus_dm'].rolling(14).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(14).mean() / df['atr'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(14).mean()

    # Use simplified DI values
    df['di_plus'] = df['plus_di']
    df['di_minus'] = df['minus_di']

    # Stochastic Oscillator
    low_min = df['low'].rolling(14).min()
    high_max = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # Create target (next return direction)
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)

    # Select features - match the training feature set
    ohlcv_features = ['open', 'high', 'low', 'close', 'volume', 'BidVolume', 'AskVolume', 'NumTrades']
    
    # Add any additional GEX features that might be present (only numeric ones)
    numeric_dtypes = ['float64', 'float32', 'int64', 'int32', 'uint32', 'uint64', 'int8', 'uint8']
    gex_features = [col for col in df.columns if any(kw in col.lower() for kw in ['gex', 'gamma', 'major', 'delta_risk', 'max_', 'spot', 'zero']) and col not in ohlcv_features and df[col].dtype.name in numeric_dtypes]
    
    feature_cols = ohlcv_features + gex_features
    print(f"Using feature set: {len(feature_cols)} features ({len(ohlcv_features)} OHLCV + {len(gex_features)} GEX)")

    # Ensure all features exist and are numeric
    available_cols = [col for col in feature_cols if col in df.columns and df[col].dtype.name in numeric_dtypes]
    if len(available_cols) != len(feature_cols):
        print(f"Warning: Missing or non-numeric features. Available: {available_cols}")
        print(f"Expected: {feature_cols}")
        feature_cols = available_cols  # Use only available numeric features

    df_features = df[available_cols].copy()

    # Handle NaN values
    df_features = df_features.fillna(method='ffill').fillna(0)

    # Convert to float32 to reduce memory pressure
    df_features = df_features.astype(np.float32)

    # Prepare indices for sequences (apply stride and maximum sample limit)
    start_idx = sequence_length
    end_idx = len(df_features) - 1
    if end_idx <= start_idx:
        return None, None

    seq_indices = list(range(start_idx, end_idx, stride if stride > 0 else 1))
    if max_samples is not None and max_samples > 0 and len(seq_indices) > max_samples:
        seq_indices = seq_indices[:max_samples]

    n_sequences = len(seq_indices)
    if n_sequences == 0:
        return None, None

    # Allocate arrays directly to avoid large python lists
    num_features = df_features.shape[1]
    X = np.empty((n_sequences, sequence_length, num_features), dtype=np.float32)
    y = np.empty((n_sequences,), dtype=np.float32)

    for out_i, i in enumerate(seq_indices):
        seq = df_features.iloc[i-sequence_length:i].values.astype(np.float32)
        target = df['target'].iloc[i]
        X[out_i] = seq
        y[out_i] = target

    # X and y already prepared as float32 arrays

    # Scale features and ensure feature dimensionality matches model expectations
    X_reshaped = X.reshape(-1, X.shape[-1])
    # For LightGBM models, we need exactly 20 features (what the model was trained on)
    # The scaler expects 13 features, but the model expects 20
    if model_type == 'lightgbm':
        # Force to 20 features for LightGBM models
        expected = 20
    else:
        # Use scaler's expected feature count for other models
        expected = None
        if scaler is not None:
            expected = getattr(scaler, 'n_features_in_', None)
        if expected is None:
            expected = required_feature_count

    # debug: shapes for preallocation and scaling
    if expected is not None and X_reshaped.shape[1] != expected:
        if X_reshaped.shape[1] < expected:
            pad = np.zeros((X_reshaped.shape[0], expected - X_reshaped.shape[1]), dtype=X_reshaped.dtype)
            X_reshaped = np.concatenate([X_reshaped, pad], axis=1)
        else:
            # Truncate extra features if expected is fewer features
            X_reshaped = X_reshaped[:, :expected]

    if scaler is not None and model_type != 'lightgbm':
        X_scaled = scaler.transform(X_reshaped)
    else:
        X_scaled = X_reshaped

    # Determine new feature dimension (match model's required feature count if available)
    if model_type == 'lightgbm':
        final_feat_dim = 20  # LightGBM model expects 20 features
    else:
        final_feat_dim = required_feature_count if required_feature_count is not None else X_scaled.shape[1]
    
    if X_scaled.shape[1] != final_feat_dim:
        if X_scaled.shape[1] < final_feat_dim:
            pad = np.zeros((X_scaled.shape[0], final_feat_dim - X_scaled.shape[1]), dtype=X_scaled.dtype)
            X_scaled = np.concatenate([X_scaled, pad], axis=1)
        else:
            X_scaled = X_scaled[:, :final_feat_dim]

    new_feat_dim = X_scaled.shape[1]
    X = X_scaled.reshape(-1, sequence_length, new_feat_dim)

    return X, y

class LSTMModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.0, model_type='lstm'):
        super().__init__()
        if model_type == 'lstm':
            self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif model_type == 'gru':
            self.rnn = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)

def load_models():
    """Load all available models and data from parquet files."""
    # Load data from parquet files instead of .npz
    data_dir = Path(DATASET)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {DATASET} not found")
    
    parquet_files = list(data_dir.glob('*.parquet'))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {DATASET}")
    
    print(f"Loading data from {len(parquet_files)} parquet files...")
    
    # Load and combine all parquet data
    all_X = []
    all_y = []
    
    for parquet_file in parquet_files:
        print(f"Processing {parquet_file.name}...")
        try:
            # Use the same preprocessing as backtest_model.py
            X, y = preprocess_day_data(str(parquet_file), None, sequence_length=60, 
                                     required_feature_count=None, use_enhanced_features=True, 
                                     model_type='lightgbm', max_samples=10000)  # Limit samples per day
            if X is not None and y is not None:
                all_X.append(X)
                all_y.append(y)
        except Exception as e:
            print(f"Warning: Failed to process {parquet_file}: {e}")
            continue
    
    if not all_X:
        raise ValueError("No valid data loaded from parquet files")
    
    # Combine all data
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)
    
    print(f"Combined dataset: {X_combined.shape[0]} samples, {X_combined.shape[2]} features")
    
    # Determine input_dim from the combined data
    input_dim = X_combined.shape[2]

    # For now, only load LightGBM model since LSTM models may have different feature counts
    lstm_model = None
    cv_models = []
    
    # Load LightGBM
    lgb_model = joblib.load(LGB_MODEL) if LGB_MODEL.exists() else None

    return lstm_model, cv_models, lgb_model, input_dim, X_combined, y_combined

def get_predictions(models, X, batch_size=1024):
    """Get predictions from all models."""
    lstm_model, cv_models, lgb_model, input_dim = models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predictions = {}

    # LSTM predictions
    if lstm_model:
        lstm_model.to(device)
        lstm_preds = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            with torch.no_grad():
                logits = lstm_model(batch_tensor)
                pred = torch.sigmoid(logits).cpu().numpy().flatten()
                lstm_preds.extend(pred)
        predictions['lstm'] = np.array(lstm_preds)

    # CV LSTM predictions (averaged)
    if cv_models:
        cv_preds = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).float().to(device)

            fold_preds = []
            for model in cv_models:
                model.to(device)
                with torch.no_grad():
                    logits = model(batch_tensor)
                    pred = torch.sigmoid(logits).cpu().numpy().flatten()
                    fold_preds.append(pred)

            # Average across folds
            avg_pred = np.mean(fold_preds, axis=0)
            cv_preds.extend(avg_pred)
        predictions['cv_lstm'] = np.array(cv_preds)

    # LightGBM predictions
    if lgb_model:
        # For time series data, use the last timestep for prediction
        X_flat = X[:, -1, :]  # Take the last timestep
        predictions['lightgbm'] = lgb_model.predict(X_flat)

    return predictions

def calculate_pnl_at_threshold(preds, labels, threshold, commission_cost=0.42, avg_profit_per_win=10.0, position_sizing='fixed'):
    """Calculate P&L at a specific threshold with position sizing."""
    if position_sizing == 'fixed':
        # Binary decision at threshold
        trades = (preds >= threshold).astype(int)
    elif position_sizing == 'confidence':
        # Position size proportional to confidence
        confidence = np.abs(preds - 0.5) * 2  # 0 to 1 scale
        trades = ((preds >= threshold) & (confidence >= 0.1)).astype(int)  # Minimum confidence
        # Could scale position size here, but for now just binary
    elif position_sizing == 'kelly_fraction':
        # Simplified Kelly criterion approximation
        # Kelly fraction = (win_rate * reward) / risk - loss_rate / risk
        # For simplicity, use confidence as proxy
        confidence = np.abs(preds - 0.5) * 2
        trades = ((preds >= threshold) & (confidence >= 0.2)).astype(int)

    labels_binary = (labels > 0).astype(int)
    trades_taken = np.sum(trades)
    correct_trades = np.sum((trades == 1) & (labels_binary == 1))
    win_rate = correct_trades / trades_taken if trades_taken > 0 else 0

    total_commissions = trades_taken * commission_cost
    gross_pnl = correct_trades * avg_profit_per_win
    net_pnl = gross_pnl - total_commissions

    pnl_per_trade = net_pnl / trades_taken if trades_taken > 0 else 0
    break_even_win_rate = commission_cost / avg_profit_per_win
    edge = win_rate - break_even_win_rate

    return {
        'threshold': threshold,
        'trades_taken': trades_taken,
        'win_rate': win_rate,
        'net_pnl': net_pnl,
        'pnl_per_trade': pnl_per_trade,
        'edge': edge,
        'total_commissions': total_commissions
    }

def optimize_thresholds(predictions, labels, thresholds=[0.30, 0.35, 0.40], position_sizings=['fixed', 'confidence']):
    """Test different thresholds for each model."""
    results = []

    for model_name, preds in predictions.items():
        print(f"\nOptimizing {model_name}...")

        model_results = []
        for threshold in thresholds:
            for position_sizing in ['fixed', 'confidence']:
                metrics = calculate_pnl_at_threshold(
                    preds, labels, threshold,
                    position_sizing=position_sizing
                )
                metrics['model'] = model_name
                metrics['position_sizing'] = position_sizing
                model_results.append(metrics)

        # Sort by net P&L
        model_results.sort(key=lambda x: x['net_pnl'], reverse=True)
        results.extend(model_results)

        # Print top 3 for this model
        print("Top 3 threshold configurations:")
        for i, r in enumerate(model_results[:3]):
            print(f"  {i+1}. Threshold {r['threshold']:.2f} ({r['position_sizing']}): "
                  f"P&L ${r['net_pnl']:.2f}, Win {r['win_rate']:.1%}, "
                  f"{r['trades_taken']} trades, Edge {r['edge']:.1%}")

    return results

def create_ensemble_predictions(predictions, method='weighted_average', weights=None):
    """Create ensemble predictions from individual model predictions."""
    if not predictions:
        return None

    pred_arrays = list(predictions.values())
    model_names = list(predictions.keys())

    if len(pred_arrays) == 0:
        return None

    if method == 'weighted_average':
        if weights is None:
            # Default: favor LightGBM if available
            if 'lightgbm' in predictions:
                weights = [0.3 if name != 'lightgbm' else 0.7 for name in model_names]
                weights = np.array(weights) / sum(weights)
            else:
                weights = np.ones(len(pred_arrays)) / len(pred_arrays)
        else:
            weights = np.array(weights)

        ensemble_pred = np.average(pred_arrays, axis=0, weights=weights)

    elif method == 'median':
        ensemble_pred = np.median(pred_arrays, axis=0)

    elif method == 'max_confidence':
        # Take the prediction with highest confidence from 0.5
        confidences = [np.abs(pred - 0.5) for pred in pred_arrays]
        max_conf_idx = np.argmax(confidences, axis=0)
        ensemble_pred = np.array([pred_arrays[idx][i] for i, idx in enumerate(max_conf_idx)])

    return ensemble_pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlflow_experiment', default='threshold_optimization', help='MLflow experiment name')
    parser.add_argument('--commission_cost', type=float, default=0.42, help='Commission cost per trade')
    args = parser.parse_args()

    try:
        import mlflow_utils
        mlflow_utils.ensure_sqlite_tracking()
    except Exception:
        pass
    mlflow.set_experiment(args.mlflow_experiment)

    # Load data and models
    print("Loading data and models...")
    models = load_models()
    lstm_model, cv_models, lgb_model, input_dim, X, y = models

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[2]} features")

    # Split into test set (same as before)
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Test set: {X_test.shape[0]} samples")

    # Get predictions from all models
    print("Getting predictions...")
    predictions = get_predictions((lstm_model, cv_models, lgb_model, input_dim), X_test)

    if not predictions:
        print("No models could be loaded!")
        return

    print(f"Available models: {list(predictions.keys())}")

    # Optimize thresholds for individual models
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL THRESHOLD OPTIMIZATION")
    print("="*60)

    individual_results = optimize_thresholds(predictions, y_test)

    # Test ensemble methods (only if we have multiple models)
    if len(predictions) > 1:
        print("\n" + "="*60)
        print("ENSEMBLE THRESHOLD OPTIMIZATION")
        print("="*60)

        ensemble_configs = [
            ('weighted_lgb_favor', 'weighted_average', [0.2, 0.2, 0.6]),  # LSTM, CV_LSTM, LightGBM
            ('equal_weight', 'weighted_average', None),
            ('median', 'median', None),
            ('max_confidence', 'max_confidence', None),
        ]

        ensemble_results = []
        for config_name, method, weights in ensemble_configs:
            print(f"\nTesting {config_name} ensemble...")

            ensemble_pred = create_ensemble_predictions(predictions, method=method, weights=weights)
            if ensemble_pred is None:
                continue

            config_results = []
            thresholds = [0.30, 0.35, 0.40]

            for threshold in thresholds:
                for position_sizing in ['fixed', 'confidence']:
                    metrics = calculate_pnl_at_threshold(
                        ensemble_pred, y_test, threshold,
                        position_sizing=position_sizing
                    )
                    metrics['model'] = f'ensemble_{config_name}'
                    metrics['position_sizing'] = position_sizing
                    config_results.append(metrics)

            # Sort by net P&L
            config_results.sort(key=lambda x: x['net_pnl'], reverse=True)
            ensemble_results.extend(config_results)

            # Print top result for this ensemble
            best = config_results[0]
            print(f"  Best: Threshold {best['threshold']:.2f} ({best['position_sizing']}): "
                  f"P&L ${best['net_pnl']:.2f}, Win {best['win_rate']:.1%}, "
                  f"{best['trades_taken']} trades, Edge {best['edge']:.1%}")
    else:
        print("\nSkipping ensemble optimization (only one model available)")
        ensemble_results = []

    # Combine all results
    all_results = individual_results + ensemble_results

    # Find overall best
    all_results.sort(key=lambda x: x['net_pnl'], reverse=True)

    print("\n" + "="*60)
    print("OVERALL BEST CONFIGURATIONS")
    print("="*60)

    for i, result in enumerate(all_results[:10]):
        print(f"{i+1:2d}. {result['model'][:20]:20} | Threshold: {result['threshold']:.2f} | "
              f"Sizing: {result['position_sizing']} | "
              f"P&L: ${result['net_pnl']:8.2f} | Win: {result['win_rate']:5.1%} | "
              f"Trades: {result['trades_taken']:4d} | Edge: {result['edge']:5.1%}")

    # Log best results to MLflow
    print("\nLogging top 5 results to MLflow...")
    for i, result in enumerate(all_results[:5]):
        with mlflow.start_run(run_name=f"top_{i+1}_{result['model']}_thresh_{result['threshold']:.2f}"):
            mlflow.log_param("model", result['model'])
            mlflow.log_param("threshold", result['threshold'])
            mlflow.log_param("position_sizing", result['position_sizing'])
            mlflow.log_param("commission_cost", args.commission_cost)

            mlflow.log_metric("net_pnl", result['net_pnl'])
            mlflow.log_metric("win_rate", result['win_rate'])
            mlflow.log_metric("trades_taken", result['trades_taken'])
            mlflow.log_metric("edge", result['edge'])
            mlflow.log_metric("pnl_per_trade", result['pnl_per_trade'])

    # Save detailed results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv('ml/threshold_optimization_results.csv', index=False)
    print("Detailed results saved to ml/threshold_optimization_results.csv")

    print("\n✅ Threshold optimization complete!")
    print(f"Best configuration: {all_results[0]['model']} at threshold {all_results[0]['threshold']:.2f}")
    print(f"Best P&L: ${all_results[0]['net_pnl']:.2f}")
    print(f"Best win rate: {all_results[0]['win_rate']:.1%}")
    print(f"Improvement potential: ${all_results[0]['net_pnl'] - all_results[-1]['net_pnl']:.2f} over baseline")

if __name__ == '__main__':
    main()