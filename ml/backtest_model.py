#!/usr/bin/env python3
"""Backtest the threshold-optimized model on 10 additional days of data."""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import mlflow
import mlflow.pytorch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import optional dependencies
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

create_enhanced_features = None
try:
    from ml.enhanced_features import create_enhanced_features
except Exception:
    try:
        from enhanced_features import create_enhanced_features
    except Exception:
        pass

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)

def load_model_and_scaler(model_path):
    """Load the trained model and scaler."""
    # Prefer joblib for .pkl / .joblib extensions to avoid torch attempts
    ext = Path(model_path).suffix.lower()
    if ext in ('.pkl', '.joblib'):
        try:
            import joblib
            sk_model = joblib.load(model_path)
            scaler = None
            # try to find the scaler in standard locations
            # Prefer the 13-feature scaler for LightGBM / sklearn baseline models
            possible = []
            if 'lightgbm' in model_path.lower() or 'mnq_lightgbm' in model_path.lower() or 'lgb' in model_path.lower():
                possible = [Path('ml/models/scaler_13_features.pkl'), Path('models/scaler_13_features.pkl')]
            elif 'enhanced' in model_path.lower():
                possible = [Path('ml/models/enhanced_gex_scaler.pkl'), Path('models/enhanced_gex_scaler.pkl')]
            else:
                possible = [Path('models/scaler_13_features.pkl'), Path('ml/models/scaler_13_features.pkl'), Path('ml/models/enhanced_gex_scaler.pkl'), Path('models/enhanced_gex_scaler.pkl')]
            for p in possible:
                try:
                    if p.exists():
                        scaler = joblib.load(p)
                        break
                except Exception:
                    pass
            return sk_model, scaler, {'model_type': 'sklearn'}
        except Exception as e:
            # If we failed to load a .pkl/.joblib with joblib, try native formats
            try:
                # Try LightGBM native format
                import lightgbm as lgb
                lgb_model = lgb.Booster(model_file=model_path)
                scaler = None
                # Try to find scaler
                possible = [Path('ml/models/scaler_13_features.pkl'), Path('models/scaler_13_features.pkl')]
                for p in possible:
                    try:
                        if p.exists():
                            import joblib
                            scaler = joblib.load(p)
                            break
                    except Exception:
                        pass
                return lgb_model, scaler, {'model_type': 'lightgbm'}
            except Exception:
                try:
                    # Try XGBoost native format
                    import xgboost as xgb
                    xgb_model = xgb.Booster(model_file=model_path)
                    scaler = None
                    # Try to find scaler
                    possible = [Path('ml/models/scaler_13_features.pkl'), Path('models/scaler_13_features.pkl')]
                    for p in possible:
                        try:
                            if p.exists():
                                import joblib
                                scaler = joblib.load(p)
                                break
                        except Exception:
                            pass
                    return xgb_model, scaler, {'model_type': 'xgboost'}
                except Exception as e2:
                    # If all loading attempts fail, raise the original exception
                    raise RuntimeError(f"Failed to load model at {model_path}: {e}") from e
    elif ext in ('.model', '.txt'):
        # Check if this is likely a LightGBM model based on filename
        if 'lightgbm' in model_path.lower() or 'lgb' in model_path.lower():
            # Try LightGBM first for models with lightgbm in the name
            try:
                import lightgbm as lgb
                lgb_model = lgb.Booster(model_file=model_path)
                scaler = None
                # Try to find scaler
                possible = [Path('ml/models/scaler_13_features.pkl'), Path('models/scaler_13_features.pkl')]
                for p in possible:
                    try:
                        if p.exists():
                            import joblib
                            scaler = joblib.load(p)
                            break
                    except Exception:
                        pass
                return lgb_model, scaler, {'model_type': 'lightgbm'}
            except Exception:
                pass  # Fall through to XGBoost
        
        # Try XGBoost native format
        try:
            import xgboost as xgb
            xgb_model = xgb.Booster(model_file=model_path)
            scaler = None
            # Try to find scaler
            possible = [Path('ml/models/scaler_13_features.pkl'), Path('models/scaler_13_features.pkl')]
            for p in possible:
                try:
                    if p.exists():
                        import joblib
                        scaler = joblib.load(p)
                        break
                except Exception:
                    pass
            return xgb_model, scaler, {'model_type': 'xgboost'}
        except Exception:
            # If XGBoost fails and we haven't tried LightGBM yet, try LightGBM
            if 'lightgbm' not in model_path.lower() and 'lgb' not in model_path.lower():
                try:
                    import lightgbm as lgb
                    lgb_model = lgb.Booster(model_file=model_path)
                    scaler = None
                    # Try to find scaler
                    possible = [Path('ml/models/scaler_13_features.pkl'), Path('models/scaler_13_features.pkl')]
                    for p in possible:
                        try:
                            if p.exists():
                                import joblib
                                scaler = joblib.load(p)
                                break
                        except Exception:
                            pass
                    return lgb_model, scaler, {'model_type': 'lightgbm'}
                except Exception as e:
                    raise RuntimeError(f"Failed to load native model at {model_path}: {e}")
            else:
                raise RuntimeError(f"Failed to load native model at {model_path}: XGBoost failed")
    try:
        # Try loading as full checkpoint first
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Full checkpoint format
            input_dim = checkpoint.get('input_dim', 13)
            hidden_dim = checkpoint.get('hidden_dim', 256)  # Default for production model
            num_layers = checkpoint.get('num_layers', 2)
            model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
            # Annotate model with input dimension for downstream preprocessing
            try:
                model.input_dim = input_dim
            except Exception:
                pass
            model.load_state_dict(checkpoint['model_state_dict'])
            scaler = checkpoint.get('scaler')
            return model, scaler, checkpoint
        else:
            # Just state dict - infer architecture from keys
            state_dict = checkpoint

            # Check for rnn layers (2-layer model)
            has_rnn_l1 = any('rnn.weight_ih_l1' in k for k in state_dict.keys())
            num_layers = 2 if has_rnn_l1 else 1

            # Infer input and hidden dimensions
            if 'rnn.weight_ih_l0' in state_dict:
                hidden_dim = state_dict['rnn.weight_ih_l0'].shape[0] // 4  # LSTM has 4 gates
                input_dim = state_dict['rnn.weight_ih_l0'].shape[1]
            elif 'lstm.weight_ih_l0' in state_dict:
                hidden_dim = state_dict['lstm.weight_ih_l0'].shape[0] // 4
                input_dim = state_dict['lstm.weight_ih_l0'].shape[1]
            else:
                input_dim = 13  # Default
                hidden_dim = 64

            print(f"Inferred architecture: input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}")

            model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
            # Annotate model with input dimension for downstream preprocessing
            try:
                model.input_dim = input_dim
            except Exception:
                pass
            model.load_state_dict(state_dict)
            
            # Try to load scaler separately
            try:
                import joblib
                scaler = joblib.load('models/scaler_13_features.pkl')
                print("Loaded scaler from models/scaler_13_features.pkl")
            except Exception as e:
                print(f"Could not load scaler: {e}")
                scaler = None
            
            return model, scaler, {'input_dim': input_dim, 'hidden_dim': hidden_dim, 'num_layers': num_layers}
    except Exception as e:
        print(f"Error loading with torch: {e}")
        # If file extension is a pickle or joblib try loading with joblib
        try:
            import joblib
            sk_model = joblib.load(model_path)
            scaler = None
            # Prefer the 13-feature scaler for LightGBM / sklearn baseline models
            possible = []
            if 'lightgbm' in model_path.lower() or 'mnq_lightgbm' in model_path.lower() or 'lgb' in model_path.lower():
                possible = [Path('ml/models/scaler_13_features.pkl'), Path('models/scaler_13_features.pkl')]
            elif 'enhanced' in model_path.lower():
                possible = [Path('ml/models/enhanced_gex_scaler.pkl'), Path('models/enhanced_gex_scaler.pkl')]
            else:
                possible = [Path('models/scaler_13_features.pkl'), Path('ml/models/scaler_13_features.pkl'), Path('ml/models/enhanced_gex_scaler.pkl'), Path('models/enhanced_gex_scaler.pkl')]
            for p in possible:
                try:
                    if p.exists():
                        scaler = joblib.load(p)
                        break
                except Exception:
                    pass
            return sk_model, scaler, {'model_type': 'sklearn'}
        except Exception as e2:
            print(f"Joblib load failed: {e2}")
            raise

    # No further generic pkl loading fallbacks. We intentionally avoid attempting torch.load
    # when encountering a .pkl/.joblib file. If control reaches here and model is not loaded,
    # propagate the earlier exception to the caller.

def load_enhanced_gex_data():
    """Load NQ GEX data filtered to market hours for MNQ usage."""
    # Load GEX snapshots
    df_gex = pd.read_parquet('./data/exports/gex_snapshots_epoch.parquet')
    df_gex['timestamp'] = pd.to_datetime(df_gex['epoch_ms'], unit='ms')
    
    # Filter to NQ_NDX and market hours
    df_nq = df_gex[df_gex['ticker'] == 'NQ_NDX'].copy()
    df_nq['timestamp_et'] = df_nq['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    
    from datetime import time
    market_open = time(9, 32)
    market_close = time(16, 0)
    
    df_market = df_nq[
        (df_nq['timestamp_et'].dt.time >= market_open) & 
        (df_nq['timestamp_et'].dt.time <= market_close)
    ].copy()
    
    # Extract max_priors components
    def extract_priors(prior_str):
        if pd.isna(prior_str):
            return pd.Series([0, 0, 0], index=['max_priors_current', 'max_priors_1m', 'max_priors_5m'])
        try:
            import json
            priors = json.loads(prior_str)
            current = priors[0][1] if len(priors) > 0 else 0
            pri_1m = priors[1][1] if len(priors) > 1 else 0  
            pri_5m = priors[4][1] if len(priors) > 4 else 0
            return pd.Series([current, pri_1m, pri_5m], index=['max_priors_current', 'max_priors_1m', 'max_priors_5m'])
        except:
            return pd.Series([0, 0, 0], index=['max_priors_current', 'max_priors_1m', 'max_priors_5m'])
    
    # Apply extraction
    priors_df = df_market['max_priors'].apply(extract_priors)
    df_market = pd.concat([df_market, priors_df], axis=1)
    
    # Select enhanced GEX features
    enhanced_cols = [
        'timestamp', 'spot_price',
        'zero_gamma', 'net_gex', 'major_pos_vol', 'major_neg_vol', 
        'sum_gex_vol', 'delta_risk_reversal',
        'max_priors_current', 'max_priors_1m', 'max_priors_5m'
    ]
    
    df_enhanced = df_market[enhanced_cols].copy()
    df_enhanced = df_enhanced.sort_values('timestamp').drop_duplicates('timestamp')
    df_enhanced = df_enhanced.set_index('timestamp', drop=False)
    
    return df_enhanced

def align_with_enhanced_gex(df_tick):
    """Align tick data with enhanced GEX data."""
    try:
        # Load enhanced GEX data (cached to avoid reloading)
        if not hasattr(align_with_enhanced_gex, '_gex_cache'):
            align_with_enhanced_gex._gex_cache = load_enhanced_gex_data()
        
        df_gex = align_with_enhanced_gex._gex_cache
        
        # Ensure tick data timestamps are tz-naive UTC for alignment
        # Handle RangeIndex or tz-naive indices safely
        if hasattr(df_tick.index, 'tz') and df_tick.index.tz is not None:
            tick_timestamps_utc = df_tick.index.tz_convert('UTC').tz_localize(None)
        else:
            # Assume tick data is already in UTC if tz-naive or RangeIndex
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

def calculate_daily_gex_regime(df, gex_threshold=0.65):
    """Calculate daily GEX regime for market filtering.
    
    Args:
        df: DataFrame with 'net_gex' column
        gex_threshold: Threshold as fraction of standard deviation (0.65 = 65%)
    
    Returns:
        dict with regime information
    """
    if 'net_gex' not in df.columns:
        return {'regime': 'neutral', 'avg_gex': 0.0, 'std_gex': 0.0}
    
    # Calculate daily GEX statistics
    avg_gex = df['net_gex'].mean()
    std_gex = df['net_gex'].std()
    
    # Determine regime based on average GEX vs standard deviation threshold
    if avg_gex > gex_threshold * std_gex:
        regime = 'positive'
    elif avg_gex < -gex_threshold * std_gex:
        regime = 'negative'
    else:
        regime = 'neutral'
    
    return {
        'regime': regime,
        'avg_gex': avg_gex,
        'std_gex': std_gex,
        'threshold_value': gex_threshold * std_gex
    }

def evaluate_model_on_day(model, X, y, threshold=0.3, commission_cost=0.42, slippage_cost=0.50, 
                         model_type='sklearn', tp_sl_ratio=1.0, stop_loss_amount=1.0, 
                         gex_regime=None, use_gex_filter=False):
    """Evaluate model performance on a single day's data.
    
    Args:
        gex_regime: Dict with daily GEX regime info
        use_gex_filter: Whether to filter trades based on GEX regime
        tp_sl_ratio: Take Profit to Stop Loss ratio (e.g., 2.0 = 2:1, 0.5 = 1:2)
        stop_loss_amount: Stop loss amount in dollars (e.g., 1.0 = $1 SL)
    """
    if X is None or len(X) == 0:
        take_profit_amount = stop_loss_amount * tp_sl_ratio
        expected_win = take_profit_amount - (commission_cost + slippage_cost)
        expected_loss = stop_loss_amount + (commission_cost + slippage_cost)
        return {
            'trades_taken': 0,
            'win_rate': 0.0,
            'net_pnl': 0.0,
            'total_commissions': 0.0,
            'edge': -(commission_cost + slippage_cost) / expected_win if expected_win > 0 else -1.0,
            'gex_regime': gex_regime['regime'] if gex_regime else 'unknown'
        }

    if model_type == 'sklearn':
        # For sklearn models (including LightGBM, XGBoost loaded via joblib)
        predictions = model.predict_proba(X.reshape(X.shape[0], -1))[:, 1]  # Get probability of positive class
    elif model_type == 'lightgbm':
        # LightGBM native model - use last timestep features
        predictions = model.predict(X[:, -1, :])
    elif model_type == 'xgboost':
        # XGBoost native model - use last timestep features
        dmatrix = xgb.DMatrix(X[:, -1, :])
        predictions = model.predict(dmatrix)
    else:
        # PyTorch model
        model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X), 64):  # Batch processing
                batch_X = torch.from_numpy(X[i:i+64]).float()
                batch_preds = model(batch_X)
                predictions.extend(torch.sigmoid(batch_preds).cpu().numpy().flatten())
        predictions = np.array(predictions)

    # Apply threshold
    trades = (predictions >= threshold).astype(int)

    # Apply GEX regime filter if enabled
    if use_gex_filter and gex_regime:
        regime = gex_regime['regime']
        if regime == 'positive':
            # Only allow long trades on positive GEX days
            pass  # Keep all trades as-is
        elif regime == 'negative':
            # Allow short trades on negative GEX days - invert the model's predictions
            trades = 1 - trades  # Flip 0->1 and 1->0 (short when model says long)
        else:
            # Neutral regime - allow all trades
            pass

    # Calculate P&L with TP/SL ratios
    trades_taken = np.sum(trades)
    if trades_taken == 0:
        take_profit_amount = stop_loss_amount * tp_sl_ratio
        expected_win = take_profit_amount - (commission_cost + slippage_cost)
        expected_loss = stop_loss_amount + (commission_cost + slippage_cost)
        return {
            'trades_taken': 0,
            'win_rate': 0.0,
            'net_pnl': 0.0,
            'total_commissions': 0.0,
            'edge': -(commission_cost + slippage_cost) / expected_win if expected_win > 0 else -1.0,
            'gex_regime': gex_regime['regime'] if gex_regime else 'unknown'
        }

    correct_trades = np.sum((trades == 1) & (y[:len(trades)] == 1))
    win_rate = correct_trades / trades_taken

    # Calculate TP/SL amounts
    take_profit_amount = stop_loss_amount * tp_sl_ratio
    
    # Per trade costs
    cost_per_trade = commission_cost + slippage_cost
    
    # P&L calculation
    winning_trades_pnl = correct_trades * (take_profit_amount - cost_per_trade)
    losing_trades_pnl = (trades_taken - correct_trades) * (-stop_loss_amount - cost_per_trade)
    net_pnl = winning_trades_pnl + losing_trades_pnl
    
    total_commissions = trades_taken * commission_cost
    total_slippage = trades_taken * slippage_cost

    # Edge calculation: expected value per trade
    expected_win = take_profit_amount - cost_per_trade
    expected_loss = stop_loss_amount + cost_per_trade
    edge = win_rate * expected_win + (1 - win_rate) * expected_loss

    return {
        'trades_taken': int(trades_taken),
        'win_rate': float(win_rate),
        'net_pnl': float(net_pnl),
        'total_commissions': float(total_commissions),
        'total_slippage': float(total_slippage),
        'edge': float(edge),
        'tp_sl_ratio': tp_sl_ratio,
        'stop_loss_amount': stop_loss_amount,
        'take_profit_amount': take_profit_amount,
        'gex_regime': gex_regime['regime'] if gex_regime else 'unknown'
    }

def find_available_days(data_dirs=['ml/output', 'output', 'data', 'data/tick/MNQ']):
    """Find available trading days for backtesting."""
    all_files = []

    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if data_path.exists():
            parquet_files = list(data_path.glob('**/*.parquet'))
            all_files.extend(parquet_files)

    print(f"Found {len(all_files)} parquet files total")

    # Extract dates from filenames
    days = []
    for file in all_files:
        filename = file.name
        # Try different date formats
        date_str = None

        # Format: MNQ_2025-11-10_1s.parquet or MNQ_20251110_1s.parquet
        if filename.startswith('MNQ_') and '_1s.parquet' in filename:
            date_str = filename.replace('MNQ_', '').replace('_1s.parquet', '').replace('-', '')

        # Format: 20251110.parquet (tick data)
        elif filename.endswith('.parquet') and len(filename) == 16:
            date_candidate = filename[:-8]  # Remove .parquet
            if len(date_candidate) == 8 and date_candidate.isdigit():
                date_str = date_candidate

        # Format: mnq_2025-09-02.parquet (processed data)
        elif filename.startswith('mnq_') and filename.endswith('.parquet'):
            date_str = filename.replace('mnq_', '').replace('.parquet', '').replace('-', '')

        if date_str and len(date_str) == 8 and date_str.isdigit():
            try:
                date = datetime.strptime(date_str, '%Y%m%d').date()
                days.append((date, file))
            except ValueError:
                continue

    # Sort by date and remove duplicates
    days.sort(key=lambda x: x[0])
    unique_days = []
    seen_dates = set()
    for date, file_path in days:
        if date not in seen_dates:
            unique_days.append((date, file_path))
            seen_dates.add(date)

    return unique_days

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/lstm_production.pt', help='Path to trained model')
    parser.add_argument('--threshold', type=float, default=0.3, help='Prediction threshold')
    parser.add_argument('--commission_cost', type=float, default=None, help='Commission cost per trade (roundtrip). Overrides --instrument')
    parser.add_argument('--commission_per_side', type=float, default=None, help='Commission per side (single contract). Roundtrip = per_side*2')
    parser.add_argument('--slippage_cost', type=float, default=0.50, help='Slippage cost per trade (in dollars). Default 0.50 for 1 tick on MNQ')
    parser.add_argument('--instrument', default='MNQ', help='Instrument being backtested (MNQ or NQ)')
    parser.add_argument('--max_days', type=int, default=10, help='Maximum number of days to backtest')
    parser.add_argument('--mlflow_experiment', default='backtest_threshold_optimized', help='MLflow experiment name')
    parser.add_argument('--tp_sl_ratios', type=str, default='4.0,2.0,1.0,0.5,0.25', help='Comma-separated list of TP:SL ratios to test (e.g., "4.0,2.0,1.0,0.5,0.25")')
    parser.add_argument('--stop_loss_amount', type=float, default=1.0, help='Stop loss amount in dollars (default: $1.00)')
    parser.add_argument('--use_gex_filter', action='store_true', help='Enable GEX regime filtering (only trade with prevailing gamma exposure)')
    parser.add_argument('--gex_threshold', type=float, default=0.65, help='GEX threshold as fraction for regime classification (default: 0.65 = 65%)')
    args = parser.parse_args()

    try:
        import mlflow_utils
        mlflow_utils.ensure_sqlite_tracking()
    except Exception:
        pass
    mlflow.set_experiment(args.mlflow_experiment)

    # Commission map (roundtrip)
    # Commission map per side (per single contract)
    COMM_PER_SIDE = {
        'MNQ': 0.21,  # per side
        'NQ': 0.84,   # per side
        'NQ_NDX': 0.84,
    }

    # Parse TP/SL ratios
    tp_sl_ratios = [float(r.strip()) for r in args.tp_sl_ratios.split(',')]
    print(f"Testing TP:SL ratios: {tp_sl_ratios}")
    print(f"Stop Loss Amount: ${args.stop_loss_amount}")
    if args.use_gex_filter:
        print(f"GEX Regime Filter: ENABLED (threshold: {args.gex_threshold*100:.0f}%)")
        print("  - Positive GEX days (>65%): Only LONG trades")
        print("  - Negative GEX days (<-65%): SHORT trades (inverted signals)")
        print("  - Neutral GEX days: All trades allowed")
    else:
        print("GEX Regime Filter: DISABLED")

    # Load model
    print(f"Loading model from {args.model}")
    try:
        model, scaler, checkpoint = load_model_and_scaler(args.model)
        model_type = checkpoint.get('model_type', 'sklearn')
        print(f"Model loaded successfully (type: {model_type})")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Determine commission (per roundtrip) if not explicitly provided
    if args.commission_per_side is not None:
        commission = args.commission_per_side * 2.0
    elif args.commission_cost is not None:
        commission = args.commission_cost
    else:
        commission = COMM_PER_SIDE.get(args.instrument.upper(), 0.21) * 2.0

    # Find available days
    print("Finding available trading days...")
    available_days = find_available_days()

    if not available_days:
        print("No trading days found!")
        return

    print(f"Found {len(available_days)} trading days")

    # Select days not used in training (avoid the days we trained on)
    training_dates = {'2025-11-10', '2025-11-11'}  # Dates we know were used for enhanced training

    # Filter out training days
    non_training_days = []
    for date, file_path in available_days:
        date_str = date.strftime('%Y-%m-%d')
        if date_str not in training_dates:
            non_training_days.append((date, file_path))

    # Select all November 2025 days
    start_idx = 0  # Include all available November days
    november_days = [day for day in non_training_days[start_idx:] if day[0].month == 11 and day[0].year == 2025]
    backtest_days = november_days

    # If we don't have enough days, take what's available
    if len(backtest_days) < args.max_days:
        print(f"Warning: Only found {len(backtest_days)} additional non-training days")
        # If still not enough, include some from the beginning or training days
        if len(backtest_days) == 0:
            backtest_days = non_training_days[:args.max_days]
            print("Using first available days instead")

    print(f"Selected {len(backtest_days)} days for backtesting:")
    for date, file_path in backtest_days:
        print(f"  {date}: {file_path}")

    # Test each TP/SL ratio
    all_results = []
    
    for ratio in tp_sl_ratios:
        print(f"\n{'='*60}")
        print(f"TESTING TP:SL RATIO {ratio}:1 (TP=${args.stop_loss_amount * ratio:.2f}, SL=${args.stop_loss_amount:.2f})")
        if args.use_gex_filter:
            print(f"GEX FILTER ENABLED (threshold: {args.gex_threshold*100:.0f}%)")
        print(f"{'='*60}")
        
        # Backtest on each day for this ratio
        results = []

        for date, file_path in backtest_days:
            print(f"\nBacktesting {date}...")

            try:
                # Load raw data to calculate GEX regime
                df_raw = pd.read_parquet(file_path)
                gex_regime = None
                if args.use_gex_filter and 'net_gex' in df_raw.columns:
                    gex_regime = calculate_daily_gex_regime(df_raw, args.gex_threshold)
                    print(f"  GEX Regime: {gex_regime['regime']} (avg: {gex_regime['avg_gex']:,.0f}, threshold: {gex_regime['threshold_value']:,.0f})")

                # Preprocess data
                # Try to infer required feature count from model if possible
                required_feature_count = None
                if '38features' in args.model:
                    required_feature_count = 38
                try:
                    if hasattr(model, 'rnn') and hasattr(model.rnn, 'input_size'):
                        required_feature_count = model.rnn.input_size
                    elif hasattr(model, 'input_dim'):
                        required_feature_count = getattr(model, 'input_dim')
                except Exception:
                    required_feature_count = None

                X, y = preprocess_day_data(file_path, scaler, sequence_length=60, required_feature_count=required_feature_count, use_enhanced_features='38features' not in args.model, model_type=model_type)

                # Evaluate model with this TP/SL ratio and GEX regime
                day_result = evaluate_model_on_day(model, X, y, args.threshold, commission, args.slippage_cost, model_type, ratio, args.stop_loss_amount, gex_regime, args.use_gex_filter)
                day_result['date'] = date.strftime('%Y-%m-%d')
                day_result['file'] = str(file_path)
                day_result['tp_sl_ratio'] = ratio

                results.append(day_result)

                regime_str = f" [{day_result['gex_regime']}]" if 'gex_regime' in day_result else ""
                print(".4f")
                print(".1%")
                print(".4f")
                print(f"Trades: {day_result['trades_taken']}, Edge: {day_result['edge']:.4f}{regime_str}")

            except Exception as e:
                print(f"Error backtesting {date}: {e}")
                continue

        # Aggregate results for this ratio
        if not results:
            print(f"No successful backtests for ratio {ratio}!")
            continue

        df_results = pd.DataFrame(results)

        total_pnl = df_results['net_pnl'].sum()
        avg_win_rate = df_results['win_rate'].mean()
        total_trades = df_results['trades_taken'].sum()
        total_commissions = df_results['total_commissions'].sum()
        total_slippage = df_results['total_slippage'].sum()
        avg_edge = df_results['edge'].mean()

        # GEX regime breakdown
        if args.use_gex_filter and 'gex_regime' in df_results.columns:
            regime_counts = df_results['gex_regime'].value_counts()
            print(f"\nGEX Regime Distribution:")
            for regime, count in regime_counts.items():
                regime_pnl = df_results[df_results['gex_regime'] == regime]['net_pnl'].sum()
                print(f"  {regime}: {count} days, P&L: ${regime_pnl:.2f}")

        print(f"\n{'='*60}")
        print(f"RESULTS FOR TP:SL RATIO {ratio}:1")
        print(f"{'='*60}")

        print(f"\nBacktested {len(results)} days")
        print(".2f")
        print(".1%")
        print(f"Total Trades: {total_trades}")
        print(".2f")
        print(".2f")
        print(".4f")

        print("\nDaily Performance:")
        for _, row in df_results.iterrows():
            regime_str = f" [{row['gex_regime']}]" if 'gex_regime' in row else ""
            print(f"{row['date']}: P&L ${row['net_pnl']:8.2f} | Win: {row['win_rate']:5.1%} | "
                  f"Trades: {row['trades_taken']:4d} | Edge: {row['edge']:6.4f}{regime_str}")

        # Store results for comparison
        ratio_summary = {
            'tp_sl_ratio': ratio,
            'total_pnl': total_pnl,
            'avg_win_rate': avg_win_rate,
            'total_trades': total_trades,
            'avg_edge': avg_edge,
            'pnl_per_day': total_pnl / len(results),
            'days_tested': len(results),
            'gex_filter_enabled': args.use_gex_filter
        }
        all_results.append(ratio_summary)

        # Log to MLflow for this ratio
        with mlflow.start_run(run_name=f"backtest_ratio_{ratio}_{len(results)}_days{'_gex_filter' if args.use_gex_filter else ''}"):
            mlflow.log_param("model_path", args.model)
            mlflow.log_param("threshold", args.threshold)
            mlflow.log_param("commission_cost", commission)
            mlflow.log_param("slippage_cost", args.slippage_cost)
            mlflow.log_param("tp_sl_ratio", ratio)
            mlflow.log_param("stop_loss_amount", args.stop_loss_amount)
            mlflow.log_param("days_backtested", len(results))
            mlflow.log_param("total_trades", total_trades)
            mlflow.log_param("gex_filter_enabled", args.use_gex_filter)
            if args.use_gex_filter:
                mlflow.log_param("gex_threshold", args.gex_threshold)

            mlflow.log_metric("total_pnl", total_pnl)
            mlflow.log_metric("avg_win_rate", avg_win_rate)
            mlflow.log_metric("avg_edge", avg_edge)
            mlflow.log_metric("total_commissions", total_commissions)
            mlflow.log_metric("total_slippage", total_slippage)
            mlflow.log_metric("pnl_per_day", total_pnl / len(results))

            # Log daily results
            for i, result in enumerate(results):
                mlflow.log_metric(f"day_{i+1}_pnl", result['net_pnl'])
                mlflow.log_metric(f"day_{i+1}_win_rate", result['win_rate'])

    # Compare all ratios
    if all_results:
        print(f"\n{'='*80}")
        print("TP:SL RATIO COMPARISON SUMMARY")
        print(f"{'='*80}")
        filter_status = "WITH GEX FILTER" if args.use_gex_filter else "WITHOUT GEX FILTER"
        print(f"{filter_status}")
        print(f"{'Ratio':<8} {'Total P&L':<12} {'Avg Win%':<10} {'Avg Edge':<12} {'P&L/Day':<12} {'Trades':<8}")
        print("-" * 80)
        
        for result in sorted(all_results, key=lambda x: x['avg_edge'], reverse=True):
            print(f"{result['tp_sl_ratio']:<8.1f} ${result['total_pnl']:<11.2f} {result['avg_win_rate']:<9.1%} {result['avg_edge']:<11.4f} ${result['pnl_per_day']:<11.2f} {result['total_trades']:<8}")

        # Save detailed results
        df_all_results = pd.DataFrame(all_results)
        filename = 'gex_filter_comparison.csv' if args.use_gex_filter else 'tp_sl_ratio_comparison.csv'
        df_all_results.to_csv(filename, index=False)
        print(f"\nDetailed results saved to {filename}")

    print("\n✅ Backtesting Complete!")

if __name__ == '__main__':
    main()