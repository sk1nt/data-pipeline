#!/usr/bin/env python3
"""Simulate P&L using model predictions and real price movement on enriched data.

Usage:
  python ml/pnl_backtest.py --model models/lstm_production.pt --days 20251020,20251021 --window 60
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import math

from ml.backtest_model import load_model_and_scaler, preprocess_day_data
try:
    import mlflow
except Exception:
    mlflow = None


def simulate_price_pnl(closes, preds, sequence_length, horizon=1, contract_value=5.0, commission=0.42, slippage=0.0, position_size=1, stop_loss=None, take_profit=None):
    """Simulate P&L for predictions.

    closes: pandas Series of close prices aligned with original df.
    preds: numpy array of predictions aligned with X windows (length N)
    sequence_length: window length in preprocess function; the prediction index maps to closes index sequence_length + k
    """
    trades = []
    cumulative = 0.0
    daily_pnls = []
    for k, prob in enumerate(preds):
        if prob <= 0.5:
            continue
        idx = sequence_length + k
        exit_idx = idx + horizon
        if exit_idx >= len(closes):
            break
        entry_price = float(closes.iloc[idx])
        # Default exit price (at horizon close); may be overridden by stop or tp triggered earlier
        exit_price = float(closes.iloc[exit_idx])
        # Evaluate in-between prices for early stop / take profit triggers
        if stop_loss is not None or take_profit is not None:
            path_prices = closes.iloc[idx:exit_idx+1].values
            triggered = False
            for j in range(1, len(path_prices)):
                price_at_j = float(path_prices[j])
                # For long positions check stop_loss first
                if stop_loss is not None:
                    stop_price = entry_price - (stop_loss / contract_value)
                    if price_at_j <= stop_price:
                        exit_price = price_at_j
                        triggered = True
                        break
                if take_profit is not None:
                    tp_price = entry_price + (take_profit / contract_value)
                    if price_at_j >= tp_price:
                        exit_price = price_at_j
                        triggered = True
                        break
        # long trade
        gross = (exit_price - entry_price) * contract_value * position_size
        # apply slippage as fraction of price move
        gross -= slippage
        # commissions
        net = gross - commission * position_size
        trades.append({'entry_idx': idx, 'exit_idx': exit_idx, 'entry_price': entry_price, 'exit_price': exit_price, 'pnl': net})
        cumulative += net
        daily_pnls.append(net)

    trades_taken = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    win_rate = len(wins) / trades_taken if trades_taken > 0 else 0.0
    total_pnl = sum(t['pnl'] for t in trades)
    avg_pnl = total_pnl / trades_taken if trades_taken > 0 else 0.0
    return {
        'trades': trades_taken,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'cumulative': cumulative,
        'trades_list': trades,
    }


def predict_from_model(model, X, device='cpu', batch_size=256):
    # If it's a scikit-learn or LightGBM estimator, flatten the windows
    # Robust detection: PyTorch nn.Module has `state_dict` or is instance
    is_torch_model = hasattr(model, 'state_dict') or hasattr(model, 'forward')
    is_sklearn = hasattr(model, 'predict') and not is_torch_model

    preds = []
    if is_sklearn:
        # flatten to 2D
        flat = X.reshape(X.shape[0], -1)
        # Ensure number of features matches model expectations; pad/truncate as needed
        expected = None
        try:
            if hasattr(model, 'n_features_in_'):
                expected = int(model.n_features_in_)
        except Exception:
            expected = None

        # LightGBM estimator wrapper exposes booster_ with num_feature()
        try:
            if expected is None and hasattr(model, 'booster_'):
                expected = int(model.booster_.num_feature())
        except Exception:
            pass

        try:
            if expected is not None and flat.shape[1] != expected:
                if flat.shape[1] < expected:
                    pad = np.zeros((flat.shape[0], expected - flat.shape[1]), dtype=flat.dtype)
                    flat = np.concatenate([flat, pad], axis=1)
                else:
                    flat = flat[:, :expected]
        except Exception:
            pass
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(flat)[:, 1]
        else:
            probs = model.predict(flat)
        return np.array(probs)

    model.to(device)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            tensor = torch.from_numpy(batch).float().to(device)
            logits = model(tensor)
            prob = torch.sigmoid(logits).cpu().numpy().flatten()
            preds.extend(prob)
    return np.array(preds)


def load_day_parquet(path):
    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else df.index
    if df.index.dtype == 'O':
        df = df.set_index('timestamp')
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='ml/models/enhanced_production.pt')
    p.add_argument('--scaler', default='ml/models/enhanced_gex_scaler.pkl', help='Optional scaler path to apply to data')
    p.add_argument('--days', default=None)
    p.add_argument('--window', type=int, default=60)
    p.add_argument('--horizon', type=int, default=1)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--contract_value', type=float, default=5.0)
    p.add_argument('--commission', type=float, default=None, help='Commission roundtrip per trade (overrides instrument)')
    p.add_argument('--commission_per_side', type=float, default=None, help='Commission per side (single contract). Roundtrip = per_side*2')
    p.add_argument('--instrument', default='MNQ', help='Instrument (MNQ or NQ) to set defaults')
    p.add_argument('--slippage', type=float, default=0.0)
    p.add_argument('--position_size', type=int, default=1, help='Number of contracts per trade')
    p.add_argument('--stop_loss', type=float, default=None, help='Stop loss in $ per contract')
    p.add_argument('--take_profit', type=float, default=None, help='Take profit in $ per contract')
    p.add_argument('--mlflow', action='store_true', help='Enable MLflow logging for this run')
    p.add_argument('--experiment', default='pnl_backtest', help='MLflow experiment name')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to run model on')
    p.add_argument('--batch-size', default=256, type=int, help='Batch size for inference')
    p.add_argument('--sample', default=None, type=int, help='If set, limit the number of windows (predictions) to this value')
    p.add_argument('--stride', default=1, type=int, help='Downsampling stride to reduce windows (use 2 to process every 2nd window)')
    p.add_argument('--bar-type', default='time', choices=['time', 'volume', 'dollar'], help='Type of bars used in preprocessed files')
    p.add_argument('--bar-size', default=1, type=float, help='Size parameter for bars (seconds for time; threshold for volume/dollar)')
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    model, scaler, ckpt = load_model_and_scaler(str(model_path))
    # If user provided scaler path try to load
    if args.scaler:
        scaler_path = Path(args.scaler)
        if scaler_path.exists():
            try:
                import joblib
                scaler = joblib.load(scaler_path)
                print('Loaded scaler from', scaler_path)
            except Exception as e:
                print('Could not load scaler from', scaler_path, e)
    # Device handling
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Determine days to test
    if args.days:
        days = [d.strip() for d in args.days.split(',') if d.strip()]
    else:
        # default use the Oct week output
        days = ['20251020', '20251021', '20251022', '20251023', '20251024']

    outputs = []
    # Default commission mapping: per-side values
    COMM_PER_SIDE = {'MNQ': 0.21, 'NQ': 0.84, 'NQ_NDX': 0.84}
    # Determine commission in roundtrip terms
    if args.commission_per_side is not None:
        commission = args.commission_per_side * 2.0
    elif args.commission is not None:
        commission = args.commission
    else:
        commission = COMM_PER_SIDE.get(args.instrument.upper(), 0.21) * 2.0

    for d in days:
        # Determine filename suffix based on bar type and size
        if args.bar_type == 'time' and int(args.bar_size) == 1:
            suffix = '1s'
        else:
            suffix = f"{args.bar_type}{int(args.bar_size) if args.bar_type in ['time','volume'] else args.bar_size}"
        input_file = Path(f'ml/output/{args.instrument}_{d}_{suffix}.parquet')
        if not input_file.exists():
            print(f"Skipping missing input: {input_file}")
            continue
        print(f"Processing day {d}...")
        # Determine required feature count from model (for LSTM input size) if possible
        required_feature_count = None
        try:
            if hasattr(model, 'rnn') and hasattr(model.rnn, 'input_size'):
                required_feature_count = model.rnn.input_size
            elif hasattr(model, 'input_dim'):
                required_feature_count = getattr(model, 'input_dim')
        except Exception:
            required_feature_count = None

        X, y = preprocess_day_data(str(input_file), scaler, sequence_length=args.window, required_feature_count=required_feature_count, stride=args.stride, max_samples=args.sample)
        # Optionally downsample to reduce memory
        if args.stride and args.stride > 1:
            X = X[::args.stride]
            y = y[::args.stride]
        # Optionally limit sample count (first N windows)
        if args.sample is not None and args.sample > 0:
            X = X[:args.sample]
            y = y[:args.sample]
        if X is None:
            print("No data available after preprocessing")
            continue
        # load closes for PnL mapping
        df = load_day_parquet(input_file)
        # Ensure length alignment: preprocess_day_data uses resampled 1s OHLC; ensure df has 'close'
        closes = df['close']
        preds = predict_from_model(model, X, device, batch_size=args.batch_size)
        # Only keep predictions above threshold, else 0
        # compute price-based PnL
        result = simulate_price_pnl(closes, (preds >= args.threshold).astype(float), args.window, horizon=args.horizon, contract_value=args.contract_value, commission=commission, slippage=args.slippage)
        result['date'] = d
        outputs.append(result)
        print(f"Day {d} trades: {result['trades']}, win_rate: {result['win_rate']:.2%}, pnl: ${result['total_pnl']:.2f}")

    # Summarize
    total_trades = sum(r['trades'] for r in outputs)
    total_pnl = sum(r['total_pnl'] for r in outputs)
    avg_win_rate = np.mean([r['win_rate'] for r in outputs]) if outputs else 0
    print("\nSUMMARY")
    print(f"Days processed: {len(outputs)}")
    print(f"Total trades: {total_trades}")
    print(f"Total PnL: ${total_pnl:.2f}")
    print(f"Average win rate: {avg_win_rate:.2%}")

    # MLflow logging
    if args.mlflow and mlflow is not None:
        try:
            import mlflow_utils
            mlflow_utils.ensure_sqlite_tracking()
        except Exception:
            pass
        mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name=f"pnl_backtest_{args.instrument}_{','.join(days)}"):
            mlflow.log_param('model', str(model_path))
            mlflow.log_param('scaler', args.scaler)
            mlflow.log_param('days', ','.join(days))
            mlflow.log_param('window', args.window)
            mlflow.log_param('horizon', args.horizon)
            mlflow.log_param('threshold', args.threshold)
            mlflow.log_param('instrument', args.instrument)
            mlflow.log_param('commission_per_side', args.commission_per_side)
            mlflow.log_param('commission_roundtrip', commission)
            mlflow.log_metric('total_trades', int(total_trades))
            mlflow.log_metric('total_pnl', float(total_pnl))
            mlflow.log_metric('avg_win_rate', float(avg_win_rate))

            # Save CSV summary and log as artifact
            import csv
            summary_path = Path('ml_backtest_summary.csv')
            with open(summary_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['date', 'trades', 'win_rate', 'total_pnl', 'avg_pnl'])
                writer.writeheader()
                for r in outputs:
                    writer.writerow({'date': r['date'], 'trades': r['trades'], 'win_rate': r['win_rate'], 'total_pnl': r['total_pnl'], 'avg_pnl': r['avg_pnl']})
            try:
                mlflow.log_artifact(str(summary_path))
            except Exception:
                pass
    elif args.mlflow and mlflow is None:
        print('MLflow requested but mlflow is not installed. Skipping logging.')


if __name__ == '__main__':
    main()
