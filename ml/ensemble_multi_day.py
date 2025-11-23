#!/usr/bin/env python3
"""Test ensemble performance across multiple days of data.

Compares ensemble methods on different trading days to assess robustness.
"""
import argparse
import numpy as np
import torch
import joblib
from pathlib import Path
import mlflow
import mlflow.pytorch
import mlflow.lightgbm

# Model paths (same as ensemble.py)
LSTM_MODEL = Path('models/lstm_large_dropout.pt')
LSTM_CV_MODELS = [Path(f'models/lstm_cv_fold{i}.pt') for i in range(5)]
LGB_MODEL = Path('models/lightgbm_tuned.pkl')

# Dataset paths
DATASETS = [
    'output/MNQ_2025-11-10_1s_w60s_h1.npz',  # 5 features (OHLCV only)
    'output/MNQ_2025-11-11_1s_w60s_h1.npz',  # 13 features (with technical indicators)
]

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

def load_models_for_dataset(input_dim, use_cv=False):
    """Load models configured for a specific input dimension."""
    if use_cv:
        lstm_models = []
        for cv_model in LSTM_CV_MODELS:
            if cv_model.exists():
                model = LSTMModel(input_dim, hidden_dim=256, num_layers=2, model_type='lstm')
                model.load_state_dict(torch.load(cv_model, weights_only=True))
                model.eval()
                lstm_models.append(model)
        lstm_model = lstm_models if lstm_models else None
    else:
        if LSTM_MODEL.exists():
            lstm_model = LSTMModel(input_dim, hidden_dim=256, num_layers=2, model_type='lstm')
            lstm_model.load_state_dict(torch.load(LSTM_MODEL, weights_only=True))
            lstm_model.eval()
        else:
            lstm_model = None

    lgb_model = joblib.load(LGB_MODEL) if LGB_MODEL.exists() else None

    return lstm_model, lgb_model

def predict_ensemble(models, X, method='average', weights=None, batch_size=1024):
    """Make ensemble predictions."""
    lstm_model, lgb_model = models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(lstm_model, list):
        lstm_models = lstm_model
        for m in lstm_models:
            m.to(device)
    else:
        if lstm_model:
            lstm_model.to(device)

    preds = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        batch_tensor = torch.from_numpy(batch).float().to(device)

        if isinstance(lstm_model, list):
            # Average predictions from CV models
            lstm_preds = []
            for m in lstm_models:
                with torch.no_grad():
                    logits = m(batch_tensor)
                    pred = torch.sigmoid(logits).cpu().numpy().flatten()
                    lstm_preds.append(pred)
            lstm_pred = np.mean(lstm_preds, axis=0)
        elif lstm_model:
            with torch.no_grad():
                logits = lstm_model(batch_tensor)
                lstm_pred = torch.sigmoid(logits).cpu().numpy().flatten()
        else:
            lstm_pred = np.zeros(len(batch))  # No LSTM predictions

        # LightGBM on flattened batch
        batch_flat = batch.reshape(batch.shape[0], -1)
        if lgb_model:
            lgb_pred = lgb_model.predict(batch_flat)
        else:
            lgb_pred = np.zeros(len(batch))  # No LightGBM predictions

        if method == 'average':
            if weights is None:
                if lstm_model and lgb_model:
                    weights = [0.5, 0.5]
                elif lstm_model:
                    weights = [1.0, 0.0]
                else:
                    weights = [0.0, 1.0]
            ensemble_pred = (weights[0] * lstm_pred + weights[1] * lgb_pred) > 0.5
        elif method == 'vote' and lstm_model and lgb_model:
            lstm_binary = (lstm_pred > 0.5).astype(int)
            lgb_binary = (lgb_pred > 0.5).astype(int)
            ensemble_pred = (lstm_binary + lgb_binary) >= 1
        else:
            ensemble_pred = (lgb_pred > 0.5) if lgb_model else (lstm_pred > 0.5)

        preds.extend(ensemble_pred.astype(int))

    return np.array(preds)

def calculate_trading_metrics(preds, labels, commission_cost=0.42, avg_profit_per_win=10.0):
    """Calculate trading performance metrics."""
    preds_binary = preds.astype(int)
    labels_binary = (labels > 0).astype(int)

    trades_taken = np.sum(preds_binary == 1)
    correct_trades = np.sum((preds_binary == 1) & (labels_binary == 1))
    win_rate = correct_trades / trades_taken if trades_taken > 0 else 0

    total_commissions = trades_taken * commission_cost
    gross_pnl = correct_trades * avg_profit_per_win
    net_pnl = gross_pnl - total_commissions

    pnl_per_trade = net_pnl / trades_taken if trades_taken > 0 else 0
    break_even_win_rate = commission_cost / avg_profit_per_win
    edge = win_rate - break_even_win_rate

    return {
        'trades_taken': trades_taken,
        'win_rate': win_rate,
        'total_commissions': total_commissions,
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        'pnl_per_trade': pnl_per_trade,
        'break_even_win_rate': break_even_win_rate,
        'edge': edge
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlflow_experiment', default='ensemble_multi_day_comparison', help='MLflow experiment name')
    parser.add_argument('--commission_cost', type=float, default=0.42, help='Commission cost per trade')
    args = parser.parse_args()

        try:
            import mlflow_utils
            mlflow_utils.ensure_sqlite_tracking()
        except Exception:
            pass
        mlflow.set_experiment(args.mlflow_experiment)

    results = []

    # Test configurations
    configs = [
        ('cv_weighted_0.3_0.7', True, 'average', [0.3, 0.7], 'CV LSTM 0.3 + LGB 0.7'),
        ('cv_equal', True, 'average', [0.5, 0.5], 'CV LSTM + LGB equal'),
        ('single_weighted_0.3_0.7', False, 'average', [0.3, 0.7], 'Single LSTM 0.3 + LGB 0.7'),
        ('single_equal', False, 'average', [0.5, 0.5], 'Single LSTM + LGB equal'),
        ('cv_majority_vote', True, 'vote', None, 'CV Majority Vote'),
        ('single_lstm_only', False, 'average', [1.0, 0.0], 'LSTM Only'),
        ('lightgbm_only', False, 'average', [0.0, 1.0], 'LightGBM Only'),
    ]

    for dataset_path in DATASETS:
        print(f"\n=== Testing Dataset: {dataset_path} ===")

        # Load dataset
        data = np.load(dataset_path)
        X = data['X']
        y = data['y']
        input_dim = X.shape[2]

        print(f"Dataset shape: {X.shape}, Input dim: {input_dim}")

        # Split into train/val/test (using same split as before)
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        print(f"Test set: {X_test.shape[0]} samples")

        # Load models for this dataset
        for config_name, use_cv, method, weights, desc in configs:
            print(f"\nTesting {desc}...")

            try:
                lstm_model, lgb_model = load_models_for_dataset(input_dim, use_cv=use_cv)

                if not lstm_model and not lgb_model:
                    print("  No models available for this configuration")
                    continue

                models = (lstm_model, lgb_model)
                preds = predict_ensemble(models, X_test, method=method, weights=weights)

                # Calculate metrics
                accuracy = np.mean(preds == (y_test > 0).astype(int))
                trading_metrics = calculate_trading_metrics(preds, y_test, commission_cost=args.commission_cost)

                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Win Rate: {trading_metrics['win_rate']:.1%}")
                print(f"  Net P&L: ${trading_metrics['net_pnl']:.2f} ({trading_metrics['trades_taken']} trades)")
                print(f"  Edge: {trading_metrics['edge']:.1%}")

                # Log to MLflow
                with mlflow.start_run(run_name=f"{Path(dataset_path).stem}_{config_name}"):
                    mlflow.log_param("dataset", dataset_path)
                    mlflow.log_param("config", config_name)
                    mlflow.log_param("description", desc)
                    mlflow.log_param("use_cv", use_cv)
                    mlflow.log_param("method", method)
                    mlflow.log_param("input_dim", input_dim)
                    mlflow.log_param("commission_cost", args.commission_cost)
                    if weights:
                        mlflow.log_param("lstm_weight", weights[0])
                        mlflow.log_param("lgb_weight", weights[1])

                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("trades_taken", trading_metrics['trades_taken'])
                    mlflow.log_metric("win_rate", trading_metrics['win_rate'])
                    mlflow.log_metric("net_pnl", trading_metrics['net_pnl'])
                    mlflow.log_metric("total_commissions", trading_metrics['total_commissions'])
                    mlflow.log_metric("edge", trading_metrics['edge'])

                # Store results
                results.append({
                    'dataset': Path(dataset_path).stem,
                    'config': config_name,
                    'description': desc,
                    'accuracy': accuracy,
                    'win_rate': trading_metrics['win_rate'],
                    'net_pnl': trading_metrics['net_pnl'],
                    'trades_taken': trading_metrics['trades_taken'],
                    'edge': trading_metrics['edge']
                })

            except Exception as e:
                print(f"  Error testing {desc}: {e}")
                continue

    # Print summary
    print("\n" + "="*100)
    print("ENSEMBLE PERFORMANCE ACROSS MULTIPLE DAYS")
    print("="*100)

    if results:
        import pandas as pd
        df_results = pd.DataFrame(results)

        # Group by configuration and show average performance across days
        config_summary = df_results.groupby('config').agg({
            'edge': ['mean', 'std'],
            'win_rate': ['mean', 'std'],
            'net_pnl': ['mean', 'sum'],
            'trades_taken': ['mean', 'sum'],
            'accuracy': 'mean'
        }).round(4)

        print("\nAverage Performance by Configuration:")
        print(config_summary.sort_values(('edge', 'mean'), ascending=False))

        # Best configurations per day
        print("\nBest Configuration per Dataset:")
        for dataset in df_results['dataset'].unique():
            day_results = df_results[df_results['dataset'] == dataset]
            best = day_results.loc[day_results['edge'].idxmax()]
            print(f"{dataset}: {best['description']} | Edge: {best['edge']:.1%} | Win: {best['win_rate']:.1%} | P&L: ${best['net_pnl']:.0f}")

        # Overall best
        overall_best = df_results.loc[df_results['edge'].idxmax()]
        print(f"\nOverall Best: {overall_best['description']} on {overall_best['dataset']}")
        print(f"Edge: {overall_best['edge']:.1%} | Win Rate: {overall_best['win_rate']:.1%} | Net P&L: ${overall_best['net_pnl']:.0f}")

        # Save detailed results
        df_results.to_csv('ensemble_multi_day_results.csv', index=False)
        print("\nDetailed results saved to ensemble_multi_day_results.csv")

if __name__ == '__main__':
    main()