#!/usr/bin/env python3
"""Comprehensive model testing across multiple days and model types.

Tests various ML models on different days of data with trading analysis.
"""
import argparse
import numpy as np
import torch
import joblib
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
import mlflow
import mlflow.pytorch
import mlflow.lightgbm
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Model configurations
MODEL_CONFIGS = {
    # LSTM variants
    'lstm_production': {
        'path': 'models/lstm_production.pt',
        'type': 'lstm',
        'input_dim': 13,  # From the working dataset
        'hidden_dim': 256,
        'num_layers': 2
    },
    'lstm_large_dropout': {
        'path': 'models/lstm_large_dropout.pt',
        'type': 'lstm',
        'input_dim': 13,
        'hidden_dim': 256,
        'num_layers': 2
    },
    'lstm_cv_fold0': {
        'path': 'models/lstm_cv_fold0.pt',
        'type': 'lstm',
        'input_dim': 13,
        'hidden_dim': 256,
        'num_layers': 2
    },

    # LightGBM variants
    'lightgbm_tuned': {
        'path': 'models/lightgbm_tuned.pkl',
        'type': 'lightgbm'
    },
    'lightgbm_compare_20251119T004816': {
        'path': 'models/lightgbm_compare_20251119T004816.pkl',
        'type': 'lightgbm'
    },

    # XGBoost variants
    'xgboost_compare_20251119T004816': {
        'path': 'models/xgboost_compare_20251119T004816.pkl',
        'type': 'xgboost'
    },

    # CNN variants
    'cnn_compare_20251119T004816': {
        'path': 'models/cnn_compare_20251119T004816.pt',
        'type': 'cnn',
        'input_dim': 13
    },

    # TCN variants
    'tcn_compare_20251119T004912': {
        'path': 'models/tcn_compare_20251119T004912.pt',
        'type': 'tcn',
        'input_dim': 13
    },

    # Transformer variants
    'transformer_compare_20251119T004912': {
        'path': 'models/transformer_compare_20251119T004912.pt',
        'type': 'transformer',
        'input_dim': 13
    },

    # GRU variants
    'gru_compare_20251119T005530': {
        'path': 'models/gru_compare_20251119T005530.pt',
        'type': 'gru',
        'input_dim': 13,
        'hidden_dim': 256,
        'num_layers': 2
    }
}

# Dataset configurations
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

class CNNModel(torch.nn.Module):
    def __init__(self, input_dim, seq_len=60):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class TCNModel(torch.nn.Module):
    def __init__(self, input_dim, seq_len=60):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, 64, kernel_size=3, dilation=1, padding=1)
        self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=3, dilation=2, padding=2)
        self.conv3 = torch.nn.Conv1d(128, 64, kernel_size=3, dilation=4, padding=4)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class TransformerModel(torch.nn.Module):
    def __init__(self, input_dim, seq_len=60):
        super().__init__()
        self.embedding = torch.nn.Linear(input_dim, 64)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

def load_model(model_config, input_dim):
    """Load a model based on its configuration."""
    model_path = Path(model_config['path'])

    if not model_path.exists():
        return None

    model_type = model_config['type']

    if model_type in ['lstm', 'gru']:
        model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=model_config.get('hidden_dim', 64),
            num_layers=model_config.get('num_layers', 1),
            model_type=model_type
        )
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model

    elif model_type == 'cnn':
        model = CNNModel(input_dim=input_dim)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model

    elif model_type == 'tcn':
        model = TCNModel(input_dim=input_dim)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model

    elif model_type == 'transformer':
        model = TransformerModel(input_dim=input_dim)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model

    elif model_type == 'lightgbm':
        return joblib.load(model_path)

    elif model_type == 'xgboost':
        return xgb.Booster(model_file=model_path)

    return None

def predict_model(model, X, model_type, batch_size=1024):
    """Make predictions with a model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type in ['lstm', 'gru', 'cnn', 'tcn', 'transformer']:
        model.to(device)
        preds = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            with torch.no_grad():
                logits = model(batch_tensor)
                pred = torch.sigmoid(logits).cpu().numpy().flatten()
                preds.extend(pred)
        return np.array(preds)

    elif model_type == 'lightgbm':
        # Flatten for tree-based model
        X_flat = X.reshape(X.shape[0], -1)
        return model.predict(X_flat)

    elif model_type == 'xgboost':
        # Flatten for tree-based model
        X_flat = X.reshape(X.shape[0], -1)
        dmatrix = xgb.DMatrix(X_flat)
        return model.predict(dmatrix)

    return None

def calculate_trading_metrics(preds, labels, commission_cost=0.42, avg_profit_per_win=10.0):
    """Calculate trading performance metrics with commission costs."""
    preds_binary = (preds > 0.5).astype(int)
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
    parser.add_argument('--mlflow_experiment', default='multi_day_multi_model_test', help='MLflow experiment name')
    parser.add_argument('--max_models', type=int, default=None, help='Limit number of models to test')
    parser.add_argument('--commission_cost', type=float, default=0.42, help='Commission cost per trade')
    args = parser.parse_args()

    try:
        import mlflow_utils
        mlflow_utils.ensure_sqlite_tracking()
    except Exception:
        pass
    mlflow.set_experiment(args.mlflow_experiment)

    results = []

    # Test each dataset
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

        print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

        # Test each model
        models_to_test = list(MODEL_CONFIGS.items())
        if args.max_models:
            models_to_test = models_to_test[:args.max_models]

        for model_name, model_config in models_to_test:
            print(f"\nTesting {model_name}...")

            try:
                # Load model
                model = load_model(model_config, input_dim)
                if model is None:
                    print(f"  Model {model_name} not found, skipping")
                    continue

                # Make predictions on test set
                preds = predict_model(model, X_test, model_config['type'])
                if preds is None:
                    print(f"  Failed to get predictions for {model_name}")
                    continue

                # Calculate metrics
                labels_binary = (y_test > 0).astype(int)
                preds_binary = (preds > 0.5).astype(int)

                accuracy = np.mean(preds_binary == labels_binary)
                precision, recall, f1, _ = precision_recall_fscore_support(labels_binary, preds_binary, average='binary')
                cm = confusion_matrix(labels_binary, preds_binary)

                # Trading metrics
                trading_metrics = calculate_trading_metrics(preds, y_test, commission_cost=args.commission_cost)

                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Win Rate: {trading_metrics['win_rate']:.1%}")
                print(f"  Net P&L: ${trading_metrics['net_pnl']:.2f} ({trading_metrics['trades_taken']} trades)")
                print(f"  Edge: {trading_metrics['edge']:.1%}")

                # Log to MLflow
                with mlflow.start_run(run_name=f"{Path(dataset_path).stem}_{model_name}"):
                    mlflow.log_param("dataset", dataset_path)
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("model_type", model_config['type'])
                    mlflow.log_param("input_dim", input_dim)
                    mlflow.log_param("commission_cost", args.commission_cost)

                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("precision", precision)
                    mlflow.log_metric("recall", recall)
                    mlflow.log_metric("f1", f1)
                    mlflow.log_metric("tn", cm[0,0])
                    mlflow.log_metric("fp", cm[0,1])
                    mlflow.log_metric("fn", cm[1,0])
                    mlflow.log_metric("tp", cm[1,1])

                    # Log trading metrics
                    for key, value in trading_metrics.items():
                        mlflow.log_metric(key, value)

                # Store results
                results.append({
                    'dataset': Path(dataset_path).stem,
                    'model': model_name,
                    'type': model_config['type'],
                    'accuracy': accuracy,
                    'precision': precision,
                    'win_rate': trading_metrics['win_rate'],
                    'net_pnl': trading_metrics['net_pnl'],
                    'trades_taken': trading_metrics['trades_taken'],
                    'edge': trading_metrics['edge']
                })

            except Exception as e:
                print(f"  Error testing {model_name}: {e}")
                continue

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)

    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('edge', ascending=False)

        print("\nTop 10 models by trading edge:")
        for _, row in df_results.head(10).iterrows():
            print(f"{row['dataset'][:15]} | {row['model'][:20]} | {row['type'][:10]} | "
                  f"Edge: {row['edge']:.1%} | Win: {row['win_rate']:.1%} | "
                  f"P&L: ${row['net_pnl']:.0f} ({row['trades_taken']})")

        print(f"\nTotal models tested: {len(results)}")
        print(f"Average edge: {df_results['edge'].mean():.1%}")
        print(f"Best edge: {df_results['edge'].max():.1%}")

        # Save results to CSV
        df_results.to_csv('multi_model_test_results.csv', index=False)
        print("Results saved to multi_model_test_results.csv")

if __name__ == '__main__':
    main()