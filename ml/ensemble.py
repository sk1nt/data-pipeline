#!/usr/bin/env python3
"""Ensemble of LSTM and LightGBM for price movement prediction with MLflow logging.

Loads trained models and combines predictions via weighted averaging.
"""
import argparse
import numpy as np
import torch
import joblib
from pathlib import Path
import mlflow
import mlflow.pytorch
import mlflow.lightgbm

# Assuming models are saved
LSTM_MODEL = Path('models/lstm_large_dropout.pt')
LSTM_CV_MODELS = [Path(f'models/lstm_cv_fold{i}.pt') for i in range(5)]
LGB_MODEL = Path('models/lightgbm_tuned.pkl')
DATA = Path('output/MNQ_20251111_1s_w60s_h1.npz')

# Dummy model classes (copy from train scripts)
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

def load_models(use_cv=False):
    # Load data to get input_dim
    data = np.load(DATA)
    X = data['X']
    y = data['y']
    input_dim = X.shape[2]

    # Recreate the train/val split used in training
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    if use_cv:
        # Load CV LSTM models
        lstm_models = []
        for cv_model in LSTM_CV_MODELS:
            model = LSTMModel(input_dim, hidden_dim=256, num_layers=2, model_type='lstm')
            model.load_state_dict(torch.load(cv_model))
            model.eval()
            lstm_models.append(model)
        lstm_model = lstm_models  # list of models
    else:
        # Load single LSTM
        lstm_model = LSTMModel(input_dim, hidden_dim=256, num_layers=2, model_type='lstm')
        lstm_model.load_state_dict(torch.load(LSTM_MODEL))
        lstm_model.eval()

    # Load LightGBM
    lgb_model = joblib.load(LGB_MODEL)

    return lstm_model, lgb_model, X_val, y_val

def predict_ensemble(models, X, method='average', weights=None, batch_size=1024):
    lstm_model, lgb_model = models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(lstm_model, list):
        # CV models
        lstm_models = lstm_model
        for m in lstm_models:
            m.to(device)
    else:
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
        else:
            with torch.no_grad():
                logits = lstm_model(batch_tensor)
                lstm_pred = torch.sigmoid(logits).cpu().numpy().flatten()

        # LightGBM on flattened batch (since it's tree-based)
        batch_flat = batch.reshape(batch.shape[0], -1)
        lgb_pred = lgb_model.predict(batch_flat)

        if method == 'average':
            if weights is None:
                weights = [0.5, 0.5]
            ensemble_pred = (weights[0] * lstm_pred + weights[1] * lgb_pred) > 0.5
        elif method == 'vote':
            lstm_binary = (lstm_pred > 0.5).astype(int)
            lgb_binary = (lgb_pred > 0.5).astype(int)
            ensemble_pred = (lstm_binary + lgb_binary) >= 1  # majority vote for 2 models
        else:
            raise ValueError(f"Unknown method: {method}")

        preds.extend(ensemble_pred.astype(int))

def predict_ensemble(models, X, method='average', weights=None, batch_size=1024):
    lstm_model, lgb_model = models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(lstm_model, list):
        # CV models
        lstm_models = lstm_model
        for m in lstm_models:
            m.to(device)
    else:
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
        else:
            with torch.no_grad():
                logits = lstm_model(batch_tensor)
                lstm_pred = torch.sigmoid(logits).cpu().numpy().flatten()

        # LightGBM on flattened batch (since it's tree-based)
        batch_flat = batch.reshape(batch.shape[0], -1)
        lgb_pred = lgb_model.predict(batch_flat)

        if method == 'average':
            if weights is None:
                weights = [0.5, 0.5]
            ensemble_pred = (weights[0] * lstm_pred + weights[1] * lgb_pred) > 0.5
        elif method == 'vote':
            lstm_binary = (lstm_pred > 0.5).astype(int)
            lgb_binary = (lgb_pred > 0.5).astype(int)
            ensemble_pred = (lstm_binary + lgb_binary) >= 1  # majority vote for 2 models
        else:
            raise ValueError(f"Unknown method: {method}")

        preds.extend(ensemble_pred.astype(int))

    return np.array(preds)

def calculate_trading_metrics(preds, labels, commission_cost=0.42, avg_profit_per_win=10.0):
    """Calculate trading performance metrics with commission costs."""
    preds_binary = preds.astype(int)
    labels_binary = labels.astype(int)

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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlflow', action='store_true', help='Log results to MLflow')
    parser.add_argument('--experiment', default='ensemble_experiments', help='MLflow experiment name')
    args = parser.parse_args()
    
    if args.mlflow:
        mlflow.set_experiment(args.experiment)

    # Test single model ensemble
    print("Single model ensemble:")
    models = load_models(use_cv=False)
    lstm_model, lgb_model, X_val, y_val = models
    labels = (y_val > 0).astype(int)

    configs = [
        ('average', [0.5, 0.5], 'Equal average'),
        ('average', [0.3, 0.7], 'Weighted average (LSTM 0.3, LGB 0.7)'),
        ('average', [0.1, 0.9], 'Weighted average (LSTM 0.1, LGB 0.9)'),
        ('vote', None, 'Majority vote'),
    ]

    for method, weights, desc in configs:
        preds = predict_ensemble((lstm_model, lgb_model), X_val, method=method, weights=weights)
        acc = np.mean(preds == labels)
        trading_metrics = calculate_trading_metrics(preds, labels)
        
        print(f'  {desc}:')
        print(f'    Accuracy: {acc:.4f}')
        print(f'    Win Rate: {trading_metrics["win_rate"]:.1%}')
        print(f'    Net P&L: ${trading_metrics["net_pnl"]:.2f} ({trading_metrics["trades_taken"]} trades)')
        print(f'    Edge: {trading_metrics["edge"]:.1%}')
        print()
        
        if args.mlflow:
            from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
            cm = confusion_matrix(labels, preds)
            run_name = f"Single_{desc.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '_')}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("ensemble_type", "single")
                mlflow.log_param("method", method)
                if weights:
                    mlflow.log_param("lstm_weight", weights[0])
                    mlflow.log_param("lgb_weight", weights[1])
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision_down", precision[0])
                mlflow.log_metric("precision_up", precision[1])
                mlflow.log_metric("recall_down", recall[0])
                mlflow.log_metric("recall_up", recall[1])
                mlflow.log_metric("f1_down", f1[0])
                mlflow.log_metric("f1_up", f1[1])
                mlflow.log_metric("tn", cm[0,0])
                mlflow.log_metric("fp", cm[0,1])
                mlflow.log_metric("fn", cm[1,0])
                mlflow.log_metric("tp", cm[1,1])
                # Log trading metrics
                for key, value in trading_metrics.items():
                    mlflow.log_metric(key, value)

    # Test CV ensemble
    print("\nCV ensemble:")
    models_cv = load_models(use_cv=True)
    lstm_models, lgb_model, X_val, y_val = models_cv

    for method, weights, desc in configs:
        preds = predict_ensemble((lstm_models, lgb_model), X_val, method=method, weights=weights)
        acc = np.mean(preds == labels)
        trading_metrics = calculate_trading_metrics(preds, labels)
        
        print(f'  {desc}:')
        print(f'    Accuracy: {acc:.4f}')
        print(f'    Win Rate: {trading_metrics["win_rate"]:.1%}')
        print(f'    Net P&L: ${trading_metrics["net_pnl"]:.2f} ({trading_metrics["trades_taken"]} trades)')
        print(f'    Edge: {trading_metrics["edge"]:.1%}')
        print()
        
        if args.mlflow:
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
            cm = confusion_matrix(labels, preds)
            run_name = f"CV_{desc.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '_')}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("ensemble_type", "cv")
                mlflow.log_param("method", method)
                if weights:
                    mlflow.log_param("lstm_weight", weights[0])
                    mlflow.log_param("lgb_weight", weights[1])
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision_down", precision[0])
                mlflow.log_metric("precision_up", precision[1])
                mlflow.log_metric("recall_down", recall[0])
                mlflow.log_metric("recall_up", recall[1])
                mlflow.log_metric("f1_down", f1[0])
                mlflow.log_metric("f1_up", f1[1])
                mlflow.log_metric("tn", cm[0,0])
                mlflow.log_metric("fp", cm[0,1])
                mlflow.log_metric("fn", cm[1,0])
                mlflow.log_metric("tp", cm[1,1])
                # Log trading metrics
                for key, value in trading_metrics.items():
                    mlflow.log_metric(key, value)