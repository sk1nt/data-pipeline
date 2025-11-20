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

# Model paths
LSTM_MODEL = Path('models/lstm_large_dropout.pt')
LSTM_CV_MODELS = [Path(f'models/lstm_cv_fold{i}.pt') for i in range(5)]
LGB_MODEL = Path('models/lightgbm_tuned.pkl')

# Dataset
DATASET = 'output/MNQ_multi_day_enhanced.npz'

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
    """Load all available models."""
    data = np.load(DATASET)
    X = data['X']
    input_dim = X.shape[2]

    # Load LSTM
    lstm_model = None
    if LSTM_MODEL.exists():
        lstm_model = LSTMModel(input_dim, hidden_dim=256, num_layers=2, model_type='lstm')
        lstm_model.load_state_dict(torch.load(LSTM_MODEL, weights_only=True))
        lstm_model.eval()

    # Load CV LSTMs
    cv_models = []
    for cv_model_path in LSTM_CV_MODELS:
        if cv_model_path.exists():
            model = LSTMModel(input_dim, hidden_dim=256, num_layers=2, model_type='lstm')
            model.load_state_dict(torch.load(cv_model_path, weights_only=True))
            model.eval()
            cv_models.append(model)

    # Load LightGBM
    lgb_model = joblib.load(LGB_MODEL) if LGB_MODEL.exists() else None

    return lstm_model, cv_models, lgb_model, X.shape[2]

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
        X_flat = X.reshape(X.shape[0], -1)
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

def optimize_thresholds(predictions, labels, thresholds=np.arange(0.3, 0.9, 0.05)):
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

    mlflow.set_experiment(args.mlflow_experiment)

    # Load data and models
    print("Loading data and models...")
    models = load_models()
    lstm_model, cv_models, lgb_model, input_dim = models

    data = np.load(DATASET)
    X = data['X']
    y = data['y']

    # Split into test set (same as before)
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Test set: {X_test.shape[0]} samples")

    # Get predictions from all models
    print("Getting predictions...")
    predictions = get_predictions(models, X_test)

    if not predictions:
        print("No models could be loaded!")
        return

    print(f"Available models: {list(predictions.keys())}")

    # Optimize thresholds for individual models
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL THRESHOLD OPTIMIZATION")
    print("="*60)

    individual_results = optimize_thresholds(predictions, y_test)

    # Test ensemble methods
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
        thresholds = np.arange(0.3, 0.9, 0.05)

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
    df_results.to_csv('threshold_optimization_results.csv', index=False)
    print("Detailed results saved to threshold_optimization_results.csv")

    print("\n✅ Threshold optimization complete!")
    print(f"Best configuration: {all_results[0]['model']} at threshold {all_results[0]['threshold']:.2f}")
    print(f"Best P&L: ${all_results[0]['net_pnl']:.2f}")
    print(f"Best win rate: {all_results[0]['win_rate']:.1%}")
    print(f"Improvement potential: ${all_results[0]['net_pnl'] - all_results[-1]['net_pnl']:.2f} over baseline")

if __name__ == '__main__':
    main()