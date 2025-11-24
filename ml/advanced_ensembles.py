#!/usr/bin/env python3
"""Advanced ensemble methods: stacking and blending for improved predictions.

Implements meta-model stacking and sophisticated blending approaches.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
import mlflow
import mlflow.pytorch
import mlflow.lightgbm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Model paths
LSTM_MODEL = Path('ml/models/lstm_large_dropout.pt')
LSTM_CV_MODELS = [Path(f'ml/models/lstm_cv_fold{i}.pt') for i in range(5)]
LGB_MODEL = Path('ml/models/lightgbm_tuned.pkl')

# Dataset
DATASET = 'output/MNQ_2025-11-11_1s_w60s_h1.npz'

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

class StackingMetaModel(nn.Module):
    """Neural network meta-model for stacking."""
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def load_base_models():
    """Load all base models."""
    data = np.load(DATASET)
    X = data['X']
    input_dim = X.shape[2]

    models = {}

    # Load LSTM
    if LSTM_MODEL.exists():
        lstm_model = LSTMModel(input_dim, hidden_dim=256, num_layers=2, model_type='lstm')
        lstm_model.load_state_dict(torch.load(LSTM_MODEL, weights_only=True))
        lstm_model.eval()
        models['lstm'] = lstm_model

    # Load CV LSTMs
    cv_models = []
    for cv_model_path in LSTM_CV_MODELS:
        if cv_model_path.exists():
            model = LSTMModel(input_dim, hidden_dim=256, num_layers=2, model_type='lstm')
            model.load_state_dict(torch.load(cv_model_path, weights_only=True))
            model.eval()
            cv_models.append(model)
    if cv_models:
        models['cv_lstms'] = cv_models

    # Load LightGBM
    if LGB_MODEL.exists():
        models['lightgbm'] = joblib.load(LGB_MODEL)

    return models, input_dim

def get_base_predictions(models, X, batch_size=1024):
    """Get predictions from all base models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions = {}

    # LSTM predictions
    if 'lstm' in models:
        lstm_model = models['lstm']
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

    # CV LSTM predictions (individual and averaged)
    if 'cv_lstms' in models:
        cv_models = models['cv_lstms']
        cv_individual_preds = []
        cv_avg_preds = []

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

            # Store individual fold predictions
            cv_individual_preds.extend(np.array(fold_preds).T)

            # Store averaged predictions
            avg_pred = np.mean(fold_preds, axis=0)
            cv_avg_preds.extend(avg_pred)

        predictions['cv_lstm_avg'] = np.array(cv_avg_preds)
        predictions['cv_lstm_individual'] = np.array(cv_individual_preds)

    # LightGBM predictions
    if 'lightgbm' in models:
        lgb_model = models['lightgbm']
        X_flat = X.reshape(X.shape[0], -1)
        predictions['lightgbm'] = lgb_model.predict(X_flat)

    return predictions

def create_meta_features(base_predictions):
    """Create meta-features for stacking from base model predictions."""
    meta_features = []

    # Individual model predictions
    if 'lstm' in base_predictions:
        meta_features.append(base_predictions['lstm'])
    if 'cv_lstm_avg' in base_predictions:
        meta_features.append(base_predictions['cv_lstm_avg'])
    if 'lightgbm' in base_predictions:
        meta_features.append(base_predictions['lightgbm'])

    # CV individual fold predictions (if available)
    if 'cv_lstm_individual' in base_predictions:
        cv_individual = base_predictions['cv_lstm_individual']
        if len(cv_individual.shape) > 1:
            for fold_idx in range(cv_individual.shape[1]):
                meta_features.append(cv_individual[:, fold_idx])

    # Prediction statistics
    all_preds = np.array(meta_features)
    if len(all_preds.shape) > 1 and all_preds.shape[0] > 0:
        meta_features.extend([
            np.mean(all_preds, axis=0),  # Mean prediction
            np.std(all_preds, axis=0),   # Prediction std
            np.max(all_preds, axis=0),   # Max prediction
            np.min(all_preds, axis=0),   # Min prediction
        ])

    return np.column_stack(meta_features)

def train_stacking_model(base_predictions, labels, meta_model_type='logistic'):
    """Train a stacking meta-model."""
    meta_features = create_meta_features(base_predictions)

    # Split for meta-model training
    X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
        meta_features, labels, test_size=0.2, random_state=42
    )

    if meta_model_type == 'logistic':
        meta_model = LogisticRegression(random_state=42, max_iter=1000)
        meta_model.fit(X_meta_train, y_meta_train)

    elif meta_model_type == 'random_forest':
        meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
        meta_model.fit(X_meta_train, y_meta_train)

    elif meta_model_type == 'neural_net':
        # Scale features for neural network
        scaler = StandardScaler()
        X_meta_train_scaled = scaler.fit_transform(X_meta_train)

        # Convert to binary labels
        y_binary = (y_meta_train > 0).astype(int)

        # Train neural network
        input_dim = X_meta_train_scaled.shape[1]
        model = StackingMetaModel(input_dim)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        X_tensor = torch.from_numpy(X_meta_train_scaled).float()
        y_tensor = torch.from_numpy(y_binary).float().unsqueeze(1)

        # Simple training loop
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        meta_model = {'model': model, 'scaler': scaler}

    return meta_model

def predict_stacking(meta_model, base_predictions, meta_model_type='logistic'):
    """Make predictions with stacking meta-model."""
    meta_features = create_meta_features(base_predictions)

    if meta_model_type == 'neural_net':
        scaler = meta_model['scaler']
        model = meta_model['model']

        meta_features_scaled = scaler.transform(meta_features)
        X_tensor = torch.from_numpy(meta_features_scaled).float()

        with torch.no_grad():
            logits = model(X_tensor)
            preds = torch.sigmoid(logits).numpy().flatten()

    else:
        preds = meta_model.predict_proba(meta_features)[:, 1]

    return preds

def advanced_blending(base_predictions, labels, method='optimal_weights'):
    """Advanced blending methods beyond simple averaging."""
    pred_arrays = []
    model_names = []

    for name, preds in base_predictions.items():
        if name != 'cv_lstm_individual':  # Skip individual fold predictions
            pred_arrays.append(preds)
            model_names.append(name)

    if not pred_arrays:
        return None

    pred_matrix = np.column_stack(pred_arrays)

    if method == 'optimal_weights':
        # Find optimal weights by maximizing correlation with labels
        from scipy.optimize import minimize

        def objective(weights):
            blended = np.average(pred_matrix, axis=1, weights=weights)
            # Use rank correlation to handle non-linear relationships
            from scipy.stats import spearmanr
            corr, _ = spearmanr(blended, labels)
            return -corr  # Minimize negative correlation

        n_models = len(pred_arrays)
        initial_weights = np.ones(n_models) / n_models
        bounds = [(0.01, 1.0)] * n_models  # Weights between 0.01 and 1.0

        result = minimize(objective, initial_weights, bounds=bounds,
                         constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        optimal_weights = result.x
        blended_pred = np.average(pred_matrix, axis=1, weights=optimal_weights)

        return blended_pred, optimal_weights, model_names

    elif method == 'rank_averaging':
        # Average ranks instead of raw predictions
        ranks = []
        for preds in pred_arrays:
            ranks.append(np.argsort(np.argsort(preds)))  # Double argsort for ranks

        avg_ranks = np.mean(ranks, axis=0)
        # Convert back to prediction scale (0-1)
        blended_pred = avg_ranks / len(avg_ranks)

        return blended_pred, None, model_names

    elif method == 'confidence_weighted':
        # Weight by confidence (distance from 0.5)
        confidences = []
        for preds in pred_arrays:
            conf = np.abs(preds - 0.5) * 2  # Scale to 0-1
            confidences.append(conf)

        confidence_matrix = np.column_stack(confidences)
        # Normalize confidences to sum to 1 for each sample
        confidence_weights = confidence_matrix / confidence_matrix.sum(axis=1, keepdims=True)

        blended_pred = np.sum(pred_matrix * confidence_weights, axis=1)

        return blended_pred, None, model_names

def calculate_trading_metrics(preds, labels, threshold=0.5, commission_cost=0.42, avg_profit_per_win=10.0):
    """Calculate trading performance metrics."""
    preds_binary = (preds >= threshold).astype(int)
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
        'net_pnl': net_pnl,
        'pnl_per_trade': pnl_per_trade,
        'edge': edge,
        'total_commissions': total_commissions
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlflow_experiment', default='advanced_ensembles', help='MLflow experiment name')
    parser.add_argument('--commission_cost', type=float, default=0.42, help='Commission cost per trade')
    args = parser.parse_args()

    try:
        import mlflow_utils
        mlflow_utils.ensure_sqlite_tracking()
    except Exception:
        pass
    mlflow.set_experiment(args.mlflow_experiment)

    # Load data and base models
    print("Loading data and base models...")
    data = np.load(DATASET)
    X = data['X']
    y = data['y']

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Load base models
    base_models, input_dim = load_base_models()
    print(f"Loaded base models: {list(base_models.keys())}")

    # Get base model predictions on training data for stacking
    print("Getting base model predictions for stacking training...")
    train_predictions = get_base_predictions(base_models, X_train)

    # Get base model predictions on test data
    print("Getting base model predictions for testing...")
    test_predictions = get_base_predictions(base_models, X_test)

    results = []

    # Test individual models (baseline)
    print("\n" + "="*60)
    print("BASELINE: INDIVIDUAL MODELS")
    print("="*60)

    for model_name, preds in test_predictions.items():
        if model_name == 'cv_lstm_individual':
            continue  # Skip individual fold predictions

        metrics = calculate_trading_metrics(preds, y_test, threshold=0.3)  # Use optimized threshold
        metrics['method'] = 'individual'
        metrics['model'] = model_name
        results.append(metrics)

        print(f"{model_name:15} | P&L: ${metrics['net_pnl']:8.2f} | Win: {metrics['win_rate']:5.1%} | "
              f"Trades: {metrics['trades_taken']:5d} | Edge: {metrics['edge']:5.1%}")

    # Test stacking methods
    print("\n" + "="*60)
    print("STACKING METHODS")
    print("="*60)

    stacking_configs = [
        ('logistic_regression', 'logistic'),
        ('random_forest', 'random_forest'),
        ('neural_network', 'neural_net'),
    ]

    for config_name, meta_type in stacking_configs:
        print(f"\nTraining {config_name} stacking model...")

        try:
            # Train stacking model
            stacking_model = train_stacking_model(train_predictions, y_train, meta_type)

            # Make predictions
            stacking_preds = predict_stacking(stacking_model, test_predictions, meta_type)

            # Evaluate
            metrics = calculate_trading_metrics(stacking_preds, y_test, threshold=0.3)
            metrics['method'] = 'stacking'
            metrics['model'] = config_name
            results.append(metrics)

            print(f"{config_name:20} | P&L: ${metrics['net_pnl']:8.2f} | Win: {metrics['win_rate']:5.1%} | "
                  f"Trades: {metrics['trades_taken']:5d} | Edge: {metrics['edge']:5.1%}")

            # Log to MLflow
            with mlflow.start_run(run_name=f"stacking_{config_name}"):
                mlflow.log_param("method", "stacking")
                mlflow.log_param("meta_model", config_name)
                mlflow.log_param("meta_type", meta_type)
                mlflow.log_param("threshold", 0.3)
                mlflow.log_param("commission_cost", args.commission_cost)

                try:
                    import mlflow_utils
                    mlflow_utils.log_trading_metrics(metrics)
                except Exception:
        try:
            import mlflow_utils
            mlflow_utils.log_trading_metrics(metrics)
        except Exception:
            mlflow.log_metric("net_pnl", metrics['net_pnl'])
            mlflow.log_metric("win_rate", metrics['win_rate'])
            mlflow.log_metric("trades_taken", metrics['trades_taken'])
            mlflow.log_metric("edge", metrics['edge'])

        except Exception as e:
            print(f"Error with {config_name}: {e}")

    # Test advanced blending methods
    print("\n" + "="*60)
    print("ADVANCED BLENDING METHODS")
    print("="*60)

    blending_configs = [
        ('optimal_weights', 'optimal_weights'),
        ('rank_averaging', 'rank_averaging'),
        ('confidence_weighted', 'confidence_weighted'),
    ]

    for config_name, blend_method in blending_configs:
        print(f"\nTesting {config_name} blending...")

        try:
            blend_result = advanced_blending(test_predictions, y_test, method=blend_method)

            if blend_result is None:
                continue

            if len(blend_result) == 3:
                blend_preds, weights, model_names = blend_result
            else:
                blend_preds, weights = blend_result
                model_names = list(test_predictions.keys())

            # Evaluate
            metrics = calculate_trading_metrics(blend_preds, y_test, threshold=0.3)
            metrics['method'] = 'blending'
            metrics['model'] = config_name
            results.append(metrics)

            print(f"{config_name:20} | P&L: ${metrics['net_pnl']:8.2f} | Win: {metrics['win_rate']:5.1%} | "
                  f"Trades: {metrics['trades_taken']:5d} | Edge: {metrics['edge']:5.1%}")

            if weights is not None:
                print(f"  Weights: {dict(zip(model_names, weights))}")

            # Log to MLflow
            with mlflow.start_run(run_name=f"blending_{config_name}"):
                mlflow.log_param("method", "blending")
                mlflow.log_param("blend_type", config_name)
                mlflow.log_param("blend_method", blend_method)
                mlflow.log_param("threshold", 0.3)
                mlflow.log_param("commission_cost", args.commission_cost)

                if weights is not None:
                    for i, weight in enumerate(weights):
                        mlflow.log_param(f"weight_{model_names[i]}", weight)

                try:
                    import mlflow_utils
                    mlflow_utils.log_trading_metrics(metrics)
                except Exception:
                    mlflow.log_metric("net_pnl", metrics['net_pnl'])
                    mlflow.log_metric("win_rate", metrics['win_rate'])
                    mlflow.log_metric("trades_taken", metrics['trades_taken'])
                    mlflow.log_metric("edge", metrics['edge'])

        except Exception as e:
            print(f"Error with {config_name}: {e}")

    # Sort and display results
    results.sort(key=lambda x: x['net_pnl'], reverse=True)

    print("\n" + "="*60)
    print("OVERALL BEST METHODS")
    print("="*60)

    for i, result in enumerate(results[:10]):
        print(f"{i+1:2d}. {result['method'][:10]:10} {result['model'][:15]:15} | "
              f"P&L: ${result['net_pnl']:8.2f} | Win: {result['win_rate']:5.1%} | "
              f"Trades: {result['trades_taken']:5d} | Edge: {result['edge']:5.1%}")

    # Save detailed results
    df_results = pd.DataFrame(results)
    df_results.to_csv('ml/advanced_ensembles_results.csv', index=False)
    print("\nDetailed results saved to ml/advanced_ensembles_results.csv")

    print("\n✅ Advanced ensemble testing complete!")
    best = results[0]
    print(f"Best method: {best['method']} {best['model']}")
    print(f"Best P&L: ${best['net_pnl']:.2f}")
    print(f"Best win rate: {best['win_rate']:.1%}")
    print(f"Improvement over worst: ${best['net_pnl'] - results[-1]['net_pnl']:.2f}")

if __name__ == '__main__':
    main()
