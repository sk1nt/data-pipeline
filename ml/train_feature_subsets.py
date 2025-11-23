#!/usr/bin/env python3
"""Train LSTM models on selected feature subsets for comparison."""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
import mlflow
import mlflow.pytorch
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Custom loss function with commission costs
class TradingLoss(nn.Module):
    def __init__(self, commission_cost=0.42, profit_target=10.0):
        super().__init__()
        self.commission_cost = commission_cost
        self.profit_target = profit_target

    def forward(self, predictions, targets):
        # Convert predictions to trade decisions (threshold at 0.5 for training)
        trade_decisions = (torch.sigmoid(predictions) > 0.5).float()

        # Calculate P&L for each trade
        correct_trades = (trade_decisions == targets.squeeze()).float()
        pnl = correct_trades * self.profit_target - trade_decisions * self.commission_cost

        # Return negative mean P&L (to maximize P&L)
        return -pnl.mean()

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def load_data_and_features():
    """Load dataset and feature names."""
    data = np.load('output/MNQ_multi_day_enhanced.npz')
    X = data['X']
    y = data['y']

    with open('output/MNQ_multi_day_enhanced_features.txt', 'r') as f:
        all_feature_names = [line.strip() for line in f.readlines()]

    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[2]} features")
    return X, y, all_feature_names

def get_feature_subset(feature_names, subset_name):
    """Get feature indices for a specific subset."""
    subsets = {
        'consensus_2plus': [
            'nq_spot', 'obv', 'sma_5', 'sma_10', 'rsi', 'close_mean_5', 'close_std_5',
            'volume_mean_5', 'returns_std_5', 'close_mean_10', 'close_std_10', 'volume_mean_10',
            'returns_std_10', 'close_mean_20', 'close_std_20', 'returns_std_20', 'spread_pct',
            'vwap', 'vwap_distance', 'cross_momentum', 'volume', 'adx', 'di_plus', 'di_minus',
            'stoch_k', 'stoch_d', 'williams_r', 'atr', 'adosc', 'macd', 'macd_hist', 'spread',
            'minute', 'is_market_open', 'hour_sin', 'hour_cos', 'volume_mean_20'
        ],
        'consensus_3plus': [
            'rsi', 'close_std_5', 'returns_std_5', 'close_std_10', 'volume_mean_10',
            'returns_std_10', 'close_std_20', 'returns_std_20', 'vwap', 'volume', 'adx',
            'di_plus', 'di_minus', 'adosc'
        ],
        'mutual_info': [
            'nq_spot', 'obv', 'sma_5', 'sma_10', 'rsi', 'close_mean_5', 'close_std_5',
            'volume_mean_5', 'returns_std_5', 'close_mean_10', 'close_std_10', 'volume_mean_10',
            'returns_std_10', 'close_mean_20', 'close_std_20', 'returns_std_20', 'spread_pct',
            'vwap', 'vwap_distance', 'cross_momentum'
        ],
        'original_13': [
            'open', 'high', 'low', 'close', 'volume', 'gex_zero', 'nq_spot',
            'adx', 'di_plus', 'di_minus', 'rsi', 'stoch_k', 'stoch_d'
        ]
    }

    if subset_name not in subsets:
        raise ValueError(f"Unknown subset: {subset_name}")

    selected_features = subsets[subset_name]
    feature_indices = [feature_names.index(feat) for feat in selected_features if feat in feature_names]

    print(f"Selected {len(feature_indices)}/{len(selected_features)} features for {subset_name}")
    missing = [feat for feat in selected_features if feat not in feature_names]
    if missing:
        print(f"Missing features: {missing}")

    return feature_indices, selected_features

def train_model(X, y, feature_indices, subset_name, epochs=20, batch_size=64, commission_cost=0.42):
    """Train LSTM model on selected features."""
    print(f"\nTraining {subset_name} model ({len(feature_indices)} features)...")

    # Select features
    X_subset = X[:, :, feature_indices]

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X_subset, y, test_size=0.3, random_state=42, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

    # Scale features
    scaler = StandardScaler()
    # Fit on training data
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_reshaped)

    # Transform all data
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
    X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

    # Create data loaders
    train_dataset = TensorDataset(torch.from_numpy(X_train_scaled).float(),
                                torch.from_numpy(y_train).float().unsqueeze(1))
    val_dataset = TensorDataset(torch.from_numpy(X_val_scaled).float(),
                              torch.from_numpy(y_val).float().unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    input_dim = len(feature_indices)
    model = LSTMModel(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.2)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_val_pnl = -float('inf')
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                val_targets.extend(y_batch.cpu().numpy().flatten())

        val_loss /= len(val_loader)

        # Calculate validation metrics and P&L
        val_preds_binary = np.array(val_preds) > 0.5
        val_targets_binary = np.array(val_targets) > 0

        # Calculate P&L at optimal threshold (0.3)
        threshold = 0.3
        val_preds_thresh = np.array(val_preds) >= threshold
        trades_taken = np.sum(val_preds_thresh)
        correct_trades = np.sum((val_preds_thresh == 1) & (val_targets_binary == 1))
        win_rate = correct_trades / trades_taken if trades_taken > 0 else 0

        total_commissions = trades_taken * commission_cost
        gross_pnl = correct_trades * 10.0  # Assume $10 profit per win
        val_pnl = gross_pnl - total_commissions

        scheduler.step(val_loss)

        if val_pnl > best_val_pnl:
            best_val_pnl = val_pnl
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val P&L: ${val_pnl:.2f} | Win: {win_rate:.1%}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation on test set
    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        test_dataset = TensorDataset(torch.from_numpy(X_test_scaled).float(),
                                   torch.from_numpy(y_test).float().unsqueeze(1))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            test_preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            test_targets.extend(y_batch.cpu().numpy().flatten())

    # Calculate final metrics at threshold 0.3
    test_preds_thresh = np.array(test_preds) >= threshold
    test_targets_binary = (np.array(test_targets) > 0).astype(int)

    trades_taken = np.sum(test_preds_thresh)
    correct_trades = np.sum((test_preds_thresh == 1) & (test_targets_binary == 1))
    win_rate = correct_trades / trades_taken if trades_taken > 0 else 0

    total_commissions = trades_taken * commission_cost
    gross_pnl = correct_trades * 10.0
    net_pnl = gross_pnl - total_commissions

    edge = win_rate - (commission_cost / 10.0)

    print("\nTest Results:")
    print(".4f")
    print(".1%")
    print(".2f")
    print(f"Trades: {trades_taken}, Edge: {edge:.1%}")

    return model, scaler, {
        'subset': subset_name,
        'num_features': len(feature_indices),
        'win_rate': win_rate,
        'net_pnl': net_pnl,
        'trades_taken': trades_taken,
        'edge': edge,
        'commission_cost': commission_cost,
        'threshold': threshold
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subsets', nargs='+', default=['consensus_2plus', 'consensus_3plus', 'mutual_info', 'original_13'],
                       help='Feature subsets to train')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--commission_cost', type=float, default=0.42, help='Commission cost per trade')
    parser.add_argument('--mlflow_experiment', default='feature_subset_training', help='MLflow experiment name')
    args = parser.parse_args()

    try:
        import mlflow_utils
        mlflow_utils.ensure_sqlite_tracking()
    except Exception:
        pass
    mlflow.set_experiment(args.mlflow_experiment)

    # Load data
    X, y, feature_names = load_data_and_features()

    results = []

    for subset_name in args.subsets:
        try:
            # Get feature subset
            feature_indices, selected_features = get_feature_subset(feature_names, subset_name)

            # Train model
            model, scaler, metrics = train_model(
                X, y, feature_indices, subset_name,
                epochs=args.epochs, commission_cost=args.commission_cost
            )

            results.append(metrics)

            # Save model
            model_path = f'models/{subset_name}_model.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'feature_indices': feature_indices,
                'selected_features': selected_features,
                'scaler': scaler,
                'metrics': metrics
            }, model_path)

            # Log to MLflow
            with mlflow.start_run(run_name=f"{subset_name}_training"):
                mlflow.log_param("subset_name", subset_name)
                mlflow.log_param("num_features", len(feature_indices))
                mlflow.log_param("features", str(selected_features))
                mlflow.log_param("epochs", args.epochs)
                mlflow.log_param("commission_cost", args.commission_cost)

                mlflow.log_metric("win_rate", metrics['win_rate'])
                mlflow.log_metric("net_pnl", metrics['net_pnl'])
                mlflow.log_metric("trades_taken", metrics['trades_taken'])
                mlflow.log_metric("edge", metrics['edge'])

                mlflow.pytorch.log_model(model, "model")

        except Exception as e:
            print(f"Error training {subset_name}: {e}")
            continue

    # Compare results
    print("\n" + "="*60)
    print("FEATURE SUBSET TRAINING RESULTS")
    print("="*60)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('net_pnl', ascending=False)

    print("\nModel Performance Comparison:")
    for i, row in df_results.iterrows():
        print(f"{i+1:2d}. {row['subset'][:18]:18} | Features: {int(row['num_features']):2d} | "
              f"P&L: ${row['net_pnl']:8.2f} | Win: {row['win_rate']:5.1%} | "
              f"Trades: {int(row['trades_taken']):4d} | Edge: {row['edge']:5.1%}")

    # Compare with baseline
    print("\nBaseline (enhanced 60 features):")
    print("      P&L: $6430.16, Win: 18.6%, Trades: 4452, Edge: 14.4%")

    # Save results
    df_results.to_csv('feature_subset_training_results.csv', index=False)

    print("\nDetailed results saved to feature_subset_training_results.csv")
    print("Models saved to models/ directory")

    print("\n✅ Feature subset training complete!")
    best = df_results.iloc[0]
    print(f"Best subset: {best['subset']} ({best['num_features']} features)")
    print(".2f")
    print(".1%")
    print(f"Improvement over baseline: ${best['net_pnl'] - 6430.16:.2f}")

if __name__ == '__main__':
    main()