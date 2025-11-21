#!/usr/bin/env python3
"""Simplified production-ready LSTM training for trading with real commission costs."""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import mlflow
import mlflow.pytorch

OUT = Path(__file__).resolve().parents[0] / 'models'
OUT.mkdir(parents=True, exist_ok=True)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.0, model_type='lstm'):
        super().__init__()
        if model_type == 'lstm':
            self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif model_type == 'gru':
            self.rnn = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError('model_type must be lstm or gru')
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

class TradingLoss(nn.Module):
    """Loss function that penalizes false positives (commission costs) more heavily."""
    def __init__(self, pos_weight, commission_cost=0.42):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.commission_cost = commission_cost

    def forward(self, logits, targets):
        # Standard BCE loss
        bce_loss = self.bce(logits, targets)

        # Penalty for false positives (commission cost)
        preds = torch.sigmoid(logits)
        false_positives = (preds > 0.5) & (targets == 0)
        fp_penalty = self.commission_cost * false_positives.float().mean()

        return bce_loss + fp_penalty

def oversample_minority_class(X, y, target_ratio=0.25):
    """Simple oversampling of minority class."""
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    target_pos = int(target_ratio * (n_pos + n_neg))

    if target_pos > n_pos:
        additional_pos = np.random.choice(pos_indices, size=target_pos - n_pos, replace=True)
        pos_indices = np.concatenate([pos_indices, additional_pos])

    all_indices = np.concatenate([pos_indices, neg_indices])
    np.random.shuffle(all_indices)

    return X[all_indices], y[all_indices]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out', default=str(OUT / 'lstm_production.pt'))
    p.add_argument('--epochs', default=15, type=int)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--batch', default=128, type=int)
    p.add_argument('--commission_cost', type=float, default=0.42, help='MNQ commission cost')
    p.add_argument('--oversample', action='store_true')
    p.add_argument('--mlflow_experiment', default='production_trading_model')
    args = p.parse_args()

    # Setup MLflow
    mlflow.set_experiment(args.mlflow_experiment)

    try:
        from ml.path_utils import resolve_cli_path
    except Exception:
        from path_utils import resolve_cli_path
    input_path = resolve_cli_path(args.input)
    data = np.load(input_path)
    X = data['X']
    y_raw = data['y']
    y = (y_raw > 0).astype(np.float32)

    print(f"Dataset: {len(X)} samples, {X.shape[1]} timesteps, {X.shape[2]} features")
    print(f"Class distribution: {np.mean(y):.1%} positive")

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Train: {len(X_train)} samples ({np.mean(y_train):.1%} positive)")
    print(f"Val: {len(X_val)} samples ({np.mean(y_val):.1%} positive)")
    print(f"Test: {len(X_test)} samples ({np.mean(y_test):.1%} positive)")

    # Oversample minority class
    if args.oversample:
        X_train, y_train = oversample_minority_class(X_train, y_train, target_ratio=0.25)
        print(f"After oversampling: {len(X_train)} samples ({np.mean(y_train):.1%} positive)")

    # Class weights
    pos_weight = torch.tensor(len(y_train) / y_train.sum(), dtype=torch.float)

    # Data loaders
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, pin_memory=torch.cuda.is_available())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_weight = pos_weight.to(device)
    loss_fn = TradingLoss(pos_weight, commission_cost=args.commission_cost)
    model = LSTMModel(X.shape[2], dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_precision = 0
    patience = 0
    max_patience = 8

    print("\n=== TRAINING ===")
    for epoch in range(args.epochs):
        # Train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits.view(-1), yb)
            loss.backward()
            opt.step()

        # Validate
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = torch.sigmoid(logits.view(-1)).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.cpu().numpy())

        preds_binary = (np.array(all_preds) > 0.5).astype(int)
        labels_binary = np.array(all_labels)

        acc = accuracy_score(labels_binary, preds_binary)
        precision = precision_score(labels_binary, preds_binary, zero_division=0)
        recall = recall_score(labels_binary, preds_binary, zero_division=0)
        f1 = f1_score(labels_binary, preds_binary, zero_division=0)

        print(f"Epoch {epoch+1:2d}: acc={acc:.3f} prec={precision:.3f} rec={recall:.3f} f1={f1:.3f}")

        # Save best model based on precision
        if precision > best_precision:
            best_precision = precision
            patience = 0
            out_path = resolve_cli_path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)
            print(f"  → Saved model with precision {precision:.3f}")
        else:
            patience += 1

        if patience >= max_patience:
            print(f"  → Early stopping at epoch {epoch+1}")
            break

    # Final evaluation
    print("\n=== FINAL EVALUATION ===")
    model.load_state_dict(torch.load(str(resolve_cli_path(args.out))))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = torch.sigmoid(logits.view(-1)).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.cpu().numpy())

    preds_binary = (np.array(all_preds) > 0.5).astype(int)
    labels_binary = np.array(all_labels)

    final_acc = accuracy_score(labels_binary, preds_binary)
    final_precision = precision_score(labels_binary, preds_binary, zero_division=0)
    final_recall = recall_score(labels_binary, preds_binary, zero_division=0)
    final_f1 = f1_score(labels_binary, preds_binary, zero_division=0)

    print(f"Test Results: acc={final_acc:.3f} prec={final_precision:.3f} rec={final_recall:.3f} f1={final_f1:.3f}")

    # Trading analysis with real commissions
    trades_taken = np.sum(preds_binary == 1)
    correct_trades = np.sum((preds_binary == 1) & (labels_binary == 1))
    win_rate = correct_trades / trades_taken if trades_taken > 0 else 0

    total_commissions = trades_taken * args.commission_cost
    avg_profit_per_win = 10.0  # $10 per winning trade
    gross_pnl = correct_trades * avg_profit_per_win
    net_pnl = gross_pnl - total_commissions

    print("\n=== TRADING ANALYSIS (MNQ - $0.42 round trip) ===")
    print(f"Trades Taken: {trades_taken}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Total Commissions: ${total_commissions:.2f}")
    print(f"Gross P&L: ${gross_pnl:.2f}")
    print(f"Net P&L: ${net_pnl:.2f}")

    if trades_taken > 0:
        pnl_per_trade = net_pnl / trades_taken
        print(f"P&L per Trade: ${pnl_per_trade:.2f}")

        # Break-even analysis
        break_even_win_rate = args.commission_cost / avg_profit_per_win
        print(f"Break-even Win Rate: {break_even_win_rate:.1%}")
        print(f"Model Win Rate: {win_rate:.1%}")
        print(f"Edge: {(win_rate - break_even_win_rate):.1%}")

    # Log to MLflow
    with mlflow.start_run(run_name=f"production_lstm_mnq"):
        mlflow.log_param("commission_cost", args.commission_cost)
        mlflow.log_param("oversample", args.oversample)
        mlflow.log_param("dropout", args.dropout)

        mlflow.log_metric("test_accuracy", final_acc)
        mlflow.log_metric("test_precision", final_precision)
        mlflow.log_metric("test_recall", final_recall)
        mlflow.log_metric("test_f1", final_f1)
        mlflow.log_metric("trades_taken", trades_taken)
        mlflow.log_metric("win_rate", win_rate)
        mlflow.log_metric("net_pnl", net_pnl)
        mlflow.log_metric("total_commissions", total_commissions)

        mlflow.pytorch.log_model(model, "model")

    print("\n✅ Production model training complete!")
    print(f"Model saved to {args.out}")
    print(f"Ready for live trading evaluation!")

    if final_precision >= 0.5 and win_rate >= 0.55:
        print("🎯 EXCELLENT: Model exceeds profitability requirements!")
    elif final_precision >= 0.4 and win_rate >= 0.5:
        print("✅ GOOD: Model meets basic profitability requirements")
    else:
        print("⚠️  NEEDS IMPROVEMENT: Consider more data or feature engineering")