#!/usr/bin/env python3
"""Production-ready LSTM training for trading with class balancing and transaction costs.

Key improvements:
- Time-based cross-validation (not random)
- Trading-aware loss function with commission penalties
- Class balancing techniques
- Focus on precision for profitable trading
"""
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
    """Loss function that balances trading costs with profit potential."""
    def __init__(self, pos_weight, commission_cost=0.42, expected_profit=10.0, risk_multiplier=2.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.commission_cost = commission_cost  # Actual commission per trade
        self.expected_profit = expected_profit  # Expected profit per winning trade
        self.risk_multiplier = risk_multiplier  # How much to penalize losses vs reward gains

    def forward(self, logits, targets):
        # Standard BCE loss with class weighting
        bce_loss = self.bce(logits, targets)

        # Trading cost analysis
        preds = torch.sigmoid(logits)

        # False positives: cost commission but no profit
        false_positives = (preds > 0.5) & (targets == 0)
        fp_cost = self.commission_cost * false_positives.float().sum()

        # True positives: get profit minus commission
        true_positives = (preds > 0.5) & (targets == 1)
        tp_profit = (self.expected_profit - self.commission_cost) * true_positives.float().sum()

        # False negatives: miss profit opportunity
        false_negatives = (preds <= 0.5) & (targets == 1)
        fn_cost = 0.5 * self.expected_profit * false_negatives.float().sum()  # Partial penalty

        # Net trading cost (normalized by batch size)
        batch_size = logits.size(0)
        net_trading_cost = (fp_cost - tp_profit + fn_cost) / batch_size

        return bce_loss + 0.1 * net_trading_cost  # Scale down the trading cost component
def oversample_minority_class(X, y, target_ratio=0.3):
    """Oversample minority class to improve balance."""
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    target_pos = int(target_ratio * (n_pos + n_neg))

    if target_pos > n_pos:
        # Duplicate positive samples
        additional_pos = np.random.choice(pos_indices, size=target_pos - n_pos, replace=True)
        pos_indices = np.concatenate([pos_indices, additional_pos])

    all_indices = np.concatenate([pos_indices, neg_indices])
    np.random.shuffle(all_indices)

    return X[all_indices], y[all_indices]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out', default=str(OUT / 'lstm_trading.pt'))
    p.add_argument('--epochs', default=20, type=int)
    p.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    p.add_argument('--batch', default=128, type=int)
    p.add_argument('--model_type', default='lstm', choices=['lstm', 'gru'])
    p.add_argument('--cv', type=int, default=5, help='Number of time-based CV folds')
    p.add_argument('--oversample', action='store_true', help='Oversample minority class')
    p.add_argument('--commission_penalty', type=float, default=0.42, help='Commission cost per trade (round trip)')
    p.add_argument('--mlflow_experiment', default='trading_model_training', help='MLflow experiment name')
    p.add_argument('--target_precision', type=float, default=0.6, help='Target precision for early stopping')
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

    # Time-based CV (walk-forward validation)
    n_samples = len(X)
    fold_size = n_samples // args.cv

    fold_results = []

    with mlflow.start_run(run_name=f"trading_lstm_{args.model_type}"):
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("cv_folds", args.cv)
        mlflow.log_param("oversample", args.oversample)
        mlflow.log_param("commission_penalty", args.commission_penalty)
        mlflow.log_param("dropout", args.dropout)
        mlflow.log_param("batch_size", args.batch)

        for fold in range(args.cv):
            print(f"\n=== Fold {fold+1}/{args.cv} ===")

            # Time-based split: train on past, validate on future
            train_end = max(fold * fold_size, fold_size)  # Ensure minimum training data
            val_start = train_end
            val_end = min((fold + 1) * fold_size, n_samples)

            if val_start >= val_end:
                print(f"Skipping fold {fold+1} - insufficient data")
                continue

            X_train = X[:train_end]
            y_train = y[:train_end]
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]

            print(f"Train: {len(X_train)} samples ({np.mean(y_train):.1%} positive)")
            print(f"Val: {len(X_val)} samples ({np.mean(y_val):.1%} positive)")

            # Oversample minority class if requested
            if args.oversample:
                X_train, y_train = oversample_minority_class(X_train, y_train, target_ratio=0.25)
                print(f"After oversampling: {len(X_train)} samples ({np.mean(y_train):.1%} positive)")

            # Compute class weights
            pos_weight = torch.tensor(len(y_train) / y_train.sum(), dtype=torch.float)

            train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
            val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())

            train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, pin_memory=torch.cuda.is_available())
            val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, pin_memory=torch.cuda.is_available())

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True

            pos_weight = pos_weight.to(device)
            loss_fn = TradingLoss(pos_weight, commission_cost=args.commission_penalty)
            model = LSTMModel(X.shape[2], dropout=args.dropout, model_type=args.model_type).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            best_precision = 0
            patience = 0
            max_patience = 7

            for epoch in range(args.epochs):
                # Training
                model.train()
                train_loss = 0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    logits = model(xb)
                    loss = loss_fn(logits.view(-1), yb)
                    loss.backward()
                    opt.step()
                    train_loss += loss.item()

                # Validation
                model.eval()
                all_preds = []
                all_labels = []
                val_loss = 0

                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)
                        preds = torch.sigmoid(logits.view(-1)).cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(yb.cpu().numpy())
                        val_loss += loss_fn(logits.view(-1), yb).item() * xb.size(0)

                val_loss /= len(val_loader.dataset)
                preds_binary = (np.array(all_preds) > 0.5).astype(int)
                labels_binary = np.array(all_labels)

                acc = accuracy_score(labels_binary, preds_binary)
                precision = precision_score(labels_binary, preds_binary, zero_division=0)
                recall = recall_score(labels_binary, preds_binary, zero_division=0)
                f1 = f1_score(labels_binary, preds_binary, zero_division=0)

                print(f"Epoch {epoch+1:2d}: acc={acc:.3f} prec={precision:.3f} rec={recall:.3f} f1={f1:.3f} loss={val_loss:.4f}")

                # Save best model based on precision (key for trading profitability)
                if precision > best_precision:
                    best_precision = precision
                    patience = 0
                            fold_out = str(resolve_cli_path(args.out)).replace('.pt', f'_fold{fold}.pt')
                            Path(fold_out).parent.mkdir(parents=True, exist_ok=True)
                            torch.save(model.state_dict(), fold_out)
                    print(f"  → Saved model with precision {precision:.3f}")
                else:
                    patience += 1

                # Early stopping if precision target reached or patience exceeded
                if precision >= args.target_precision or patience >= max_patience:
                    print(f"  → Stopping: precision={precision:.3f}, patience={patience}")
                    break

            fold_results.append({
                'fold': fold,
                'precision': best_precision,
                'accuracy': acc,
                'recall': recall,
                'f1': f1
            })

        # Summary
        precisions = [r['precision'] for r in fold_results]
        avg_precision = np.mean(precisions)
        std_precision = np.std(precisions)

        print("\n=== TRAINING SUMMARY ===")
        print(f"CV Average Precision: {avg_precision:.3f} ± {std_precision:.3f}")
        print(f"Best Fold Precision: {max(precisions):.3f}")
        print(f"Folds completed: {len(fold_results)}")

        # Log to MLflow
        mlflow.log_metric("cv_avg_precision", avg_precision)
        mlflow.log_metric("cv_std_precision", std_precision)
        mlflow.log_metric("best_precision", max(precisions))
        mlflow.log_param("final_model_path", str(resolve_cli_path(args.out)))

        # Save final model (last fold)
        if fold_results:
            final_model_path = str(resolve_cli_path(args.out)).replace('.pt', f'_fold{len(fold_results)-1}.pt')
            import shutil
            shutil.copy(final_model_path, str(resolve_cli_path(args.out)))
            print(f"Saved final model to {str(resolve_cli_path(args.out))}")

            # Log model to MLflow
            model = LSTMModel(X.shape[2], dropout=args.dropout, model_type=args.model_type)
            model.load_state_dict(torch.load(str(resolve_cli_path(args.out))))
            mlflow.pytorch.log_model(model, "model")

        print("\n🎯 Production-ready model training complete!")
        print(f"Target precision: {args.target_precision}")
        print(f"Achieved: {avg_precision:.3f}")
        if avg_precision >= args.target_precision:
            print("✅ SUCCESS: Model meets trading precision requirements!")
        else:
            print("⚠️  WARNING: Model below target precision - may need more data/tuning")