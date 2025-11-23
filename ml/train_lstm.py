#!/usr/bin/env python3
"""Train a simple PyTorch LSTM baseline on windowed datasets.

Usage:
  python train_lstm.py --input output/MNQ_20251111_1s_w60s_h1.npz
"""
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from pathlib import Path

OUT = Path(__file__).resolve().parents[0] / 'models'
OUT.mkdir(parents=True, exist_ok=True)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.0, model_type='lstm'):
        super().__init__()
        if model_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif model_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError('model_type must be lstm or gru')
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.rnn(x)
        # take last timestep
        out = out[:, -1, :]
        out = self.fc(out)
        return out

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out', default=str(OUT / 'lstm_model.pt'))
    p.add_argument('--epochs', default=10, type=int)
    p.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for LSTM layers')
    p.add_argument('--patience', type=int, default=0, help='Early stopping patience (0 disables)')
    p.add_argument('--batch', default=64, type=int)
    p.add_argument('--model_type', default='lstm', choices=['lstm', 'gru'], help='RNN type')
    p.add_argument('--cv', type=int, default=0, help='Number of CV folds (0 disables)')
    p.add_argument('--mlflow', action='store_true', help='Log to MLflow server if available')
    p.add_argument('--backtest-days', default=None, type=str, help='Comma-separated days to run pnl_backtest after training')
    args = p.parse_args()
    mlflow_module = None
    if args.mlflow:
        try:
            import mlflow
            import mlflow.pytorch
            try:
                import mlflow_utils
                mlflow_utils.ensure_sqlite_tracking()
            except Exception:
                pass
            mlflow_module = mlflow
            mlflow_module.start_run()
            mlflow_module.log_params({
                'epochs': args.epochs,
                'batch': args.batch,
                'model_type': args.model_type,
                'dropout': args.dropout,
                'cv': args.cv,
            })
        except Exception:
            print('MLflow not available; skipping mlflow logging')
            mlflow_module = None

    try:
        from ml.path_utils import resolve_cli_path
    except Exception:
        from path_utils import resolve_cli_path
    input_path = resolve_cli_path(args.input)
    data = np.load(input_path)
    X = data['X']
    y = data['y']
    y = (y > 0).astype(np.float32)

    if args.cv > 0:
        # K-fold CV
        kf = KFold(n_splits=args.cv, shuffle=True, random_state=42)
        fold_models = []
        fold_accs = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f'Fold {fold+1}/{args.cv}')
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Compute pos_weight
            pos_weight = torch.tensor(len(y_train) / y_train.sum(), dtype=torch.float)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
            val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())

            pin_memory = torch.cuda.is_available()
            train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, pin_memory=pin_memory)
            val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, pin_memory=pin_memory)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
            pos_weight = pos_weight.to(device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            model = LSTMModel(X.shape[2], dropout=args.dropout, model_type=args.model_type).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            best_val_loss = float('inf')
            for e in range(args.epochs):
                model.train()
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    opt.zero_grad()
                    logits = model(xb)
                    loss = loss_fn(logits.view(-1), yb)
                    loss.backward()
                    opt.step()
                model.eval()
                all_preds = []
                all_labels = []
                valloss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)
                        preds = torch.sigmoid(logits.squeeze()).cpu().numpy()
                        all_preds.extend(preds.tolist())
                        all_labels.extend(yb.cpu().numpy().tolist())
                        valloss += loss_fn(logits.view(-1), yb).item() * xb.size(0)
                val_loss = valloss / len(val_loader.dataset)
                acc = accuracy_score((np.array(all_preds) > 0.5).astype(int), np.array(all_labels))
                print(f'  Epoch {e+1}/{args.epochs} val_acc={acc:.4f} val_loss={val_loss:.4f}')
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    fold_models.append(model.state_dict())
                    fold_accs.append(acc)
            # Save fold model
            fold_out = args.out.replace('.pt', f'_fold{fold}.pt')
            torch.save(model.state_dict(), fold_out)
        print(f'Saved fold {fold} model to {fold_out}')
    if args.cv > 0 and mlflow_module:
        try:
            mlflow_module.log_metric('cv_avg_acc', float(np.mean(fold_accs)) if fold_accs else 0.0)
        except Exception:
            pass
        try:
            mlflow.pytorch.log_model(model, artifact_path='model')
        except Exception:
            mlflow_module.log_artifact(args.out)
        mlflow_module.end_run()

        # Average accuracies
        avg_acc = np.mean(fold_accs)
        print(f'CV average accuracy: {avg_acc:.4f}')
        # For ensemble, we can load all fold models later
    else:
        # Single train/val split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Compute pos_weight for class imbalance
        pos_weight = torch.tensor(len(y_train) / y_train.sum(), dtype=torch.float)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())

        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, pin_memory=pin_memory)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        pos_weight = pos_weight.to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Recreate with pos_weight on device
        model = LSTMModel(X.shape[2], dropout=args.dropout, model_type=args.model_type).to(device)
        best_val = float('inf')
        best_epoch = 0
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        for e in range(args.epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits.view(-1), yb)
                loss.backward()
                opt.step()
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    preds = torch.sigmoid(logits.squeeze()).cpu().numpy()
                    all_preds.extend(preds.tolist())
                    all_labels.extend(yb.cpu().numpy().tolist())
            acc = accuracy_score((np.array(all_preds) > 0.5).astype(int), np.array(all_labels))
            print(f'Epoch {e+1}/{args.epochs} val_acc={acc:.4f}')
            # Use val loss for early stopping and saving best model
            val_loss = None
            # recalc val_loss
            # There's no return of loss in existing code; compute a proper val_loss by rolling through val_loader
            valloss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    valloss += loss_fn(logits.view(-1), yb).item() * xb.size(0)
            val_loss = valloss / len(val_loader.dataset)
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = e
                out_path = resolve_cli_path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out_path)
            # early stopping
            if args.patience > 0 and (e - best_epoch) >= args.patience:
                print('Early stopping at epoch', e)
                break
        # If patience disabled and not saved yet, save latest model
        if args.patience == 0:
            out_path = resolve_cli_path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)
        print(f'Saved {args.model_type.upper()} model to', resolve_cli_path(args.out))
        # Optionally run PnL backtest and log results
        if args.backtest_days:
            try:
                import subprocess, re
                days = args.backtest_days
                cmd = [sys.executable, '-m', 'ml.pnl_backtest', '--model', str(resolve_cli_path(args.out)), '--days', days, '--window', '60', '--instrument', 'MNQ', '--threshold', '0.5']
                res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
                out_text = res.stdout + '\n' + res.stderr
                print(out_text)
                total_pnl = None
                for line in out_text.splitlines():
                    if 'Total PnL:' in line:
                        m = re.search(r'Total PnL:\s*\$([\-\d.,]+)', line)
                        if m:
                            total_pnl = float(m.group(1).replace(',',''))
                if total_pnl is not None and args.mlflow:
                    try:
                        import mlflow
                        mlflow.log_metric('post_training_total_pnl', float(total_pnl))
                    except Exception:
                        pass
            except Exception as e:
                print('Backtest failed:', e)
    if args.cv == 0 and mlflow_module:
        try:
            mlflow_module.log_metric('val_acc', acc)
            mlflow_module.log_metric('val_loss', best_val)
        except Exception:
            pass
        try:
            mlflow.pytorch.log_model(model, artifact_path='model')
        except Exception as exc:
            print('MLflow model log skipped:', exc)
        finally:
            mlflow_module.end_run()
