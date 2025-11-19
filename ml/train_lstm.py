#!/usr/bin/env python3
"""Train a simple PyTorch LSTM baseline on windowed datasets.

Usage:
  python train_lstm.py --input output/MNQ_20251111_1s_w60s_h1.npz
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

OUT = Path('models')
OUT.mkdir(parents=True, exist_ok=True)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.0, model_type='lstm'):
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
    p.add_argument('--mlflow', action='store_true', help='Log to MLflow server if available')
    args = p.parse_args()

    data = np.load(args.input)
    X = data['X']
    y = data['y']
    y = (y > 0).astype(np.float32)

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, pin_memory=pin_memory)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    model = LSTMModel(X.shape[2], dropout=args.dropout, model_type=args.model_type).to(device)
    best_val = float('inf')
    best_epoch = 0
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    mlflow_module = None
    if args.mlflow:
        try:
            import mlflow
            import mlflow.pytorch
            mlflow_module = mlflow
            mlflow_module.start_run()
            mlflow_module.log_params({
                'epochs': args.epochs,
                'batch': args.batch,
                'lr': 1e-3,
                'dropout': args.dropout,
            })
        except Exception:
            print('MLflow not available in environment (skipping mlflow logging)')

    for e in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits.squeeze(), yb)
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
                valloss += loss_fn(logits.squeeze(), yb).item() * xb.size(0)
        val_loss = valloss / len(val_loader.dataset)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = e
            torch.save(model.state_dict(), args.out)
            if mlflow_module:
                try:
                    import mlflow.pytorch
                    mlflow.pytorch.log_model(model, artifact_path='model')
                except Exception:
                    # mlflow/pytorch integration may fail; keep saving file
                    pass
        # early stopping
        if args.patience > 0 and (e - best_epoch) >= args.patience:
            print('Early stopping at epoch', e)
            break
    # If patience disabled and not saved yet, save latest model
    if args.patience == 0:
        torch.save(model.state_dict(), args.out)
        if mlflow_module:
            try:
                import mlflow.pytorch
                mlflow.pytorch.log_model(model, artifact_path='model')
            except Exception:
                pass
    if mlflow_module:
        try:
            mlflow_module.log_metric('best_val_loss', float(best_val))
            mlflow_module.end_run()
        except Exception:
            pass
    print(f'Saved {args.model_type.upper()} model to', args.out)
