#!/usr/bin/env python3
"""Train a simple 1D-CNN model for sequence classification using PyTorch.

Usage: python train_cnn.py --input output/MNQ_20251111_1s_w60s_h1.npz --epochs 1 --batch 128
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

OUT = Path(__file__).resolve().parents[0] / 'models'
OUT.mkdir(parents=True, exist_ok=True)

class CNNModel(nn.Module):
    def __init__(self, in_channels, channels=[32,64], kernel_size=3):
        super().__init__()
        layers = []
        prev = in_channels
        for c in channels:
            layers.append(nn.Conv1d(prev, c, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            prev = c
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(prev, 1)

    def forward(self, x):
        # x: B, T, F -> B, F, T for conv1d
        x = x.permute(0,2,1)
        out = self.conv(x)
        out = out.mean(dim=2)  # global avg pool
        return self.fc(out)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--out', default=str(OUT / 'cnn_model.pt'))
    p.add_argument('--mlflow', action='store_true', help='Log metrics and model to MLflow if available')
    args = p.parse_args()

    try:
        from ml.path_utils import resolve_cli_path
    except Exception:
        from path_utils import resolve_cli_path
    input_path = resolve_cli_path(args.input)
    data = np.load(input_path)
    X = data['X']
    y = data['y']
    y = (y > 0).astype(np.float32)

    # train small subset to be fast
    sample = min(20000, X.shape[0])
    X = X[:sample]
    y = y[:sample]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, pin_memory=pin_memory)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNModel(X.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

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
            mlflow_module.log_params({'epochs': args.epochs, 'batch': args.batch})
        except Exception:
            print('MLflow not available; skipping logging')

    for e in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb).squeeze()
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds.extend(torch.sigmoid(logits.squeeze()).cpu().numpy().tolist())
                labels.extend(yb.cpu().numpy().tolist())
        acc = accuracy_score((np.array(preds) > 0.5).astype(int), np.array(labels))
        print(f'Epoch {e+1} val_acc={acc:.4f}')
        if mlflow_module:
            try:
                mlflow_module.log_metric('val_acc', float(acc), step=e+1)
            except Exception:
                pass

    # Normalize out path to repo-root-aware output path
    out_path = resolve_cli_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print('Saved CNN model to', out_path)
    if mlflow_module:
        try:
            mlflow.pytorch.log_model(model, artifact_path='model')
        except Exception:
            pass
