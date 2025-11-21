#!/usr/bin/env python3
"""Train a simple TCN (dilated Conv1D) using PyTorch.
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

class TCNBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_c, out_c, kernel_size, padding=pad, dilation=dilation)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_c)
    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))

class TCNModel(nn.Module):
    def __init__(self, in_channels, layers=[32,32,64], kernel_size=3):
        super().__init__()
        blocks = []
        in_c = in_channels
        for i, c in enumerate(layers):
            blocks.append(TCNBlock(in_c, c, kernel_size=kernel_size, dilation=2**i))
            in_c = c
        self.net = nn.Sequential(*blocks)
        self.fc = nn.Linear(in_c, 1)

    def forward(self, x):
        # x: B, T, F -> Conv expects B, F, T
        x = x.permute(0,2,1)
        out = self.net(x)
        out = out.mean(dim=2)
        return self.fc(out)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--out', default=str(OUT / 'tcn_model.pt'))
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
    model = TCNModel(X.shape[2]).to(device)
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

    out_path = resolve_cli_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print('Saved TCN model to', out_path)
    if mlflow_module:
        try:
            mlflow.pytorch.log_model(model, artifact_path='model')
        except Exception:
            pass
