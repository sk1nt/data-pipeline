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
    p.add_argument('--device', default='auto', choices=['auto','cpu','cuda'], help='Device to train on')
    p.add_argument('--num-workers', default=0, type=int, help='DataLoader num_workers')
    p.add_argument('--sample', default=None, type=int, help='Limit dataset to first N samples to reduce memory')
    p.add_argument('--out', default=str(OUT / 'cnn_model.pt'))
    p.add_argument('--mlflow', action='store_true', help='Log metrics and model to MLflow if available')
    args = p.parse_args()

    try:
        from ml.path_utils import resolve_cli_path, repo_root
    except Exception:
        from path_utils import resolve_cli_path, repo_root
    input_path = resolve_cli_path(args.input)
    data = np.load(input_path)
    X = data['X']
    y = data['y']
    y = (y > 0).astype(np.float32)

    # Optional sampling to limit memory and runtime
    if args.sample is None:
        sample = min(20000, X.shape[0])
    else:
        sample = min(args.sample, X.shape[0])
    X = X[:sample]
    y = y[:sample]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())

    # Device selection and pin_memory
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    pin_memory = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, pin_memory=pin_memory, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, pin_memory=pin_memory, num_workers=args.num_workers)

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

    print('Training: device=', device, 'sample=', sample, 'batch_size=', args.batch, 'num_workers=', args.num_workers)
    for e in range(args.epochs):
        model.train()
        for i, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb).squeeze()
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            if i % 20 == 0:
                print(f'  epoch {e+1} batch {i} loss={loss.item():.4f}')
        model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for i, (xb, yb) in enumerate(val_loader):
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds.extend(torch.sigmoid(logits.squeeze()).cpu().numpy().tolist())
                labels.extend(yb.cpu().numpy().tolist())
                if i % 20 == 0:
                    print(f'  eval batch {i} processed')
        acc = accuracy_score((np.array(preds) > 0.5).astype(int), np.array(labels))
        print(f'Epoch {e+1} val_acc={acc:.4f}')
        if mlflow_module:
            try:
                mlflow_module.log_metric('val_acc', float(acc), step=e+1)
            except Exception:
                pass

    # Normalize out path to repo-root-aware output path
    out_path = resolve_cli_path(args.out)
    # Prevent accidental writes to repo root: if resolved output is under repo_root
    # but not under repo_root/ml, map it into ml/models to avoid polluting workspace
    try:
        rroot = repo_root()
        if out_path.exists():
            resolved_out = out_path.resolve()
        else:
            resolved_out = out_path
        if resolved_out.is_relative_to(rroot) if hasattr(resolved_out, 'is_relative_to') else str(resolved_out).startswith(str(rroot)):
            # If not under ml/, remap to ml/models
            if not str(resolved_out).startswith(str(rroot / 'ml')):
                out_path = rroot / 'ml' / 'models' / out_path.name
    except Exception:
        # Fallback: prefer ml/models under repo root for outputs that would otherwise land at project root
        try:
            out_path = (Path(__file__).resolve().parents[1] / 'ml' / 'models' / out_path.name).resolve()
        except Exception:
            pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print('Resolved input:', input_path)
    print('Resolved output:', out_path)
    torch.save(model.state_dict(), out_path)
    print('Saved CNN model to', out_path)
    if mlflow_module:
        try:
            mlflow.pytorch.log_model(model, artifact_path='model')
        except Exception:
            pass
