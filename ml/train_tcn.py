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

OUT = Path('ml/models')
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
    args = p.parse_args()

    data = np.load(args.input)
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

    torch.save(model.state_dict(), args.out)
    print('Saved TCN model to', args.out)
