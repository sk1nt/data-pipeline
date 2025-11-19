#!/usr/bin/env python3
"""Train a simple PyTorch LSTM baseline on windowed datasets.

Usage:
  python ml/train_lstm.py --input ml/output/MNQ_20251111_1s_w60s_h1.npz
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

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        # take last timestep
        out = out[:, -1, :]
        out = self.fc(out)
        return out

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out', default=str(OUT / 'lstm_model.pt'))
    p.add_argument('--epochs', default=10, type=int)
    p.add_argument('--batch', default=64, type=int)
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
    model = LSTMModel(X.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

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
    torch.save(model.state_dict(), args.out)
    print('Saved LSTM model to', args.out)
