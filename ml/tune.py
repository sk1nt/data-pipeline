#!/usr/bin/env python3
"""Hyperparameter tuning script for LSTM using Optuna (fallback random search).

Saves best hyperparams and model to `ml/experiments/<timestamp>/`.

Usage:
 python ml/tune.py --input ml/output/MNQ_20251111_1s_w60s_h1.npz --trials 10 --max-epochs 3

If Optuna is not installed, it falls back to a simple random search.
"""
import argparse
import os
from pathlib import Path
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import random

try:
    import optuna
    OPTUNA = True
except Exception:
    OPTUNA = False

OUT = Path('ml/experiments')
OUT.mkdir(parents=True, exist_ok=True)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def train_eval(X_train, X_val, y_train, y_val, hidden_dim, num_layers, lr, batch_size, epochs, device):
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    model = LSTMModel(X_train.shape[2], hidden_dim, num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for e in range(epochs):
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
    return acc, model


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--max-epochs', type=int, default=3)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--sample', type=int, default=20000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    data = np.load(args.input)
    X = data['X']
    y = data['y']
    y = (y > 0).astype(np.float32)

    # sample subset to speed up tuning
    sample = min(args.sample, X.shape[0])
    X = X[:sample]
    y = y[:sample]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    run_dir = OUT / f'lstm_tune_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)

    best = {'acc': -1, 'params': None}

    def objective(trial):
        hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        epochs = args.max_epochs
        acc, _ = train_eval(X_train, X_val, y_train, y_val, hidden_dim, num_layers, lr, batch_size, epochs, device)
        trial.report(acc, step=0)
        return acc

    if OPTUNA:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=args.trials)
        best_trial = study.best_trial
        best['acc'] = best_trial.value
        best['params'] = best_trial.params
    else:
        print('Optuna not installed; running a small random search.')
        for i in range(args.trials):
            hidden_dim = random.randint(32, 256)
            num_layers = random.randint(1,3)
            lr = 10**random.uniform(-4, -2)
            batch_size = random.choice([64, 128, 256])
            acc, model = train_eval(X_train, X_val, y_train, y_val, hidden_dim, num_layers, lr, batch_size, args.max_epochs, device)
            if acc > best['acc']:
                best['acc'] = acc
                best['params'] = dict(hidden_dim=hidden_dim, num_layers=num_layers, lr=lr, batch_size=batch_size)

    print('Best:', best)
    with open(run_dir / 'best.json', 'w') as f:
        json.dump(best, f, indent=2)

    # Train best model on full sample and save it
    params = best['params']
    if params:
        acc, model = train_eval(X_train, X_val, y_train, y_val, params['hidden_dim'], params['num_layers'], params['lr'], params['batch_size'], args.max_epochs, device)
        torch.save(model.state_dict(), run_dir / 'best_model.pt')
        print('Saved best model to', run_dir / 'best_model.pt')

    print('Done. Results in:', run_dir)
