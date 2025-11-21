"""
Train a longer LSTM run using the preprocessed dataset.
This script builds on `train_lstm.py` but runs more epochs and saves a checkpoint.
It expects `output/*.npz` to exist (created by `preprocess.py`).
"""
from pathlib import Path
import argparse
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split


def load_npz(path):
    data = np.load(path)
    return data['X'], data['y']


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def train(model, device, loader, opt, criterion):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(xb.float())
        loss = criterion(pred.squeeze(), yb.float())
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, device, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb.float()).squeeze()
            loss = criterion(pred, yb.float())
            total_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(pred) > 0.5).int()
            correct += (preds == yb.int()).sum().item()
            total += xb.size(0)
    return total_loss / len(loader.dataset), correct / total


def make_loaders(X, y, batch_size=128, test_size=0.1, val_size=0.1, seed=42):
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=test_size + val_size, random_state=seed)
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=rel_val, random_state=seed)
    import torch.utils.data as td
    train_ds = td.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = td.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = td.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='output/MNQ_20251111_1s_w60s_h1.npz')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out', type=str, default='models/lstm_long.pt')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=0, help='Early stopping patience (0 disables)')
    parser.add_argument('--mlflow', action='store_true', help='Log metrics and artifacts to MLflow if available')
    args = parser.parse_args()

    try:
        from ml.path_utils import resolve_cli_path
    except Exception:
        from path_utils import resolve_cli_path
    input_path = resolve_cli_path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Missing input {input_path}. Run ml/preprocess.py first")
    X, y = load_npz(input_path)
    # Convert continuous returns to binary labels by sign for classification
    y = (y > 0).astype('float32')
    print('Loaded shapes', X.shape, y.shape)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    # reduce dataset for speed if large
    max_samples = min(len(X), 50000)
    X = X[:max_samples]
    y = y[:max_samples]

    loaders = make_loaders(X, y, batch_size=args.batch)
    input_dim = X.shape[-1]
    model = LSTMModel(input_dim, args.hidden, args.layers, output_dim=1, dropout=args.dropout)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_epoch = 0
    mlflow = None
    if args.mlflow:
        try:
            import mlflow
            try:
                import mlflow_utils
                mlflow_utils.ensure_sqlite_tracking()
            except Exception:
                pass
            mlflow = mlflow
            mlflow.start_run()
            mlflow.log_params({
                'epochs': args.epochs,
                'batch': args.batch,
                'hidden': args.hidden,
                'layers': args.layers,
                'dropout': args.dropout,
                'lr': args.lr,
            })
        except Exception:
            print('MLflow not available; skipping logging')
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, device, loaders[0], opt, criterion)
        val_loss, val_acc = evaluate(model, device, loaders[1], criterion)
        print(f'Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            out_path = resolve_cli_path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)
            best_epoch = epoch
        if mlflow:
            try:
                mlflow.log_metric('train_loss', float(train_loss), step=epoch)
                mlflow.log_metric('val_loss', float(val_loss), step=epoch)
                mlflow.log_metric('val_acc', float(val_acc), step=epoch)
            except Exception:
                pass
        if args.patience > 0 and (epoch - best_epoch) >= args.patience:
            print('Early stopping at epoch', epoch)
            break
    test_loss, test_acc = evaluate(model, device, loaders[2], criterion)
    print('Test', test_loss, test_acc)
    if mlflow:
        try:
            mlflow.log_metric('test_loss', float(test_loss))
            mlflow.log_metric('test_acc', float(test_acc))
        except Exception:
            pass
        try:
            mlflow.log_artifact(out_path)
            import mlflow.pytorch
            mlflow.pytorch.log_model(model, artifact_path='model')
        except Exception:
            pass
        mlflow.end_run()


if __name__ == '__main__':
    main()
