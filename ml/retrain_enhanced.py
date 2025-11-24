#!/usr/bin/env python3
"""Retrain LSTM model and scaler with enhanced GEX features."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)

def load_enhanced_data(sample_files):
    """Load and combine data from multiple days with enhanced GEX features."""
    import sys
    sys.path.append('.')
    from backtest_model import preprocess_day_data

    X_list = []
    y_list = []

    for file_path in sample_files:
        print(f"Processing {file_path}...")
        X, y = preprocess_day_data(file_path, scaler=None)
        if X is not None and y is not None:
            X_list.append(X)
            y_list.append(y)
            print(f"  Loaded {len(X)} sequences")
        else:
            print(f"  Failed to load {file_path}")

    if not X_list:
        raise ValueError("No data loaded")

    X_combined = np.concatenate(X_list, axis=0)
    y_combined = np.concatenate(y_list, axis=0)

    print(f"\nCombined dataset: {len(X_combined)} sequences, {X_combined.shape[2]} features")
    print(f"Target distribution: {np.mean(y_combined):.1%} positive")

    return X_combined, y_combined

def retrain_scaler_and_model(X, y, model_path='ml/models/enhanced_gex_model.pt', scaler_path='ml/models/enhanced_gex_scaler.pkl'):
    """Retrain both scaler and model with enhanced features."""

    print("Retraining model with enhanced GEX features (no scaling like original)...")

    # Split data (no scaling for now, like original model)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")

    # Create dummy scaler (identity transform) for compatibility
    scaler = StandardScaler()
    X_dummy = np.ones((100, X.shape[2]))  # Dummy data to fit scaler
    scaler.fit(X_dummy)  # This creates an identity-like transform

    # Use raw data (no scaling)
    X_train_scaled = X_train.astype(np.float32)
    X_val_scaled = X_val.astype(np.float32)
    X_test_scaled = X_test.astype(np.float32)

    # Create data loaders
    train_dataset = TensorDataset(torch.from_numpy(X_train_scaled).float(),
                                torch.from_numpy(y_train).float().unsqueeze(1))
    val_dataset = TensorDataset(torch.from_numpy(X_val_scaled).float(),
                              torch.from_numpy(y_val).float().unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Model with enhanced input dimension (21 features)
    input_dim = X.shape[2]  # Should be 21
    model = LSTMModel(input_dim=input_dim, hidden_dim=256, num_layers=2, dropout=0.2)

    print(f"\nRetraining LSTM model with {input_dim} features...")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0

    epochs = 50
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                preds = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader)

        # Calculate metrics
        val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
        val_targets = np.array(val_targets).astype(int)

        accuracy = accuracy_score(val_targets, val_preds_binary)
        precision = precision_score(val_targets, val_preds_binary, zero_division=0)
        recall = recall_score(val_targets, val_preds_binary, zero_division=0)

        print(f"Epoch {epoch+1:2d}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Acc: {accuracy:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step(val_loss)

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Save model and scaler
    print("\nSaving enhanced model and scaler...")

    # Create checkpoint with model, scaler, and metadata
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'input_dim': input_dim,
        'hidden_dim': 256,
        'num_layers': 2,
        'features': [
            'open', 'high', 'low', 'close', 'volume', 'zero_gamma', 'spot_price',
            'net_gex', 'major_pos_vol', 'major_neg_vol', 'sum_gex_vol', 'delta_risk_reversal',
            'max_priors_current', 'max_priors_1m', 'max_priors_5m',
            'adx', 'di_plus', 'di_minus', 'rsi', 'stoch_k', 'stoch_d'
        ],
        'trained_at': datetime.now().isoformat(),
        'val_loss': best_val_loss
    }

    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")

    # Also save scaler separately for convenience
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    model.eval()
    test_preds = []
    test_targets = []

    test_dataset = TensorDataset(torch.from_numpy(X_test_scaled).float(),
                               torch.from_numpy(y_test).float().unsqueeze(1))
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = torch.sigmoid(outputs).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(y_batch.cpu().numpy())

    test_preds_binary = (np.array(test_preds) > 0.5).astype(int)
    test_targets = np.array(test_targets).astype(int)

    final_accuracy = accuracy_score(test_targets, test_preds_binary)
    final_precision = precision_score(test_targets, test_preds_binary, zero_division=0)
    final_recall = recall_score(test_targets, test_preds_binary, zero_division=0)
    final_f1 = f1_score(test_targets, test_preds_binary, zero_division=0)

    print("Test Results:")
    print(f"  Accuracy: {final_accuracy:.4f}")
    print(f"  Precision: {final_precision:.4f}")
    print(f"  Recall: {final_recall:.4f}")
    print(f"  F1 Score: {final_f1:.4f}")

    return model, scaler, {
        'accuracy': final_accuracy,
        'precision': final_precision,
        'recall': final_recall,
        'f1': final_f1,
        'val_loss': best_val_loss
    }

if __name__ == '__main__':
    # Sample files for retraining (adjust as needed)
    sample_files = [
        'data/parquet/tick/MNQ/20251114.parquet',
        'data/parquet/tick/MNQ/20251113.parquet',
        'data/parquet/tick/MNQ/20251112.parquet'
    ]

    print("Loading enhanced GEX data...")
    X, y = load_enhanced_data(sample_files)

    print(f"\nRetraining with {X.shape[2]} enhanced features...")
    model, scaler, metrics = retrain_scaler_and_model(X, y)

    print("\n✅ Retraining complete!")
    print("Model: ml/models/enhanced_gex_model.pt")
    print("Scaler: ml/models/enhanced_gex_scaler.pkl")
    print(f"Features: {X.shape[2]} (enhanced with comprehensive GEX signals)")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {metrics['f1']:.4f}")