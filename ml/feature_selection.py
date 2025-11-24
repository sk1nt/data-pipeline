#!/usr/bin/env python3
"""Feature selection and importance analysis for enhanced trading models.

Uses multiple techniques to identify the most predictive features from the 60-feature set.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import torch
import mlflow
import mlflow.pytorch
import warnings
warnings.filterwarnings('ignore')

# Dataset
DATASET = 'output/MNQ_multi_day_enhanced.npz'
FEATURES_FILE = 'output/MNQ_multi_day_enhanced_features.txt'

class LSTMModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.0, model_type='lstm'):
        super().__init__()
        if model_type == 'lstm':
            self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        elif model_type == 'gru':
            self.rnn = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)

def load_data_and_features():
    """Load dataset and feature names."""
    data = np.load(DATASET)
    X = data['X']
    y = data['y']

    # Load feature names
    with open(FEATURES_FILE, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]

    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[2]} features")
    print(f"Features: {feature_names[:10]}...")

    return X, y, feature_names

def flatten_for_sklearn(X):
    """Flatten time series data for sklearn feature selection."""
    n_samples, n_timesteps, n_features = X.shape
    # Take only the last timestep (most recent) for feature selection
    X_flat = X[:, -1, :]  # Shape: (n_samples, n_features)
    return X_flat

def mutual_info_selection(X_flat, y, feature_names, k=20):
    """Select features using mutual information."""
    print(f"\nRunning mutual information selection (k={k})...")

    # Convert to binary classification for mutual info
    y_binary = (y > 0).astype(int)

    selector = SelectKBest(score_func=mutual_info_regression, k=k)
    selector.fit_transform(X_flat, y_binary)

    # Get selected feature indices and scores
    selected_indices = selector.get_support(indices=True)
    scores = selector.scores_

    selected_features = [feature_names[i] for i in selected_indices]
    selected_scores = scores[selected_indices]

    print(f"Top {k} features by mutual information:")
    for feat, score in zip(selected_features, selected_scores):
        print(f"{feat}: {score:.4f}")

    return selected_indices, selected_features, scores

def lasso_selection(X_flat, y, feature_names, alpha_range=None):
    """Select features using LASSO regression."""
    print("\nRunning LASSO feature selection...")

    if alpha_range is None:
        alpha_range = np.logspace(-4, 0, 50)

    # Use binary labels for LASSO
    y_binary = (y > 0).astype(int)

    lasso = LassoCV(alphas=alpha_range, cv=5, random_state=42)
    lasso.fit(X_flat, y_binary)

    # Get coefficients
    coefficients = lasso.coef_

    # Select features with non-zero coefficients
    selected_indices = np.where(coefficients != 0)[0]
    selected_features = [feature_names[i] for i in selected_indices]
    selected_coeffs = coefficients[selected_indices]

    print(f"LASSO selected {len(selected_features)} features with alpha={lasso.alpha_:.6f}")
    print("Top features by coefficient magnitude:")
    sorted_idx = np.argsort(np.abs(selected_coeffs))[::-1]
    for i in sorted_idx[:10]:
        print(f"{selected_features[i]}: {selected_coeffs[i]:.4f}")

    return selected_indices, selected_features, coefficients

def random_forest_importance(X_flat, y, feature_names, n_estimators=100):
    """Get feature importance using Random Forest."""
    print("\nRunning Random Forest feature importance...")

    # Use binary labels
    y_binary = (y > 0).astype(int)

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_flat, y_binary)

    importances = rf.feature_importances_

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    print("Top 20 features by Random Forest importance:")
    for i in range(min(20, len(indices))):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    return indices, importances

def correlation_analysis(X_flat, y, feature_names, threshold=0.1):
    """Analyze feature correlations with target."""
    print("\nRunning correlation analysis...")

    # Calculate correlations with target
    y_binary = (y > 0).astype(int)
    correlations = []

    for i, feature_name in enumerate(feature_names):
        corr = np.corrcoef(X_flat[:, i], y_binary)[0, 1]
        correlations.append((feature_name, abs(corr), corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)

    print("Top 20 features by correlation with target:")
    for feat, abs_corr, corr in correlations[:20]:
        print(f"{feat}: {corr:.4f} (|corr|={abs_corr:.4f})")

    # Select features above threshold
    selected = [(feat, corr) for feat, abs_corr, corr in correlations if abs_corr >= threshold]
    selected_features = [feat for feat, _ in selected]
    selected_indices = [feature_names.index(feat) for feat in selected_features]

    print(f"Selected {len(selected_features)} features with correlation >= {threshold}")

    return selected_indices, selected_features, correlations

def create_feature_subsets(feature_names, methods_results):
    """Create consensus feature subsets from multiple methods."""
    print("\nCreating consensus feature subsets...")

    subsets = {}

    # Individual method subsets
    for method_name, (indices, features, _) in methods_results.items():
        subsets[method_name] = features
        print(f"{method_name}: {len(features)} features")

    # Consensus subsets
    all_selected_features = []
    for method_results in methods_results.values():
        all_selected_features.extend(method_results[1])

    from collections import Counter
    feature_counts = Counter(all_selected_features)

    # Features selected by multiple methods
    consensus_2plus = [feat for feat, count in feature_counts.items() if count >= 2]
    consensus_3plus = [feat for feat, count in feature_counts.items() if count >= 3]

    subsets['consensus_2plus'] = consensus_2plus
    subsets['consensus_3plus'] = consensus_3plus

    print(f"Consensus (2+ methods): {len(consensus_2plus)} features")
    print(f"Consensus (3+ methods): {len(consensus_3plus)} features")

    # Show most selected features
    print("\nMost frequently selected features:")
    for feat, count in feature_counts.most_common(15):
        methods = [method for method, (_, features, _) in methods_results.items() if feat in features]
        print(f"  {feat}: selected by {count} methods ({', '.join(methods)})")

    return subsets

def evaluate_feature_subset(X, y, feature_indices, subset_name, mlflow_experiment):
    """Train and evaluate a model on a feature subset."""
    print(f"\nEvaluating {subset_name} ({len(feature_indices)} features)...")

    # Create subset
    X_subset = X[:, :, feature_indices]

    # Use only a small subset of data to avoid memory issues
    n_samples = min(10000, len(X_subset))  # Use max 10K samples
    indices = np.random.choice(len(X_subset), n_samples, replace=False)
    X_subset = X_subset[indices]
    y_subset = y[indices]

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X_subset, y_subset, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Use Logistic Regression instead of LSTM for faster evaluation

    # Flatten for sklearn
    X_train_flat = X_train[:, -1, :]  # Use last timestep
    X_test_flat = X_test[:, -1, :]

    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_flat, (y_train > 0).astype(int))

    # Evaluate
    test_preds_proba = model.predict_proba(X_test_flat)[:, 1]

    # Calculate metrics at optimal threshold (0.3)
    threshold = 0.3
    preds_binary = (test_preds_proba >= threshold).astype(int)
    labels_binary = (y_test > 0).astype(int)

    trades_taken = np.sum(preds_binary == 1)
    correct_trades = np.sum((preds_binary == 1) & (labels_binary == 1))
    win_rate = correct_trades / trades_taken if trades_taken > 0 else 0

    # P&L calculation
    commission_cost = 0.42
    avg_profit_per_win = 10.0

    total_commissions = trades_taken * commission_cost
    gross_pnl = correct_trades * avg_profit_per_win
    net_pnl = gross_pnl - total_commissions

    edge = win_rate - (commission_cost / avg_profit_per_win)

    print(".4f")
    print(".1%")
    print(".2f")
    print(f"Trades: {trades_taken}, Edge: {edge:.1%}")

    # Log to MLflow
    with mlflow.start_run(run_name=f"feature_selection_{subset_name}"):
        mlflow.log_param("subset_name", subset_name)
        mlflow.log_param("num_features", len(feature_indices))
        mlflow.log_param("features", str(feature_indices))
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("commission_cost", commission_cost)
        mlflow.log_param("samples_used", n_samples)

        mlflow.log_metric("win_rate", win_rate)
        mlflow.log_metric("net_pnl", net_pnl)
        mlflow.log_metric("trades_taken", trades_taken)
        mlflow.log_metric("edge", edge)
        mlflow.log_metric("total_commissions", total_commissions)

    return {
        'subset': subset_name,
        'num_features': len(feature_indices),
        'win_rate': win_rate,
        'net_pnl': net_pnl,
        'trades_taken': trades_taken,
        'edge': edge
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlflow_experiment', default='feature_selection_analysis', help='MLflow experiment name')
    parser.add_argument('--max_features', type=int, default=20, help='Max features to select per method')
    args = parser.parse_args()

    try:
        import mlflow_utils
        mlflow_utils.ensure_sqlite_tracking()
    except Exception:
        pass
    mlflow.set_experiment(args.mlflow_experiment)

    # Load data
    X, y, feature_names = load_data_and_features()
    X_flat = flatten_for_sklearn(X)

    # Run feature selection methods
    methods_results = {}

    # 1. Mutual Information
    mi_indices, mi_features, mi_scores = mutual_info_selection(
        X_flat, y, feature_names, k=args.max_features)
    methods_results['mutual_info'] = (mi_indices, mi_features, mi_scores)

    # 2. LASSO
    lasso_indices, lasso_features, lasso_coeffs = lasso_selection(
        X_flat, y, feature_names)
    methods_results['lasso'] = (lasso_indices, lasso_features, lasso_coeffs)

    # 3. Random Forest
    rf_indices, rf_importances = random_forest_importance(
        X_flat, y, feature_names, n_estimators=50)
    rf_top_indices = rf_indices[:args.max_features]
    rf_top_features = [feature_names[i] for i in rf_top_indices]
    methods_results['random_forest'] = (rf_top_indices, rf_top_features, rf_importances)

    # 4. Correlation
    corr_indices, corr_features, correlations = correlation_analysis(
        X_flat, y, feature_names, threshold=0.01)
    methods_results['correlation'] = (corr_indices, corr_features, correlations)

    # Create consensus subsets
    subsets = create_feature_subsets(feature_names, methods_results)

    # Evaluate each subset
    print("\n" + "="*60)
    print("EVALUATING FEATURE SUBSETS")
    print("="*60)

    evaluation_results = []

    # Evaluate individual method subsets
    for method_name, features in subsets.items():
        if not features:
            continue

        feature_indices = [feature_names.index(feat) for feat in features]
        result = evaluate_feature_subset(X, y, feature_indices, method_name, args.mlflow_experiment)
        evaluation_results.append(result)

    # Sort results by net P&L
    evaluation_results.sort(key=lambda x: x['net_pnl'], reverse=True)

    print("\n" + "="*60)
    print("FEATURE SELECTION RESULTS")
    print("="*60)

    print("\nBest performing feature subsets:")
    for i, result in enumerate(evaluation_results[:10]):
        print(f"{i+1:2d}. {result['subset'][:20]:20} | Features: {result['num_features']:2d} | "
              f"P&L: ${result['net_pnl']:8.2f} | Win: {result['win_rate']:5.1%} | "
              f"Trades: {result['trades_taken']:4d} | Edge: {result['edge']:5.1%}")

    # Compare with baseline (all 60 features)
    print("\nBaseline (all 60 features):")
    print("      P&L: $6430.16, Win: 18.6%, Trades: 4452, Edge: 14.4%")

    # Save detailed results
    df_results = pd.DataFrame(evaluation_results)
    df_results.to_csv('feature_selection_results.csv', index=False)

    # Save feature subsets for future use
    with open('selected_feature_subsets.txt', 'w') as f:
        f.write("Feature Selection Results\n")
        f.write("="*50 + "\n\n")

        for subset_name, features in subsets.items():
            f.write(f"{subset_name} ({len(features)} features):\n")
            for feat in features:
                f.write(f"  {feat}\n")
            f.write("\n")

    print("\nDetailed results saved to feature_selection_results.csv")
    print("Feature subsets saved to selected_feature_subsets.txt")

    print("\n✅ Feature selection analysis complete!")
    best = evaluation_results[0]
    print(f"Best subset: {best['subset']} ({best['num_features']} features)")
    print(".2f")
    print(".1%")
    print(f"Improvement over baseline: ${best['net_pnl'] - 6430.16:.2f}")

if __name__ == '__main__':
    main()
