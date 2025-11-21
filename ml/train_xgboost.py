#!/usr/bin/env python3
"""Train an XGBoost classifier on flattened windows as a baseline.
"""
import argparse
import numpy as np
import xgboost as xgb
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from pathlib import Path

OUT = Path(__file__).resolve().parents[0] / 'models'
OUT.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out', default=str(OUT / 'xgb_model.json'))
    p.add_argument('--use-gpu', action='store_true', help='Use GPU for XGBoost if available')
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
    y_class = (y > 0).astype(int)

    sample = min(20000, X.shape[0])
    X = X[:sample]
    y_class = y_class[:sample]

    X_flat = X.reshape(X.shape[0], -1)
    X_train, X_val, y_train, y_val = train_test_split(X_flat, y_class, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', tree_method='gpu_hist' if args.use_gpu else 'hist')
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, preds)
    acc = accuracy_score(y_val, (preds > 0.5).astype(int))
    print('AUC:', auc, 'ACC:', acc)
    out_path = resolve_cli_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_path))
    if args.mlflow:
        try:
            import mlflow
            import mlflow.xgboost
            mlflow.start_run()
            mlflow.log_params({'use_gpu': args.use_gpu})
            mlflow.log_metric('auc', float(auc))
            mlflow.log_metric('acc', float(acc))
            try:
                mlflow.xgboost.log_model(model, artifact_path='model')
            except Exception:
                mlflow.log_artifact(str(out_path))
            mlflow.end_run()
        except Exception:
            print('MLflow not available; skipping mlflow logging')
    print('Saved xgb model to', out_path)
