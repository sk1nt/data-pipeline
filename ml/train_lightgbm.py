#!/usr/bin/env python3
"""Train a LightGBM baseline on windowed datasets.

Usage:
  python train_lightgbm.py --input output/MNQ_20251111_1s_w60s_h1.npz
"""
import argparse
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import pickle
from pathlib import Path

OUT = Path(__file__).resolve().parents[0] / 'models'
OUT.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out', default=str(OUT / 'lightgbm_model.pkl'))
    p.add_argument('--use-gpu', action='store_true', help='Use GPU for LightGBM if available')
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

    # classification: predict sign of return
    y_class = (y > 0).astype(int)
    X_flat = X.reshape(X.shape[0], -1)

    X_train, X_val, y_train, y_val = train_test_split(X_flat, y_class, test_size=0.2, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    # Use GPU if requested via CLI; requires GPU-enabled LightGBM
    gpu_flag = params = None
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt'
    }
    if args.use_gpu:
        params['device'] = 'gpu'

    mlflow_module = None
    if args.mlflow:
        try:
            import mlflow
            import mlflow.lightgbm
            try:
                import mlflow_utils
                mlflow_utils.ensure_sqlite_tracking()
            except Exception:
                pass
            mlflow_module = mlflow
            mlflow_module.start_run()
            mlflow_module.log_params({'use_gpu': args.use_gpu, 'num_rounds': 100})
        except Exception:
            print('MLflow not available; skipping mlflow logging')
            mlflow_module = None
    try:
        model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[val_data])
    except Exception as e:
        print('LightGBM training error (maybe GPU unavailable):', e)
        print('Retrying on CPU...')
        params.pop('device', None)
        model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[val_data])

    preds = model.predict(X_val)
    auc = roc_auc_score(y_val, preds)
    acc = accuracy_score(y_val, (preds > 0.5).astype(int))

    print('AUC:', auc, 'ACC:', acc)
    if mlflow_module:
        try:
            mlflow_module.log_metric('auc', float(auc))
            mlflow_module.log_metric('acc', float(acc))
        except Exception:
            pass
        try:
            mlflow.lightgbm.log_model(model, artifact_path='model')
        except Exception as exc:
            print('MLflow model log skipped:', exc)
        finally:
            mlflow_module.end_run()
    out_path = resolve_cli_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(model, f)
    print('Saved model to', out_path)
