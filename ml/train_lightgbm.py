#!/usr/bin/env python3
"""Train a LightGBM baseline on windowed datasets.

Usage:
  python ml/train_lightgbm.py --input ml/output/MNQ_20251111_1s_w60s_h1.npz
"""
import argparse
import numpy as np
import lightgbm as lgb
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import pickle
from pathlib import Path

OUT = Path('ml/models')
OUT.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out', default=str(OUT / 'lightgbm_model.pkl'))
    p.add_argument('--use-gpu', action='store_true', help='Use GPU for LightGBM if available')
    args = p.parse_args()

    data = np.load(args.input)
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

    with open(args.out, 'wb') as f:
        pickle.dump(model, f)
    print('Saved model to', args.out)
