#!/usr/bin/env python3
"""Wrapper to run `threshold_optimization.py` for a specific dataset and optional model paths.
Overlays the DATASET constant and optional model paths before calling the module's main.
"""
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, help='Path to .npz dataset (relative to ml/)')
    p.add_argument('--mlflow_experiment', default='threshold_optimization', help='MLflow experiment name')
    p.add_argument('--lstm_model', default=None, help='Optional LSTM model path')
    p.add_argument('--lgb_model', default=None, help='Optional LightGBM model path')
    args = p.parse_args()

    import threshold_optimization as to
    to.DATASET = args.dataset
    if args.lstm_model:
        to.LSTM_MODEL = Path(args.lstm_model)
    if args.lgb_model:
        to.LGB_MODEL = Path(args.lgb_model)

    import sys
    sys.argv = [sys.argv[0], '--mlflow_experiment', args.mlflow_experiment]
    to.main()

if __name__ == '__main__':
    main()
