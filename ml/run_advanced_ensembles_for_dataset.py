#!/usr/bin/env python3
"""Wrapper to run `advanced_ensembles.py` for a specific dataset.
This sets the `DATASET` global variable in the imported module then calls its main().
"""
import argparse
from pathlib import Path
import importlib

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, help='Path to .npz dataset (relative to ml/)')
    p.add_argument('--mlflow_experiment', default='advanced_ensembles', help='MLflow experiment name')
    args = p.parse_args()

    # Import the module and override dataset
    import advanced_ensembles as ae
    ae.DATASET = args.dataset
    # Forward the MLflow experiment param by setting arg in sys.argv
    import sys
    sys.argv = [sys.argv[0], '--mlflow_experiment', args.mlflow_experiment]
    ae.main()

if __name__ == '__main__':
    main()
