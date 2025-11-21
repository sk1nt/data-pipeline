#!/usr/bin/env python3
"""Run PnL simulation across multiple models for the first full week of a month.

Computes the first Monday of the month and runs Friday-inclusive (Mon-Fri) backtests
for the `MODELS` defined below.
"""
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta

ML_DIR = Path(__file__).resolve().parent
MODELS = [
    (str(ML_DIR / 'models' / 'lstm_trading.pt'), str(ML_DIR / 'models' / 'scaler_13_features.pkl'), 'MNQ'),
    (str(ML_DIR / 'models' / 'lstm_production.pt'), str(ML_DIR / 'models' / 'scaler_13_features.pkl'), 'MNQ'),
    (str(ML_DIR / 'models' / 'enhanced_production.pt'), str(ML_DIR / 'models' / 'enhanced_gex_scaler.pkl'), 'MNQ'),
    (str(ML_DIR / 'models' / 'lightgbm_tuned.pkl'), str(ML_DIR / 'models' / 'scaler_13_features.pkl'), 'MNQ'),
]


def first_monday(year: int, month: int) -> datetime:
    d = datetime(year, month, 1)
    # Monday is weekday() == 0
    offset = (0 - d.weekday()) % 7
    return d + timedelta(days=offset)


def run_for_model(model_spec, days, experiment, bar_type='time', bar_size=1):
    model, scaler, instrument = model_spec
    cmd = [
        sys.executable, '-m', 'ml.pnl_backtest',
        '--mlflow', '--experiment', experiment,
        '--model', model,
        '--days', ','.join(days),
        '--window', '60',
        '--threshold', '0.5',
        '--instrument', instrument,
    ]
    if scaler:
        cmd += ['--scaler', scaler]
    cmd += ['--bar-type', bar_type, '--bar-size', str(bar_size)]
    cmd += ['--position_size', '1', '--stop_loss', '100', '--take_profit', '200']
    print('Running:', ' '.join(cmd))
    res = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=str(ML_DIR.parent))
    print(res.stdout)
    if res.returncode != 0:
        print('Error:', res.stderr)
    return res.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--month', type=int, required=True)
    parser.add_argument('--experiment', default=None, help='MLflow experiment name')
    parser.add_argument('--bar-type', default='time', choices=['time','volume','dollar'], help='Bar type to locate preprocessed files')
    parser.add_argument('--bar-size', default=1, type=float, help='Bar size parameter to locate preprocessed files')
    args = parser.parse_args()
    exp = args.experiment or f'first_week_{args.year}_{args.month:02d}_multi_model'

    start = first_monday(args.year, args.month)
    days = [(start + timedelta(days=i)).strftime('%Y%m%d') for i in range(5)]

    print(f"First full week for {args.year}-{args.month:02d} starting {start.date()} => days: {days}")

    for spec in MODELS:
        rc = run_for_model(spec, days, exp, bar_type=args.bar_type, bar_size=args.bar_size)
        if rc != 0:
            print(f"Model {spec[0]} failed with exit code {rc}")


if __name__ == '__main__':
    main()
