#!/usr/bin/env python3
"""Run PnL simulation across multiple models for the October week and log results.
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ML_DIR = Path(__file__).resolve().parent
MODELS = [
    # (model_path, scaler, instrument); make absolute paths under ml/models
    (str(ML_DIR / 'models' / 'lstm_trading.pt'), str(ML_DIR / 'models' / 'scaler_13_features.pkl'), 'MNQ'),
    (str(ML_DIR / 'models' / 'lstm_production.pt'), str(ML_DIR / 'models' / 'scaler_13_features.pkl'), 'MNQ'),
    (str(ML_DIR / 'models' / 'enhanced_production.pt'), str(ML_DIR / 'models' / 'enhanced_gex_scaler.pkl'), 'MNQ'),
    (str(ML_DIR / 'models' / 'lightgbm_tuned.pkl'), str(ML_DIR / 'models' / 'scaler_13_features.pkl'), 'MNQ'),
]

def run_for_model(model_spec, days):
    model, scaler, instrument = model_spec
    cmd = [
        sys.executable, '-m', 'ml.pnl_backtest',
        '--mlflow', '--experiment', 'oct_week_multi_model',
        '--model', model,
        '--days', ','.join(days),
        '--window', '60',
        '--threshold', '0.5',
        '--instrument', instrument,
    ]
    if scaler:
        cmd += ['--scaler', scaler]
    # Add stop loss and position sizing defaults
    cmd += ['--position_size', '1', '--stop_loss', '100', '--take_profit', '200']
    print('Running:', ' '.join(cmd))
    # Run from repo root so package imports (`import ml`) resolve correctly in the child process
    res = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=str(ML_DIR.parent))
    print(res.stdout)
    # Parse summary to detect catastrophic runs and surface quickly
    total_pnl = None
    avg_win_rate = None
    total_trades = None
    try:
        import re
        m = re.search(r"Total PnL:\\s*\\$([-+0-9.,]+)", res.stdout)
        if m:
            total_pnl = float(m.group(1).replace(',', ''))
        m = re.search(r"Average win rate:\\s*([0-9.]+)%", res.stdout)
        if m:
            avg_win_rate = float(m.group(1)) / 100.0
        m = re.search(r"Total trades:\\s*([0-9.,]+)", res.stdout)
        if m:
            total_trades = float(m.group(1).replace(',', ''))
    except Exception:
        pass

    if total_pnl is not None and total_trades:
        if total_trades > 1000 and total_pnl < -50000:
            print(f"❌ Catastrophic PnL detected for {model}: total_pnl={total_pnl}, trades={total_trades}. Stopping remaining runs.")
            return 2
        if avg_win_rate is not None and avg_win_rate < 0.05:
            print(f"⚠️  Very low win rate ({avg_win_rate:.2%}) for {model}; consider checking commissions/thresholds.")

    if res.returncode != 0:
        print('Error:', res.stderr)
    return res.returncode


def main():
    # Oct week: 2025-10-20 -> 2025-10-24
    days = [d.strftime('%Y%m%d') for d in [datetime(2025, 10, 20), datetime(2025, 10, 21), datetime(2025, 10, 22), datetime(2025, 10, 23), datetime(2025, 10, 24)]]
    for spec in MODELS:
        rc = run_for_model(spec, days)
        if rc == 2:
            break
        if rc != 0:
            print(f"Model {spec[0]} failed with exit code {rc}")

if __name__ == '__main__':
    main()
