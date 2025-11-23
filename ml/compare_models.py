"""
Compare different models by running their training scripts with provided epochs and collecting results.
Saves metrics under `experiments/compare_{timestamp}/metrics.json` and checkpoints.
"""
from pathlib import Path
import argparse
import subprocess
import json
import time
try:
    from ml.path_utils import resolve_cli_path
except Exception:
    from path_utils import resolve_cli_path

ML_DIR = Path(__file__).resolve().parent
BASE = ML_DIR / 'experiments'
BASE.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd):
    print('Running:', ' '.join(cmd))
    # Run from within ml/ to ensure relative paths resolve correctly
    res = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(ML_DIR))
    return res.stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='output/MNQ_20251111_1s_w60s_h1.npz')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--models', default='lstm,cnn,tcn,transformer,lightgbm,xgboost,gru')
    args = p.parse_args()

    timestamp = time.strftime('%Y%m%dT%H%M%S')
    out_dir = BASE / f'compare_{timestamp}'
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    results = {}

    # Resolve input to repo-local path to avoid accidental root writes
    args.input = str(resolve_cli_path(args.input))
    for m in models:
        if m == 'lstm':
            cmd = ['python', str(ML_DIR / 'train_lstm.py'), '--input', args.input, '--epochs', str(args.epochs), '--batch', str(args.batch), '--out', str(ML_DIR / 'models' / f'lstm_compare_{timestamp}.pt')]
        elif m == 'cnn':
            cmd = ['python', str(ML_DIR / 'train_cnn.py'), '--input', args.input, '--epochs', str(args.epochs), '--batch', str(args.batch), '--out', str(ML_DIR / 'models' / f'cnn_compare_{timestamp}.pt')]
        elif m == 'tcn':
            cmd = ['python', str(ML_DIR / 'train_tcn.py'), '--input', args.input, '--epochs', str(args.epochs), '--batch', str(args.batch), '--out', str(ML_DIR / 'models' / f'tcn_compare_{timestamp}.pt')]
        elif m == 'transformer':
            cmd = ['python', str(ML_DIR / 'train_transformer.py'), '--input', args.input, '--epochs', str(args.epochs), '--batch', str(args.batch), '--out', str(ML_DIR / 'models' / f'transformer_compare_{timestamp}.pt')]
        elif m == 'gru':
            cmd = ['python', str(ML_DIR / 'train_lstm.py'), '--input', args.input, '--epochs', str(args.epochs), '--batch', str(args.batch), '--model_type', 'gru', '--out', str(ML_DIR / 'models' / f'gru_compare_{timestamp}.pt')]
        elif m == 'lightgbm':
            cmd = ['python', str(ML_DIR / 'train_lightgbm.py'), '--input', args.input, '--out', str(ML_DIR / 'models' / f'lightgbm_compare_{timestamp}.pkl')]
        elif m == 'xgboost':
            cmd = ['python', str(ML_DIR / 'train_xgboost.py'), '--input', args.input, '--out', str(ML_DIR / 'models' / f'xgboost_compare_{timestamp}.pkl')]
        else:
            print('Unknown model', m)
            continue
        try:
            out = run_cmd(cmd)
            # capture last line for val/test metrics if printed
            last_lines = out.strip().splitlines()[-5:]
            results[m] = '\n'.join(last_lines)
            Path(out_dir / f'{m}.log').write_text(out)
        except subprocess.CalledProcessError as e:
            print('Model failed:', m, e)
            results[m] = f'error: {e}'

    Path(out_dir / 'metrics.json').write_text(json.dumps(results, indent=2))
    print('Saved results to', out_dir)


if __name__ == '__main__':
    main()
