"""
Compare different models by running their training scripts with provided epochs and collecting results.
Saves metrics under `experiments/compare_{timestamp}/metrics.json` and checkpoints.
"""
from pathlib import Path
import argparse
import subprocess
import json
import time

BASE = Path('experiments')
BASE.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd):
    print('Running:', ' '.join(cmd))
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
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

    for m in models:
        if m == 'lstm':
            cmd = ['python', 'train_lstm.py', '--input', args.input, '--epochs', str(args.epochs), '--batch', str(args.batch), '--out', f'models/lstm_compare_{timestamp}.pt']
        elif m == 'cnn':
            cmd = ['python', 'train_cnn.py', '--input', args.input, '--epochs', str(args.epochs), '--batch', str(args.batch), '--out', f'models/cnn_compare_{timestamp}.pt']
        elif m == 'tcn':
            cmd = ['python', 'train_tcn.py', '--input', args.input, '--epochs', str(args.epochs), '--batch', str(args.batch), '--out', f'models/tcn_compare_{timestamp}.pt']
        elif m == 'transformer':
            cmd = ['python', 'train_transformer.py', '--input', args.input, '--epochs', str(args.epochs), '--batch', str(args.batch), '--out', f'models/transformer_compare_{timestamp}.pt']
        elif m == 'gru':
            cmd = ['python', 'train_lstm.py', '--input', args.input, '--epochs', str(args.epochs), '--batch', str(args.batch), '--model_type', 'gru', '--out', f'models/gru_compare_{timestamp}.pt']
        elif m == 'lightgbm':
            cmd = ['python', 'train_lightgbm.py', '--input', args.input, '--out', f'models/lightgbm_compare_{timestamp}.pkl']
        elif m == 'xgboost':
            cmd = ['python', 'train_xgboost.py', '--input', args.input, '--out', f'models/xgboost_compare_{timestamp}.pkl']
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
