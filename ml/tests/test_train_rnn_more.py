import subprocess
from pathlib import Path
ML_DIR = Path(__file__).resolve().parents[1]


def test_train_rnn_more_smoke(tmp_path):
    # assumes preprocess has been run; fallback to sample file if not
    sample = ML_DIR / 'output' / 'MNQ_20251111_1s_w60s_h1.npz'
    if not sample.exists():
        # run preprocess for sample date with gex and nq_spot labels
        subprocess.run([
            'python', 'preprocess.py', '--inputs', str(ML_DIR / 'output' / 'MNQ_20251111_1s.parquet'),
            '--window', '60', '--stride', '1', '--features', 'open,high,low,close,volume,gex_zero,nq_spot',
            '--label-source', 'nq_spot'
        ], check=True, cwd=str(ML_DIR))
    out = ML_DIR / 'models' / 'lstm_long.pt'
    if out.exists():
        out.unlink()
    subprocess.run(['python', 'train_rnn_more.py', '--input', str(sample), '--epochs', '2', '--batch', '256', '--out', str(out)], check=True, cwd=str(ML_DIR))
    assert out.exists()
