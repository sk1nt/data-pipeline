import subprocess
from pathlib import Path
ML_DIR = Path(__file__).resolve().parents[1]


def test_train_lstm_early_stop(tmp_path):
    sample = ML_DIR / 'output' / 'MNQ_20251111_1s_w60s_h1.npz'
    if not sample.exists():
        subprocess.run(['python', 'preprocess.py', '--inputs', str(ML_DIR / 'output' / 'MNQ_20251111_1s.parquet'), '--window', '60', '--stride', '1', '--features', 'open,high,low,close,volume,gex_zero,nq_spot', '--label-source', 'nq_spot'], check=True, cwd=str(ML_DIR))
    out = ML_DIR / 'models' / 'lstm_early.pt'
    if out.exists():
        out.unlink()
    subprocess.run(['python', 'train_lstm.py', '--input', str(sample), '--epochs', '5', '--batch', '256', '--dropout', '0.1', '--patience', '2', '--out', str(out)], check=True, cwd=str(ML_DIR))
    assert out.exists()
