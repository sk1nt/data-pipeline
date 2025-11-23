import subprocess
from pathlib import Path
ML_DIR = Path(__file__).resolve().parents[1]


def test_train_rnn_more_mlflow(tmp_path):
    sample = ML_DIR / 'output' / 'MNQ_20251111_1s_w60s_h1.npz'
    if not sample.exists():
        subprocess.run(['python', 'preprocess.py', '--inputs', str(ML_DIR / 'output' / 'MNQ_20251111_1s.parquet'), '--window', '60', '--stride', '1', '--features', 'open,high,low,close,volume,gex_zero,nq_spot', '--label-source', 'nq_spot'], check=True, cwd=str(ML_DIR))
    out = ML_DIR / 'models' / 'lstm_long_mlflow.pt'
    if out.exists():
        out.unlink()
    subprocess.run(['python', 'train_rnn_more.py', '--input', str(sample), '--epochs', '2', '--batch', '256', '--dropout', '0.2', '--patience', '1', '--mlflow', '--out', str(out)], check=True, cwd=str(ML_DIR))
    assert out.exists()
