import subprocess
from pathlib import Path


def test_train_lstm_early_stop(tmp_path):
    sample = Path('ml/output/MNQ_20251111_1s_w60s_h1.npz')
    if not sample.exists():
        subprocess.run(['python', 'preprocess.py', '--inputs', 'output/MNQ_20251111_1s.parquet', '--window', '60', '--stride', '1', '--features', 'open,high,low,close,volume,gex_zero,nq_spot','--label-source','nq_spot'], check=True, cwd='ml')
    out = Path('ml/models/lstm_early.pt')
    if out.exists():
        out.unlink()
    subprocess.run(['python', 'train_lstm.py', '--input', str(sample.relative_to('ml')), '--epochs', '5', '--batch', '256', '--dropout', '0.1', '--patience', '2', '--out', str(out.relative_to('ml'))], check=True, cwd='ml')
    assert out.exists()
