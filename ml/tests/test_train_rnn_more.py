import subprocess
from pathlib import Path


def test_train_rnn_more_smoke(tmp_path):
    # assumes preprocess has been run; fallback to sample file if not
    sample = Path('ml/output/MNQ_20251111_1s_w60s_h1.npz')
    if not sample.exists():
        # run preprocess for sample date with gex and nq_spot labels
            subprocess.run(['python', 'preprocess.py', '--inputs', 'output/MNQ_20251111_1s.parquet', '--window', '60', '--stride', '1', '--features', 'open,high,low,close,volume,gex_zero,nq_spot', '--label-source', 'nq_spot'], check=True, cwd='ml')
    out = Path('ml/models/lstm_long.pt')
    if out.exists():
        out.unlink()
    subprocess.run(['python', 'train_rnn_more.py', '--input', str(sample.relative_to('ml')), '--epochs', '2', '--batch', '256'], check=True, cwd='ml')
    assert out.exists()
