import subprocess
from pathlib import Path
ML_DIR = Path(__file__).resolve().parents[1]


def test_compare_models_smoke(tmp_path):
    sample = ML_DIR / 'output' / 'MNQ_20251111_1s_w60s_h1.npz'
    if not sample.exists():
        subprocess.run(['python', 'preprocess.py', '--inputs', str(ML_DIR / 'output' / 'MNQ_20251111_1s.parquet'), '--window', '60', '--stride', '1'], check=True, cwd=str(ML_DIR))
    out_dir = ML_DIR / 'experiments'
    before = list(out_dir.glob('compare_*'))
    subprocess.run(['python', 'compare_models.py', '--input', str(sample), '--epochs', '1', '--batch', '256', '--models', 'lstm,cnn'], check=True, cwd=str(ML_DIR))
    after = list(out_dir.glob('compare_*'))
    assert len(after) > len(before)
    # ensure metrics exist in the latest folder
    latest = sorted(after)[-1]
    assert (latest / 'metrics.json').exists()
