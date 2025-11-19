import subprocess
from pathlib import Path

def test_compare_models_smoke(tmp_path):
    sample = Path('ml/output/MNQ_20251111_1s_w60s_h1.npz')
    if not sample.exists():
        subprocess.run(['python', 'preprocess.py', '--inputs', 'output/MNQ_20251111_1s.parquet', '--window', '60', '--stride', '1'], check=True, cwd='ml')
    out_dir = Path('experiments')
    before = list(out_dir.glob('compare_*'))
    subprocess.run(['python', 'compare_models.py', '--input', str(sample), '--epochs', '1', '--batch', '256', '--models', 'lstm,cnn'], check=True, cwd='ml')
    after = list(out_dir.glob('compare_*'))
    assert len(after) > len(before)
    # ensure metrics exist in the latest folder
    latest = sorted(after)[-1]
    assert (latest / 'metrics.json').exists()
