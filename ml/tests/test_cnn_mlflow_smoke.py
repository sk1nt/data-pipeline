import subprocess
from pathlib import Path


def test_cnn_mlflow_smoke(tmp_path):
    sample = Path('ml/output/MNQ_20251111_1s_w60s_h1.npz')
    if not sample.exists():
        subprocess.run(['python', 'preprocess.py', '--inputs', 'output/MNQ_20251111_1s.parquet', '--window', '60', '--stride', '1', '--features', 'open,high,low,close,volume,gex_zero,nq_spot','--label-source','nq_spot'], check=True, cwd='ml')
    out = Path('ml/models/cnn_mlflow_test.pt')
    if out.exists():
        out.unlink()
    # pass relative input/out args to subprocess (cwd='ml')
        cmd = 'export MLFLOW_TRACKING_URI=http://127.0.0.1:5000 && python train_cnn.py --input {} --epochs 1 --batch 256 --mlflow --out {}'.format(sample.relative_to('ml'), out.relative_to('ml'))
        subprocess.run(cmd, shell=True, check=True, cwd='ml')
    assert out.exists()
