import subprocess
from pathlib import Path


def test_lightgbm_mlflow_smoke(tmp_path):
    sample = Path('output/MNQ_20251111_1s_w60s_h1.npz')
    if not sample.exists():
        subprocess.run(['python', 'preprocess.py', '--inputs', 'output/MNQ_20251111_1s.parquet', '--window', '60', '--stride', '1', '--features', 'open,high,low,close,volume,gex_zero,nq_spot','--label-source','nq_spot'], check=True, cwd='ml')
    out = Path('models/lgb_mlflow_test.pkl')
    if out.exists():
        out.unlink()
    subprocess.run(['bash', '-lc', 'export MLFLOW_TRACKING_URI=http://127.0.0.1:5000 && python train_lightgbm.py --input ' + str(sample) + ' --out ' + str(out) + ' --mlflow'], shell=True, check=True, cwd='ml')
    assert out.exists()
