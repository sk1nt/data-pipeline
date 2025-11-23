import subprocess
from pathlib import Path
ML_DIR = Path(__file__).resolve().parents[1]


def test_lightgbm_mlflow_smoke(tmp_path):
    sample = ML_DIR / 'output' / 'MNQ_20251111_1s_w60s_h1.npz'
    if not sample.exists():
        subprocess.run(['python', 'preprocess.py', '--inputs', str(ML_DIR / 'output' / 'MNQ_20251111_1s.parquet'), '--window', '60', '--stride', '1', '--features', 'open,high,low,close,volume,gex_zero,nq_spot', '--label-source', 'nq_spot'], check=True, cwd=str(ML_DIR))
    out = ML_DIR / 'models' / 'lgb_mlflow_test.pkl'
    if out.exists():
        out.unlink()
    cmd = ['python', 'train_lightgbm.py', '--input', str(sample), '--out', str(out), '--mlflow']
    subprocess.run(cmd, check=True, cwd=str(ML_DIR))
    assert out.exists()
