import subprocess
from pathlib import Path
ML_DIR = Path(__file__).resolve().parents[1]

NPZ = Path('output/MNQ_20251111_1s_w60s_h1.npz')


def test_tune_smoke(tmp_path):
    cmd = ['python', 'tune.py', '--input', str(ML_DIR / NPZ), '--trials', '2', '--max-epochs', '1', '--sample', '2000']
    subprocess.run(cmd, check=True, cwd=str(ML_DIR))
    # ensure experiment directory created
    import glob
    dirs = glob.glob(str(ML_DIR / 'experiments' / 'lstm_tune_*'))
    assert len(dirs) > 0
