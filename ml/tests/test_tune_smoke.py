import subprocess
from pathlib import Path

NPZ = Path('ml/output/MNQ_20251111_1s_w60s_h1.npz')


def test_tune_smoke(tmp_path):
    cmd = ['python','ml/tune.py','--input',str(NPZ),'--trials','2','--max-epochs','1','--sample','2000']
    subprocess.run(cmd, check=True)
    # ensure experiment directory created
    import glob
    dirs = glob.glob('ml/experiments/lstm_tune_*')
    assert len(dirs) > 0
