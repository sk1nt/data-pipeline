import subprocess
import os
from pathlib import Path

DATA_NPZ = Path('output/MNQ_20251111_1s_w60s_h1.npz')

MODELS = [
    ['python','train_cnn.py','--input', str(DATA_NPZ),'--epochs','1','--batch','256'],
    ['python','train_tcn.py','--input', str(DATA_NPZ),'--epochs','1','--batch','256'],
    ['python','train_transformer.py','--input', str(DATA_NPZ),'--epochs','1','--batch','256'],
    ['python','train_lstm.py','--input', str(DATA_NPZ),'--epochs','1','--batch','256'],
    ['python','train_lightgbm.py','--input', str(DATA_NPZ),'--out','models/smoke_lgb.pkl'],
    ['python','train_xgboost.py','--input', str(DATA_NPZ),'--out','models/smoke_xgb.json']
]


def test_models_run_smoke(tmp_path):
    # run each script; expect them to complete without exception
    for cmd in MODELS:
        # skip xgboost if not installed
        try:
            subprocess.run(cmd, check=True, cwd='ml')
        except Exception as e:
            print('Model failed:', cmd, e)
            assert False, 'Smoke test failed for model: ' + ' '.join(cmd)
