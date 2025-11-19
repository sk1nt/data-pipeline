import subprocess
from pathlib import Path

OUT_DIR = Path('ml/output')


def test_extract_and_preprocess_two_days(tmp_path):
    symbol = 'MNQ'
    dates = '2025-11-11,2025-11-12'
    subprocess.run(['python','extract.py','--symbol',symbol,'--date',dates,'--gex-db','data/gex_data.db','--gex-ticker','NQ_NDX'], check=True, cwd='ml')
    file_date = '20251111'
    out1 = OUT_DIR / f"{symbol}_{file_date}_1s.parquet"
    file_date2 = '20251112'
    out2 = OUT_DIR / f"{symbol}_{file_date2}_1s.parquet"
    assert out1.exists() and out2.exists()
    # Preprocess both as combined
    subprocess.run(['python','preprocess.py','--inputs',str(out1)+','+str(out2),'--window','60','--stride','1','--features','open,high,low,close,volume,gex_zero,nq_spot','--label-source','nq_spot'], check=True, cwd='ml')
    import glob
    npz_files = glob.glob('ml/output/*_w60s_h1.npz')
    assert len(npz_files) > 0
    import numpy as np
    data = np.load(npz_files[0])
    assert 'X' in data and 'y' in data
    assert data['X'].shape[2] >= 6  # expects gex and nq_spot included
