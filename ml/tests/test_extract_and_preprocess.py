import os
import subprocess
from pathlib import Path

OUT_DIR = Path('ml/data')


def test_extract_and_preprocess(tmp_path):
    # Run extraction for a day known to exist in reports (use MNQ 20251111)
    symbol = 'MNQ'
    date = '2025-11-11'
    subprocess.run(['python', 'ml/extract.py', '--symbol', symbol, '--date', date], check=True)
    # check output exists
    file_date = '20251111'
    out_path = OUT_DIR / f"{symbol}_{file_date}_1s.parquet"
    assert out_path.exists()
    # run preprocess
    subprocess.run(['python', 'ml/preprocess.py', '--input', str(out_path), '--window', '60', '--stride', '1'], check=True)
    # check windows exist
    import glob
    npz_files = glob.glob('ml/output/*_w60s_h1.npz')
    assert len(npz_files) > 0
