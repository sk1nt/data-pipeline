import subprocess
from pathlib import Path
ML_DIR = Path(__file__).resolve().parents[1]

OUT_DIR = ML_DIR / 'output'


def test_extract_and_preprocess(tmp_path):
    # Run extraction for a day known to exist in reports (use MNQ 20251111)
    symbol = 'MNQ'
    date = '2025-11-11'
    subprocess.run(['python', 'extract.py', '--symbol', symbol, '--date', date], check=True, cwd=str(ML_DIR))
    # check output exists
    file_date = '20251111'
    out_path = OUT_DIR / f"{symbol}_{file_date}_1s.parquet"
    assert out_path.exists()
    # run preprocess
    subprocess.run(['python', 'preprocess.py', '--input', str(out_path), '--window', '60', '--stride', '1'], check=True, cwd=str(ML_DIR))
    # check windows exist
    import glob
    npz_files = glob.glob(str(ML_DIR / 'output' / '*_w60s_h1.npz'))
    assert len(npz_files) > 0
