import os
import sys
import subprocess
from pathlib import Path


def run(cmd, env=None, cwd=None):
    print('Running:', ' '.join(cmd))
    res = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env, cwd=cwd)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise SystemExit(res.returncode)
    return res


def test_volume_bar_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    ml_dir = repo_root / 'ml'
    env = os.environ.copy()
    # Ensure MLflow writes into ml/ during the test
    env['MLFLOW_TRACKING_URI'] = f"sqlite:///{ml_dir / 'mlflow.db'}"
    env['MLFLOW_ARTIFACT_URI'] = f"file://{ml_dir / 'mlruns'}"
    # Clean up any root-level mlruns leftover from previous runs to ensure test isolation
    import shutil
    root_mlruns = repo_root / 'mlruns'
    if root_mlruns.exists():
        shutil.rmtree(root_mlruns)

    # Use a compact date sample (Oct 21) and a small volume-size to keep execution fast
    date = '20251021'
    # Prepare a small synthetic tick parquet if it doesn't exist to avoid relying on large data
    tick_dir = ml_dir / 'data' / 'tick' / 'MNQ'
    tick_dir.mkdir(parents=True, exist_ok=True)
    tick_parquet = tick_dir / f"{date}.parquet"
    if not tick_parquet.exists():
        import pandas as pd
        import numpy as np
        ts = pd.date_range('2025-10-21 09:30:00', periods=500, freq='S')
        prices = 2500000 + np.cumsum(np.random.randint(-10, 10, size=len(ts)))
        vol = np.random.randint(1, 5, size=len(ts))
        df_ticks = pd.DataFrame({'timestamp': ts, 'price': prices, 'volume': vol, 'ts_ms': (ts.astype('int64') // 1_000_000)})
        df_ticks.to_parquet(tick_parquet, index=False)

    # Ensure repo-root data/tick path contains a copy, since CLI extract uses repo-root `data/tick`
    repo_tick_dir = repo_root / 'data' / 'tick' / 'MNQ'
    repo_tick_dir.mkdir(parents=True, exist_ok=True)
    repo_tick_parquet = repo_tick_dir / f"{date}.parquet"
    if not repo_tick_parquet.exists():
        # copy from ml location if possible
        import shutil
        shutil.copy2(tick_parquet, repo_tick_parquet)

    # Run extract as a script from within `ml/` so outputs are written to `ml/output`
    extract_cmd = [sys.executable, 'extract.py', '--symbol', 'MNQ', '--date', date, '--bar-type', 'volume', '--bar-size', '100']
    run(extract_cmd, env=env, cwd=str(ml_dir))

    parquet = ml_dir / 'output' / f"MNQ_{date}_volume100.parquet"
    assert parquet.exists(), f'Expected parquet file {parquet} to exist'

    preprocess_cmd = [sys.executable, 'preprocess.py', '--inputs', str(parquet), '--window', '60', '--stride', '10', '--horizon', '1']
    run(preprocess_cmd, env=env, cwd=str(ml_dir))

    npz = ml_dir / 'output' / f"MNQ_{date}_volume100_w60s_h1.npz"
    assert npz.exists(), f'Expected dataset {npz} to exist'

    # Train a small LightGBM model on the sample dataset to ensure feature shape matches and keep the run fast
    import joblib
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        LGBMClassifier = None

    if LGBMClassifier is not None:
        data = np.load(str(npz))
        X = data['X']
        y = data['y']
        X_flat = X.reshape(X.shape[0], -1)
        y_binary = (y > 0).astype(int)
        # sample small subset for quick training
        n_sample = min(200, X_flat.shape[0])
        X_train, _, y_train, _ = train_test_split(X_flat, y_binary, train_size=n_sample, random_state=42, stratify=y_binary)
        model = LGBMClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        model_path = ml_dir / 'models' / 'lgb_smoke.pkl'
        joblib.dump(model, model_path)
        backtest_model_arg = str(Path('ml') / 'models' / 'lgb_smoke.pkl')
    else:
        # Fallback: use existing lightgbm_tuned, but it may error due to shape mismatch
        backtest_model_arg = 'ml/models/lightgbm_tuned.pkl'

    backtest_cmd = [sys.executable, '-m', 'ml.pnl_backtest', '--mlflow', '--experiment', 'ci_volume_smoke', '--model', backtest_model_arg, '--days', date, '--window', '60', '--threshold', '0.5', '--instrument', 'MNQ', '--bar-type', 'volume', '--bar-size', '100', '--sample', '50', '--stride', '10']
    run(backtest_cmd, env=env, cwd=str(repo_root))

    # Assert artifacts are created under ml/ (artifacts should be logged to mlruns under ml/)
    assert (ml_dir / 'mlruns').exists(), 'ml/mlruns must exist'
    # Verify that at least one mlrun contains the backtest summary artifact
    found = False
    for p in (ml_dir / 'mlruns').glob('**/artifacts/ml_backtest_summary.csv'):
        if p.exists():
            found = True
            break
    assert found, 'Expected at least one ml_backtest_summary.csv under ml/mlruns artifacts'
