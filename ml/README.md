# ML Workspace

This folder contains data extraction, preprocessing, and example training scripts for prototyping models using the repository's data.

All scripts are designed to read from the canonical data files under `data/` and produce outputs under `ml/`.

Commands:
In this folder you'll find ML models, backtest helpers, and scripts.

**Practical Tips:**
- See `docs/TRADING_SYSTEMS_TIPS.md` for short, practical guidelines derived from live trading systems. The doc covers bar-aggregation (volume/time/dollar bars), overfitting risks and walk-forward validation, transaction-cost assumptions, latency effects, and recommendations to prefer low-variance models as first steps.

Commands:
- Extract 1s aggregated tick features and join GEX snapshots:
  `python ml/extract.py --symbol MNQ --date 2025-11-11 --gex-db data/gex_data.db --gex-ticker NQ_NDX`

- Enrich the MNQ ticks for the last full week of October 2025 with NQ_NDX snapshots:
  `python ml/enrich_october_week.py` (writes per-day parquet to `ml/output/` and a summary JSON).

- Preprocess and create sliding-window datasets (CSV/Parquet):
  `python ml/preprocess.py --inputs ml/data/MNQ_20251111_1s.parquet --window 60 --stride 1 --features open,high,low,close,volume,gex_zero,nq_spot --label-source nq_spot`

- Train LightGBM baseline (fast):
  `python ml/train_lightgbm.py --input ml/data/MNQ_20251111_1s.parquet --output ml/models/lightgbm_mnq.bst`

 - Train LSTM baseline (requires PyTorch):
  `python ml/train_lstm.py --input ml/output/MNQ_20251111_1s_w60s_h1.npz --out ml/models/lstm_mnq.pt`
  
  Use 'gex_zero' and 'nq_spot' as features and use `--label-source nq_spot` during preprocessing to use NQ index movement derived from GEX snapshots as a proxy label.
 - Running multi-day experiments: pass multiple files to `--inputs` (comma-separated or glob) when calling `ml/preprocess.py`.
  - By default the LSTM script uses CUDA if available; make sure PyTorch is GPU-enabled (`pip install torch --extra-index-url https://download.pytorch.org/whl/cu121`).
  - LightGBM can use GPU if installed with GPU support and `--use-gpu` is passed to `ml/train_lightgbm.py`.

Notes:
- All outputs (datasets, models, logs) are stored under `ml/`.
- Scripts do not modify files outside the `data/` directory and `ml/`.

MLflow server
 - Use the included helper to run a local MLflow server that points at `ml/mlflow.db` and stores artifacts in `ml/mlruns`:
   `./scripts/start_mlflow_server.sh 5000` (or choose a port). Tests that set `MLFLOW_TRACKING_URI` to `http://127.0.0.1:5000` assume a server is running.
