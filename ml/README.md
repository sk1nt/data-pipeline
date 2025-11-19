# ML Workspace

This folder contains data extraction, preprocessing, and example training scripts for prototyping models using the repository's data.

All scripts are designed to read from the canonical data files under `data/` and produce outputs under `ml/`.

Commands:

- Extract 1s aggregated tick features and join GEX snapshots:
  `python ml/extract.py --symbol MNQ --date 2025-11-11`

- Preprocess and create sliding-window datasets (CSV/Parquet):
  `python ml/preprocess.py --symbol MNQ --date 2025-11-11 --window 60 --stride 1`

- Train LightGBM baseline (fast):
  `python ml/train_lightgbm.py --input ml/data/MNQ_20251111_1s.parquet --output ml/models/lightgbm_mnq.bst`

- Train LSTM baseline (requires PyTorch):
  `python ml/train_lstm.py --input ml/data/MNQ_20251111_windows.npz --output ml/models/lstm_mnq.pt`
  - By default the LSTM script uses CUDA if available; make sure PyTorch is GPU-enabled (`pip install torch --extra-index-url https://download.pytorch.org/whl/cu121`).
  - LightGBM can use GPU if installed with GPU support and `--use-gpu` is passed to `ml/train_lightgbm.py`.

Notes:
- All outputs (datasets, models, logs) are stored under `ml/`.
- Scripts do not modify files outside the `data/` directory and `ml/`.
