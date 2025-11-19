#!/usr/bin/env bash
set -euo pipefail

SYMBOL=${1:-MNQ}
DATE=${2:-2025-11-11}
FILE_DATE=$(echo "$DATE" | sed 's/-//g')

python ml/extract.py --symbol "$SYMBOL" --date "$DATE"
INPUT=ml/data/${SYMBOL}_${FILE_DATE}_1s.parquet
python ml/preprocess.py --input "$INPUT" --window 60 --stride 1
NPZ=ml/output/${SYMBOL}_${FILE_DATE}_1s_w60s_h1.npz
python ml/train_lightgbm.py --input "$NPZ" --out ml/models/${SYMBOL}_lightgbm.pkl --use-gpu || python ml/train_lightgbm.py --input "$NPZ" --out ml/models/${SYMBOL}_lightgbm_cpu.pkl
python ml/train_lstm.py --input "$NPZ" --out ml/models/${SYMBOL}_lstm.pt --epochs 3 --batch 128

echo "Done. Models saved under ml/models/"