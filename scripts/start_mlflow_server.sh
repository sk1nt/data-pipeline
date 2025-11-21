#!/usr/bin/env bash
# Starts an MLflow server pointing at the repo-local sqlite DB and mlruns
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
resolve_path() {
    python - <<'PY' "$1"
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser()
print(path.resolve())
PY
}
DB_PATH=$(resolve_path "$ROOT/ml/mlflow.db")
ARTIFACT_ROOT=$(resolve_path "$ROOT/ml/mlruns")
PORT=${1:-5000}

# Ensure DB file and folders exist
mkdir -p "$(dirname "$DB_PATH")"
touch "$DB_PATH"
mkdir -p "$ARTIFACT_ROOT"

echo "Starting MLflow server on http://127.0.0.1:$PORT"
echo "Backend store: sqlite:///$DB_PATH"
echo "Default artifact root: file://$ARTIFACT_ROOT"
mlflow server --backend-store-uri "sqlite:///$DB_PATH" --default-artifact-root "file://$ARTIFACT_ROOT" --host 127.0.0.1 --port "$PORT"
