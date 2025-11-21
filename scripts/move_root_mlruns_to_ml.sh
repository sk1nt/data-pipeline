#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC="$ROOT/mlruns"
DEST="$ROOT/ml/mlruns"
TIMESTAMP=$(date +%Y%m%dT%H%M%S)

if [ ! -d "$SRC" ]; then
  echo "No root-level mlruns directory found at $SRC. Nothing to move."
  exit 0
fi

mkdir -p "$DEST"

shopt -s dotglob
for item in "$SRC"/*; do
  if [ ! -e "$item" ]; then
    continue
  fi
  name=$(basename "$item")
  if [ -e "$DEST/$name" ]; then
    echo "Destination $DEST/$name already exists; moving to ${name}_moved_$TIMESTAMP"
    mv "$item" "$DEST/${name}_moved_$TIMESTAMP"
  else
    echo "Moving $item -> $DEST/$name"
    mv "$item" "$DEST/"
  fi
done

echo "Cleaning up root mlruns dir if empty"
if [ -z "$(ls -A "$SRC")" ]; then
  rmdir "$SRC"
  echo "Removed empty $SRC"
else
  echo "$SRC not empty after move; leaving as-is"
fi

echo "Move complete. New mlruns location: $DEST"
