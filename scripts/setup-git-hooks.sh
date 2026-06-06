#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
chmod +x .githooks/pre-push
git config core.hooksPath .githooks
echo "Git hooks: pre-push runs scripts/verify_contracts.py"