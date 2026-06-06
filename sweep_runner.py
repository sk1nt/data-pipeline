#!/usr/bin/env python3
"""MOVED — sweep stack runs from data-trading on the Sierra Chart / WSL host."""

from __future__ import annotations

import os
import sys

ROOT = os.path.expanduser(os.getenv("DATA_TRADING_ROOT", "~/projects/data-trading"))

print(
    "\n"
    "sweep_runner.py no longer runs from data-pipeline.\n"
    f"  Use: cd {ROOT}\n"
    "       python sweep_runner.py [--dry-run]\n"
    "\n"
    "  See: docs/SWEEP_MOVED_TO_DATA_TRADING.md\n"
    "       ~/projects/data-trading/docs/REPO_OWNERSHIP.md\n",
    file=sys.stderr,
)
sys.exit(2)