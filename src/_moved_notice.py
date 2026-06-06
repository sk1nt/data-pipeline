"""Modules that moved to data-trading raise via moved_import()."""

from __future__ import annotations

import os

DATA_TRADING_ROOT = os.getenv(
    "DATA_TRADING_ROOT",
    os.path.expanduser("~/projects/data-trading"),
)


def moved_import(module_label: str, relative_path: str) -> None:
    raise ImportError(
        f"{module_label} moved to the data-trading repo.\n"
        f"  Path: {DATA_TRADING_ROOT}/{relative_path}\n"
        f"  Run:  cd {DATA_TRADING_ROOT} && python sweep_runner.py\n"
        f"  Doc:  docs/SWEEP_MOVED_TO_DATA_TRADING.md"
    )