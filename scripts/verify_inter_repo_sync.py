#!/usr/bin/env python3
"""Compare contract files with data-trading sibling clone."""

from __future__ import annotations

import filecmp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIBLING = Path(
    os.getenv(
        "DATA_TRADING_ROOT",
        ROOT.parent / "data-trading",
    )
)

PAIRS = (
    "contracts/CONTRACT_VERSION",
    "contracts/redis_channels.py",
    "contracts/sweep_alert.py",
)


def main() -> int:
    if not SIBLING.is_dir():
        print(f"SKIP: sibling repo not found at {SIBLING}")
        print("  Set DATA_TRADING_ROOT or clone data-trading next to data-pipeline")
        return 0

    failed = False
    for rel in PAIRS:
        a = ROOT / rel
        b = SIBLING / rel
        if not b.exists():
            print(f"FAIL: missing in sibling: {rel}")
            failed = True
            continue
        if not filecmp.cmp(a, b, shallow=False):
            print(f"FAIL: drift: {rel}")
            failed = True
        else:
            print(f"OK: {rel}")

    if failed:
        print("\nFix: sync contracts/ and bump CONTRACT_VERSION in both repos")
        return 1

    print("verify_inter_repo_sync: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())