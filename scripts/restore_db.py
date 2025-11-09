#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import shutil
import argparse


def restore(backup: Path, target: Path) -> None:
    if not backup.exists():
        raise FileNotFoundError(backup)
    shutil.copy2(backup, target)


def main() -> None:
    p = argparse.ArgumentParser(description="Restore DB backup to target path")
    p.add_argument("backup", help="backup file path")
    p.add_argument("target", help="target path to restore to")
    args = p.parse_args()
    restore(Path(args.backup), Path(args.target))
    print(f"Restored {args.backup} to {args.target}")


if __name__ == "__main__":
    main()
