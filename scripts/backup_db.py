#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from datetime import datetime


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(db_paths: list[Path], out_dir: Path) -> list[tuple[Path, Path, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    for p in db_paths:
        if not p.exists():
            raise FileNotFoundError(p)
        dest = out_dir / f"{p.name}.{ts}.bak"
        shutil.copy2(p, dest)
        ch = checksum(dest)
        results.append((p, dest, ch))
    return results


def restore(backup: Path, target: Path) -> None:
    if not backup.exists():
        raise FileNotFoundError(backup)
    shutil.copy2(backup, target)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Snapshot or restore DuckDB DB files before imports"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    snap = sub.add_parser("snapshot")
    snap.add_argument("db_files", nargs="+", help="paths to DB files to snapshot")
    snap.add_argument(
        "--out-dir", default="data/backups", help="backup output directory"
    )

    rst = sub.add_parser("restore")
    rst.add_argument("backup_file", help="backup file to restore from")
    rst.add_argument("target", help="target DB file path to restore to")

    args = p.parse_args()

    if args.cmd == "snapshot":
        dbs = [Path(x) for x in args.db_files]
        out = Path(args.out_dir)
        for orig, dest, ch in snapshot(dbs, out):
            print(f"{orig} -> {dest} (sha256={ch})")

    elif args.cmd == "restore":
        restore(Path(args.backup_file), Path(args.target))
        print(f"Restored {args.backup_file} to {args.target}")


if __name__ == "__main__":
    main()
