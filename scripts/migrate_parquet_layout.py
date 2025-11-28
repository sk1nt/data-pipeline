#!/usr/bin/env python3
"""Migrate Parquet layout from year=/month= dirs to YYYY/MM."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move Parquet files from year=/month= layout to YYYY/MM layout."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/parquet/gexbot"),
        help="Root directory that contains the legacy year=/month= folders.",
    )
    parser.add_argument(
        "--legacy-bucket",
        default="_legacy",
        help="Folder name to use for files that are not nested under ticker/endpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without modifying the filesystem.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove empty legacy directories after migration.",
    )
    return parser.parse_args()


def collect_moves(
    root: Path, legacy_bucket: str
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """Return (dir_moves, file_moves) from legacy layout."""
    dir_moves: List[Tuple[Path, Path]] = []
    file_moves: List[Tuple[Path, Path]] = []

    for year_dir in sorted(root.glob("year=*")):
        year = year_dir.name.split("=", 1)[1]
        for month_dir in sorted(year_dir.glob("month=*")):
            month = month_dir.name.split("=", 1)[1]

            # Nested ticker/endpoint directories
            for ticker_dir in sorted(p for p in month_dir.iterdir() if p.is_dir()):
                for endpoint_dir in sorted(
                    p for p in ticker_dir.iterdir() if p.is_dir()
                ):
                    dest = root / year / month / ticker_dir.name / endpoint_dir.name
                    dir_moves.append((endpoint_dir, dest))

            # Stray Parquet files directly under month directory
            for file_path in sorted(month_dir.glob("*.parquet")):
                dest = root / year / month / legacy_bucket / file_path.name
                file_moves.append((file_path, dest))

    return dir_moves, file_moves


def move_path(src: Path, dest: Path, dry_run: bool) -> None:
    logging.info("Move %s -> %s", src, dest)
    if dry_run:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        # Merge contents instead of clobbering
        if src.is_dir() and dest.is_dir():
            for child in src.iterdir():
                move_path(child, dest / child.name, dry_run=False)
            if not any(src.iterdir()):
                src.rmdir()
            return
        raise FileExistsError(f"Destination already exists: {dest}")
    shutil.move(str(src), str(dest))


def cleanup_legacy_dirs(root: Path, dry_run: bool) -> None:
    """Remove empty legacy dirs anywhere under year=/month= paths."""
    legacy_dirs = sorted(
        (
            p
            for p in root.rglob("*")
            if p.is_dir() and ("year=" in str(p) or "month=" in str(p))
        ),
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for directory in legacy_dirs:
        try:
            if any(directory.iterdir()):
                continue
        except FileNotFoundError:
            continue
        logging.info("Removing empty legacy dir %s", directory)
        if not dry_run:
            directory.rmdir()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = args.root
    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")

    dir_moves, file_moves = collect_moves(root, args.legacy_bucket)
    if not dir_moves and not file_moves:
        logging.info("No legacy directories detected under %s", root)
        if args.cleanup:
            cleanup_legacy_dirs(root, args.dry_run)
        return

    for src, dest in dir_moves:
        move_path(src, dest, args.dry_run)
    for src, dest in file_moves:
        move_path(src, dest, args.dry_run)

    if args.cleanup:
        cleanup_legacy_dirs(root, args.dry_run)

    logging.info(
        "Migration %s. Directories processed: %d, stray files processed: %d",
        "planned" if args.dry_run else "completed",
        len(dir_moves),
        len(file_moves),
    )


if __name__ == "__main__":
    main()
