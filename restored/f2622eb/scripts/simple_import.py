#!/usr/bin/env python3
"""Simple, opinionated importer CLI for a single URL.

Usage: python scripts/simple_import.py <url> [--ticker TICKER]

This script performs a single import synchronously and prints clear
progress to stdout so you can see what's happening. It attempts to set
the process title to `gex-import-<jobid>` when `setproctitle` is available.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from setproctitle import setproctitle
except Exception:
    setproctitle = None

from src.import_gex_history_safe import download_to_staging, safe_import


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def main():
    p = argparse.ArgumentParser(description="Simple import CLI for GEX history JSON -> Parquet")
    p.add_argument("url", help="HTTP or file:// URL to the JSON")
    p.add_argument("--ticker", help="Ticker name to use when staging (optional)")
    args = p.parse_args()

    url = args.url
    ticker = args.ticker or "UNKNOWN"

    print(f"[{now()}] STARTING import for url={url}")

    # Download to staging
    try:
        staged = download_to_staging(url, ticker)
        print(f"[{now()}] DOWNLOADED -> {staged}")
    except Exception as e:
        print(f"[{now()}] ERROR downloading: {e}")
        sys.exit(2)

    # Optionally set a human-readable process title
    jobid = staged.stem
    proc_name = f"gex-import-{jobid}"
    try:
        if setproctitle:
            setproctitle(proc_name)
    except Exception:
        pass

    print(f"[{now()}] IMPORTING (job={jobid})")
    try:
        result = safe_import(staged, publish=True)
        records = result.get("records", 0)
        print(f"[{now()}] IMPORT COMPLETE: job={result.get('job_id')} records={records}")
        sys.exit(0)
    except Exception as e:
        print(f"[{now()}] IMPORT FAILED: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
