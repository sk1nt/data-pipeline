#!/usr/bin/env python
"""CLI wrapper to export MNQ 1s OHLCV + GEX enriched Parquet files."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.enriched_exporter import main  # noqa: E402

if __name__ == "__main__":
    main()
