#!/usr/bin/env python3
"""Migration script: ensure gex_snapshots table has `strikes` column.

Usage: python scripts/migrate_add_strikes_to_gex_snapshots.py
"""

import duckdb
from pathlib import Path

db_path = Path("data/gex_data.db")
if not db_path.exists():
    print("gex_data.db not found; nothing to do")
    exit(1)

con = duckdb.connect(str(db_path))
try:
    print("Adding column `strikes` to gex_snapshots if not exists...")
    try:
        con.execute(
            "ALTER TABLE gex_snapshots ADD COLUMN IF NOT EXISTS strikes VARCHAR"
        )
        print("Done.")
    except Exception as e:
        print("Alter statement failed:", e)
finally:
    con.close()
