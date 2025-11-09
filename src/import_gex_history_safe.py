from __future__ import annotations

import json
import uuid
import logging
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import requests

LOG = logging.getLogger("import_gex_history_safe")


def download_to_staging(url: str, ticker: str, endpoint: str = "gex_zero") -> Path:
    dest_dir = Path("data/source/gexbot")
    dest_dir.mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex[:8]
    tmp_name = f"{ticker}_{endpoint}_history_{job_id}.json.tmp"
    final_name = f"{ticker}_{endpoint}_history_{job_id}.json"
    tmp_path = dest_dir / tmp_name
    final_path = dest_dir / final_name

    LOG.info("Downloading %s to %s", url, tmp_path)
    # Support file:// local paths as well as HTTP URLs
    if url.startswith("file://"):
        src = Path(url[7:])
        if not src.exists():
            raise FileNotFoundError(src)
        tmp_path.write_bytes(src.read_bytes())
    else:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    # Validate it's parseable JSON (light check)
    try:
        with tmp_path.open("r", encoding="utf-8") as f:
            _ = json.load(f)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    tmp_path.rename(final_path)
    LOG.info("Downloaded and moved to %s", final_path)
    return final_path


def safe_import(file_path: Path | str, duckdb_path: Path | str = Path("data/gex_data.db"), publish: bool = True) -> dict:
    """Safely import historical GEX JSON into a staging table, validate, then publish.

    If `publish` is False the staging table is left in place for inspection.
    Returns a dict with job_id, staging_table and record counts.
    """
    file_path = Path(file_path)
    duckdb_path = Path(duckdb_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    job_id = uuid.uuid4().hex[:8]
    staging_table = f"staging_strikes_{job_id}"

    LOG.info("Starting safe import job=%s file=%s db=%s", job_id, file_path, duckdb_path)

    # Load JSON into pandas (assume array of objects)
    df = pd.read_json(file_path, orient="records")

    # Basic validations
    required_cols = {"timestamp", "ticker", "spot"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    con = duckdb.connect(str(duckdb_path))
    try:
        # Register and create staging table from DataFrame
        con.register("batch_df", df)
        con.execute(f"CREATE TABLE {staging_table} AS SELECT * FROM batch_df")

        # Run integrity checks in SQL (example checks)
        res = con.execute(f"SELECT COUNT(*) FROM {staging_table}").fetchone()
        count = int(res[0])
        if count == 0:
            raise ValueError("Staging table contains zero records")

        # Check timestamps are sensible (non-null)
        null_ts = con.execute(f"SELECT COUNT(*) FROM {staging_table} WHERE timestamp IS NULL").fetchone()[0]
        if null_ts > 0:
            raise ValueError("Staging table contains null timestamps")

        result = {"job_id": job_id, "staging_table": staging_table, "records": count}

        if publish:
            LOG.info("Publishing staging table %s into `strikes`", staging_table)
            # Insert into final table in a transaction to avoid partial writes
            con.begin()
            # Ensure target table exists; create if not
            con.execute("CREATE TABLE IF NOT EXISTS strikes AS SELECT * FROM (SELECT * FROM (SELECT NULL AS timestamp) LIMIT 0)")
            con.execute(f"INSERT INTO strikes SELECT * FROM {staging_table}")
            con.commit()

        return result
    finally:
        con.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Safe import historical GEX JSON into DuckDB staging and publish")
    p.add_argument("url_or_path", help="HTTP URL or file:// path to JSON file")
    p.add_argument("--ticker", required=False, help="ticker for file naming (optional)")
    p.add_argument("--duckdb", default="data/gex_data.db", help="path to duckdb database")
    p.add_argument("--no-publish", dest="publish", action="store_false", help="Do not publish, only stage")
    args = p.parse_args()

    path = args.url_or_path
    if path.startswith("http://") or path.startswith("https://") or path.startswith("file://"):
        staged = download_to_staging(path, args.ticker or "UNKNOWN")
    else:
        staged = Path(path)

    out = safe_import(staged, duckdb_path=args.duckdb, publish=args.publish)
    print(out)
