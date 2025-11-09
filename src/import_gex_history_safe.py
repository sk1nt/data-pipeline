from __future__ import annotations

import sys
from pathlib import Path

# Add the parent directory to sys.path for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import uuid
import logging
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import requests
from src.import_job_store import ImportJobStore
from pathlib import Path
import hashlib

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


def safe_import(file_path: Path | str, duckdb_path: Path | str = Path("data/gex_data.db"), publish: bool = True, history_db_path: Path | str = Path("data/gex_history.db")) -> dict:
    """Safely import historical GEX JSON into a staging table, validate, then publish.

    If `publish` is False the staging table is left in place for inspection.
    Returns a dict with job_id, staging_table and record counts.
    """
    file_path = Path(file_path)
    duckdb_path = Path(duckdb_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    job_id = None
    staging_table = None

    # compute checksum and check job store for dedupe/resume
    job_store = ImportJobStore(db_path=Path(history_db_path))
    checksum = job_store.compute_checksum(file_path)
    existing = job_store.find_by_checksum(checksum)
    if existing and existing.get("status") == "completed":
        # skip re-import
        return {"job_id": existing.get("id"), "staging_table": None, "records": existing.get("records_processed"), "skipped": True}

    if existing:
        job_id = existing.get("id")
    else:
        job_id = job_store.create_job(None, checksum, None)

    staging_table = f"staging_strikes_{job_id}"

    LOG.info("Starting safe import job=%s file=%s db=%s", job_id, file_path, duckdb_path)

    job_store.mark_started(job_id)

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
            try:
                con.begin()
                # Ensure target table exists; create with same schema as staging if not present
                con.execute(f"CREATE TABLE IF NOT EXISTS strikes AS SELECT * FROM {staging_table} LIMIT 0")
                con.execute(f"INSERT INTO strikes SELECT * FROM {staging_table}")
                con.commit()

                # Determine canonical parquet path using earliest timestamp in the batch
                try:
                    min_ts = con.execute(f"SELECT MIN(timestamp) FROM {staging_table}").fetchone()[0]
                except Exception:
                    min_ts = None

                # Fallback to current time if we couldn't determine timestamp
                from datetime import datetime

                if min_ts is None:
                    now = datetime.utcnow()
                    year = now.year
                    month = now.month
                else:
                    try:
                        # assume epoch seconds
                        dt = datetime.utcfromtimestamp(float(min_ts))
                    except Exception:
                        # if it's already a date-like string, parse via pandas
                        try:
                            import pandas as _pd

                            dt = _pd.to_datetime(min_ts).to_pydatetime()
                        except Exception:
                            dt = datetime.utcnow()
                    year = dt.year
                    month = dt.month

                # Try to infer ticker and endpoint from filename or job metadata
                ticker = None
                endpoint = None
                try:
                    if existing:
                        ticker = existing.get("ticker")
                        endpoint = existing.get("endpoint")
                except Exception:
                    pass

                if not ticker or not endpoint:
                    # filename pattern: <ticker>_<endpoint>_history_<id>.json
                    stem = file_path.stem
                    if "_history_" in stem:
                        prefix = stem.split("_history_")[0]
                        if "_" in prefix:
                            ticker, endpoint = prefix.rsplit("_", 1)
                        else:
                            ticker = prefix
                            endpoint = "gex_zero"
                    else:
                        ticker = ticker or "UNKNOWN"
                        endpoint = endpoint or "gex_zero"

                dest_dir = Path(f"data/parquet/gex/year={year}/month={month:02d}/{ticker}/{endpoint}")
                dest_dir.mkdir(parents=True, exist_ok=True)

                target = dest_dir / "strikes.parquet"
                # If a file already exists for this slot, avoid overwriting: write with job_id suffix
                if target.exists():
                    target = dest_dir / f"strikes_{job_id}.parquet"

                # Use DuckDB to write Parquet for schema stability
                con.execute(f"COPY (SELECT * FROM {staging_table}) TO '{str(target)}' (FORMAT PARQUET)")

                # Mark job completed only after successful Parquet write
                job_store.mark_completed(job_id, count)
            except Exception as e:
                try:
                    con.rollback()
                except Exception:
                    pass
                job_store.mark_failed(job_id, str(e))
                raise

        return result
    finally:
        con.close()


def process_latest_job(duckdb_path: Path | str = Path("data/gex_data.db"), history_db_path: Path | str = Path("data/gex_history.db")):
    """Fetch the latest pending job from the database, download the file, and import it."""
    job_store = ImportJobStore(db_path=history_db_path)
    con = job_store._connect()

    try:
        # Fetch the latest pending job
        job = con.execute("SELECT id, url, ticker FROM import_jobs WHERE status = 'pending' ORDER BY created_at LIMIT 1").fetchone()
        if not job:
            LOG.info("No pending jobs found.")
            return

        job_id, url, ticker = job
        LOG.info("Processing job %s for ticker %s from URL %s", job_id, ticker, url)

        # Mark job as started
        job_store.mark_started(job_id)

        # Download and import
        staged_file = download_to_staging(url, ticker)
        result = safe_import(staged_file, duckdb_path=duckdb_path)

        LOG.info("Job %s completed: %d records imported.", job_id, result.get("records", 0))
    except Exception as e:
        LOG.error("Failed to process job: %s", e)
        job_store.mark_failed(job_id, str(e))
    finally:
        con.close()


if __name__ == "__main__":
    process_latest_job()
