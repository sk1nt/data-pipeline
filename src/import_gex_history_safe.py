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
import traceback
from datetime import datetime

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

        # Timestamp range validation: compute min/max and ensure they are within reasonable bounds
        try:
            min_ts = con.execute(f"SELECT MIN(timestamp) FROM {staging_table}").fetchone()[0]
            max_ts = con.execute(f"SELECT MAX(timestamp) FROM {staging_table}").fetchone()[0]
        except Exception:
            min_ts = None
            max_ts = None

        def _to_year(val):
            if val is None:
                return None
            try:
                # numeric epoch seconds
                return datetime.utcfromtimestamp(float(val)).year
            except Exception:
                try:
                    return pd.to_datetime(val, errors='coerce').year
                except Exception:
                    return None

        min_year = _to_year(min_ts)
        max_year = _to_year(max_ts)

        current_year = datetime.utcnow().year
        # Acceptable range: 2000..(current_year+1)
        if min_year is not None and (min_year < 2000 or min_year > current_year + 1):
            raise ValueError(f"Min timestamp year {min_year} out of acceptable range")
        if max_year is not None and (max_year < 2000 or max_year > current_year + 1):
            raise ValueError(f"Max timestamp year {max_year} out of acceptable range")

        LOG.info("Staging timestamps range: min=%s max=%s (years %s-%s)", min_ts, max_ts, min_year, max_year)

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

                dest_dir = Path(f"data/parquet/gex/{year}/{month:02d}/{ticker}/{endpoint}")
                dest_dir.mkdir(parents=True, exist_ok=True)

                target = dest_dir / "strikes.parquet"
                # If a file already exists for this slot, avoid overwriting: write with job_id suffix
                if target.exists():
                    target = dest_dir / f"strikes_{job_id}.parquet"

                # Use DuckDB to write Parquet for schema stability
                con.execute(f"COPY (SELECT * FROM {staging_table}) TO '{str(target)}' (FORMAT PARQUET)")

                # Mark job completed only after successful Parquet write
                job_store.mark_completed(job_id, count)

                # Remove staged JSON to save space if it's under the canonical staging directory
                # Prune staged JSON files older than retention (keep for 14 days)
                try:
                    prune_staged_files(staged_root=Path("data/source/gexbot"), retention_days=14)
                except Exception:
                    LOG.exception("Failed to prune staged JSON files")

                # Update the reconciliation report after successful import
                try:
                    generate_reconciliation_report(history_db_path=history_db_path, parquet_root=Path("data/parquet/gex"))
                except Exception:
                    LOG.exception("Failed to update reconciliation report")
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


def generate_reconciliation_report(history_db_path: Path | str = Path("data/gex_history.db"), parquet_root: Path | str = Path("data/parquet/gex")) -> Path:
    """Generate a reconciliation report comparing completed jobs and Parquet coverage.

    Writes `docs/RECONCILE_JOBSTORE_VS_PARQUET.md` and returns the Path.
    """
    history_db_path = Path(history_db_path)
    parquet_root = Path(parquet_root)
    report = []
    report.append("# Reconciliation: job-store vs Parquet")
    report.append("")
    report.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    report.append("")

    # collect completed jobs
    try:
        con = duckdb.connect(str(history_db_path))
        rows = con.execute("SELECT id, url, ticker, status, created_at FROM import_jobs WHERE status = 'completed' ORDER BY created_at").fetchall()
    except Exception:
        rows = []
    finally:
        try:
            con.close()
        except Exception:
            pass

    report.append('## Completed jobs (local-file inspected where available)')
    report.append('')
    job_dates = {}
    jobs_no_local = []
    jobs_failed = []
    for job in rows:
        job_id, url, ticker, status, created_at = job
        report.append(f'- Job `{job_id}` ticker=`{ticker}` url=`{url}` created_at=`{created_at}`')
        if url and str(url).startswith('file://'):
            fp = Path(url[7:])
            if fp.exists():
                try:
                    df = pd.read_json(fp, orient='records')
                    if 'timestamp' in df.columns:
                        ts = df['timestamp']
                        try:
                            dates = pd.to_datetime(ts, unit='s', errors='coerce').dt.date.dropna().unique().tolist()
                        except Exception:
                            dates = pd.to_datetime(ts, errors='coerce').dt.date.dropna().unique().tolist()
                    else:
                        dates = []
                    job_dates[job_id] = sorted([str(d) for d in dates])
                    if dates:
                        report.append(f'  - Dates in file: {job_dates[job_id]}')
                    else:
                        report.append('  - Dates in file: [] (no timestamp column or empty)')
                except Exception as e:
                    report.append('  - ERROR reading file: ' + repr(e))
                    report.append(traceback.format_exc())
                    jobs_failed.append(job_id)
            else:
                report.append('  - Local staged file not found: ' + str(fp))
                jobs_no_local.append(job_id)
        else:
            report.append('  - URL is remote or missing; skipping local inspection')
            jobs_no_local.append(job_id)

    report.append('')

    # collect parquet dates
    report.append('## Parquet coverage (all files under `data/parquet/gex`)')
    report.append('')
    parquet_dates = set()
    parquet_files = list(parquet_root.rglob('*.parquet'))
    corrupt_files = []
    con = duckdb.connect()
    for f in sorted(parquet_files):
        report.append(f'### {f}')
        try:
            q = f"SELECT DISTINCT DATE(TIMESTAMP 'epoch' + timestamp * INTERVAL '1 second') AS d FROM '{str(f)}'"
            rows = con.execute(q).fetchall()
            dates = sorted({str(r[0]) for r in rows if r[0] is not None})
        except Exception:
            try:
                q2 = f"SELECT DISTINCT DATE(timestamp) AS d FROM '{str(f)}'"
                rows2 = con.execute(q2).fetchall()
                dates = sorted({str(r[0]) for r in rows2 if r[0] is not None})
            except Exception as e:
                err = f'ERROR: {e}'
                dates = [err]
                corrupt_files.append((str(f), str(e)))
        if dates:
            for d in dates:
                if not d.startswith('ERROR'):
                    parquet_dates.add(d)
            report.append('  - Dates:')
            for d in dates:
                report.append(f'    - {d}')
        else:
            report.append('  - Dates: []')
        report.append('')
    con.close()

    # aggregate job dates
    all_job_dates = set()
    for jid, ds in job_dates.items():
        for d in ds:
            all_job_dates.add(d)

    report.append('')
    report.append('## Comparison')
    report.append('')
    report.append(f'- Dates inferred from completed job local files: {sorted(list(all_job_dates))}')
    report.append(f'- Dates present in Parquet files: {sorted(list(parquet_dates))}')
    report.append('')

    # Determine overall date range from parquet (furthest -> most recent), exclude weekends
    parsed_parquet_dates = []
    for d in parquet_dates:
        try:
            parsed_parquet_dates.append(datetime.fromisoformat(d).date())
        except Exception:
            try:
                parsed_parquet_dates.append(pd.to_datetime(d, errors='coerce').date())
            except Exception:
                pass

    if parsed_parquet_dates:
        min_date = min(parsed_parquet_dates)
        max_date = max(parsed_parquet_dates)
        report.append(f'- Date range determined from parquet: {min_date.isoformat()} -> {max_date.isoformat()} (weekends excluded)')
        # build expected business-day set
        expected = set()
        cur = min_date
        from datetime import timedelta
        while cur <= max_date:
            if cur.weekday() < 5:  # Mon-Fri
                expected.add(cur.isoformat())
            cur = cur + timedelta(days=1)

        missing_business = sorted(list(expected - parquet_dates))
        extra = sorted(list(parquet_dates - expected))
        report.append(f'- Business days missing from Parquet in the range: {missing_business}')
        report.append(f'- Parquet contains dates outside expected business days in range: {extra}')
    else:
        report.append('- No valid parquet dates found to determine range.')
        missing_business = []

    report.append('')
    # Show corrupt files explicitly
    if corrupt_files:
        report.append('## Corrupt or unreadable Parquet files')
        report.append('')
        for fn, err in corrupt_files:
            report.append(f'- {fn}: {err}')
        report.append('')
    report.append('## Jobs without local staged file or remote-only')
    for jid in jobs_no_local:
        report.append(f'- {jid}')

    report.append('')
    report.append('## Jobs that failed to be inspected (errors reading file)')
    for jid in jobs_failed:
        report.append(f'- {jid}')

    rp = Path('docs/RECONCILE_JOBSTORE_VS_PARQUET.md')
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text('\n'.join(report))
    return rp


def prune_staged_files(staged_root: Path | str = Path("data/source/gexbot"), retention_days: int = 14) -> list:
    """Remove staged JSON files older than `retention_days` from `staged_root`.

    Returns a list of removed file paths.
    """
    staged_root = Path(staged_root)
    removed = []
    if not staged_root.exists():
        return removed

    now_ts = datetime.utcnow().timestamp()
    for p in staged_root.iterdir():
        try:
            if not p.is_file():
                continue
            if p.suffix.lower() != '.json':
                continue
            mtime = p.stat().st_mtime
            age_days = (now_ts - mtime) / 86400.0
            if age_days > retention_days:
                p.unlink()
                removed.append(str(p))
                LOG.info('Pruned staged JSON older than %s days: %s', retention_days, p)
        except Exception:
            LOG.exception('Failed to prune staged file: %s', p)
    return removed
