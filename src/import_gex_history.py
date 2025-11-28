"""
Historical GEX data import processor.

Processes queued historical data import jobs from gex_history_queue,
downloads JSON data, parses with full schema support, and stores
snapshots in DuckDB with strikes exported to Parquet.
"""

import re
from pathlib import Path
from datetime import datetime
import datetime as dt
from urllib.parse import urlsplit

import requests

from .config import settings
from .lib.gex_history_queue import gex_history_queue
from .lib.gex_database import gex_db
from .lib.logging_config import setup_logging

# Setup logging
logger = setup_logging()
CHUNK_SIZE = 5000


class GEXHistoryImporter:
    """Handles historical GEX data import processing."""

    def __init__(self):
        """Initialize the importer."""
        self.data_dir = settings.data_path
        self.source_dir = settings.staging_path
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = CHUNK_SIZE
        self.parquet_dir = settings.parquet_path
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

    def process_queue(self) -> None:
        """Process all pending jobs in the history queue."""
        jobs = gex_history_queue.get_pending_jobs(limit=10)

        for job in jobs:
            job_id, url, ticker, endpoint, attempts = job
            try:
                logger.info(f"Processing job {job_id}: {ticker} from {url}")
                self._process_job(job_id, url, ticker, endpoint)
            except Exception as e:
                logger.error(f"Failed to process job {job_id}: {e}")
                gex_history_queue.mark_job_failed(job_id, str(e))

    def _process_job(self, job_id: int, url: str, ticker: str, endpoint: str) -> None:
        """Process a single import job."""
        # Mark job as started
        gex_history_queue.mark_job_started(job_id)

        # Download the data
        local_file = self._download_file(url, ticker, endpoint)
        trade_day = self._resolve_trade_day(local_file)

        # Parse and import the data
        imported_count = self._import_data(local_file, ticker, endpoint, trade_day)

        # Mark job as completed
        gex_history_queue.mark_job_completed(job_id)

        logger.info(f"Completed job {job_id}: imported {imported_count} snapshots")

    def _download_file(self, url: str, ticker: str, endpoint: str) -> Path:
        """Download file from URL to local storage."""
        filename = self._build_filename(url)
        ticker_dir = self._safe_name(ticker or "UNKNOWN")
        endpoint_dir = self._safe_name(endpoint or "gex_zero")
        dest_dir = self.source_dir / ticker_dir / endpoint_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        local_path = dest_dir / filename

        logger.info(f"Downloading {url} to {local_path}")
        # Support local file:// URLs as well as HTTP/HTTPS
        if url.startswith("file://"):
            src = Path(url[7:])
            if not src.exists():
                raise FileNotFoundError(src)
            local_path.write_bytes(src.read_bytes())
        else:
            response = requests.get(url, timeout=120)
            response.raise_for_status()

            with open(local_path, "wb") as f:
                f.write(response.content)

        try:
            size = local_path.stat().st_size
        except Exception:
            size = None
        logger.info(f"Downloaded {size if size is not None else 'unknown'} bytes")
        return local_path

    def _build_filename(self, url: str) -> str:
        """Preserve the remote file name when downloading to staging."""
        parsed = urlsplit(url)
        candidate = Path(parsed.path).name
        if candidate:
            return candidate
        # Fallback to timestamped name if URL has no basename
        timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"gex_history_{timestamp}.json"

    def _resolve_trade_day(self, local_file: Path) -> dt.date:
        match = re.search(r"(\d{4}-\d{2}-\d{2})", local_file.name)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y-%m-%d").date()
            except ValueError:
                pass
        return datetime.utcnow().date()

    @staticmethod
    def _safe_name(value: str) -> str:
        return (value or "unknown").replace("/", "_").replace(" ", "_")

    def _import_data(
        self, file_path: Path, ticker: str, endpoint: str, trade_day: dt.date
    ) -> int:
        """Import data from downloaded file."""
        logger.info(f"Importing data from {file_path}")

        staging_table = (
            f"staging_raw_{ticker.lower()}_{int(datetime.utcnow().timestamp())}"
        )
        try:
            self._load_into_staging(file_path, staging_table)
            inserted = self._insert_snapshots_from_staging(
                staging_table, ticker, trade_day
            )
            strike_rows = self._insert_strikes_from_staging(
                staging_table, ticker, trade_day
            )
            self._export_strikes_to_parquet(staging_table, ticker, endpoint, trade_day)
            logger.info("Inserted %s snapshots and %s strikes", inserted, strike_rows)
        finally:
            self._drop_staging(staging_table)

        return inserted

    def _load_into_staging(self, file_path: Path, table_name: str) -> None:
        with gex_db.gex_data_connection() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(
                f"""
                CREATE TABLE {table_name} AS
                SELECT * FROM read_json_auto('{file_path}', maximum_depth=4)
                """
            )

    def _drop_staging(self, table_name: str) -> None:
        with gex_db.gex_data_connection() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")

    def _insert_snapshots_from_staging(
        self, table_name: str, ticker: str, trade_day: dt.date
    ) -> int:
        ticker_sql = (ticker or "UNKNOWN").upper()
        day_str = trade_day.isoformat()
        with gex_db.gex_data_connection() as conn:
            # Ensure new columns exist
            try:
                conn.execute(
                    "ALTER TABLE gex_snapshots ADD COLUMN IF NOT EXISTS strikes VARCHAR"
                )
            except Exception:
                pass
            conn.execute(
                """
                DELETE FROM gex_snapshots
                WHERE ticker = ? AND CAST(to_timestamp(timestamp/1000.0) AS DATE) = CAST(? AS DATE)
                """,
                [ticker_sql, day_str],
            )
            conn.execute(
                f"""
                INSERT INTO gex_snapshots (
                    timestamp, ticker, spot_price, zero_gamma, net_gex,
                    min_dte, sec_min_dte, major_pos_vol, major_pos_oi,
                    major_neg_vol, major_neg_oi, sum_gex_vol, sum_gex_oi,
                    delta_risk_reversal, max_priors
                )
                SELECT
                    CAST(timestamp * 1000 AS BIGINT) AS timestamp,
                    UPPER(COALESCE(ticker, '{ticker_sql}')) AS ticker,
                    spot AS spot_price,
                    zero_gamma,
                    COALESCE(sum_gex_vol, 0) AS net_gex,
                    CAST(min_dte AS INTEGER) AS min_dte,
                    CAST(sec_min_dte AS INTEGER) AS sec_min_dte,
                    major_pos_vol,
                    major_pos_oi,
                    major_neg_vol,
                    major_neg_oi,
                    sum_gex_vol,
                    sum_gex_oi,
                    delta_risk_reversal,
                    CASE WHEN max_priors IS NULL THEN NULL ELSE to_json(max_priors) END AS max_priors,
                    CASE WHEN strikes IS NULL THEN NULL ELSE to_json(strikes) END AS strikes
                FROM {table_name}
                """
            )
            inserted = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        return inserted

    def _insert_strikes_from_staging(
        self, table_name: str, ticker: str, trade_day: dt.date
    ) -> int:
        ticker_sql = (ticker or "UNKNOWN").upper()
        day_str = trade_day.isoformat()
        with gex_db.gex_data_connection() as conn:
            conn.execute(
                """
                DELETE FROM gex_strikes
                WHERE ticker = ? AND CAST(to_timestamp(timestamp/1000.0) AS DATE) = CAST(? AS DATE)
                """,
                [ticker_sql, day_str],
            )
            strike_count = conn.execute(
                f"SELECT COALESCE(SUM(array_length(strikes)), 0) FROM {table_name}"
            ).fetchone()[0]
            conn.execute(
                f"""
                INSERT INTO gex_strikes (
                    timestamp, ticker, strike, gamma, oi_gamma, priors
                )
                SELECT
                    CAST(t.timestamp * 1000 AS BIGINT) AS timestamp,
                    '{ticker_sql}' AS ticker,
                    TRY_CAST(strike[1] AS DOUBLE) AS strike,
                    TRY_CAST(strike[2] AS DOUBLE) AS gamma,
                    TRY_CAST(strike[3] AS DOUBLE) AS oi_gamma,
                    CASE WHEN array_length(strike) >= 4 THEN strike[4] ELSE NULL END AS priors
                FROM {table_name} AS t,
                     UNNEST(t.strikes) AS strike_entry(strike)
                WHERE t.strikes IS NOT NULL
                """
            )
        return strike_count

    def _export_strikes_to_parquet(
        self,
        table_name: str,
        ticker: str,
        endpoint: str,
        trade_day: dt.date,
    ) -> None:
        ticker_sql = (ticker or "UNKNOWN").upper()
        endpoint_clean = (endpoint or "gex_zero").replace("/", "_").replace(" ", "_")
        day_str = trade_day.strftime("%Y%m%d")
        target_dir = self.parquet_dir / ticker_sql / endpoint_clean
        target_dir.mkdir(parents=True, exist_ok=True)
        parquet_file = target_dir / f"{day_str}.strikes.parquet"
        if parquet_file.exists():
            parquet_file.unlink()
        select_sql = f"""
            SELECT
                CAST(t.timestamp * 1000 AS BIGINT) AS timestamp,
                '{ticker_sql}' AS ticker,
                '{endpoint_clean}' AS endpoint,
                TRY_CAST(strike[1] AS DOUBLE) AS strike,
                TRY_CAST(strike[2] AS DOUBLE) AS gamma,
                TRY_CAST(strike[3] AS DOUBLE) AS oi_gamma,
                CASE WHEN array_length(strike) >= 4 THEN strike[4] ELSE NULL END AS priors
            FROM {table_name} AS t,
                 UNNEST(t.strikes) AS strike_entry(strike)
            WHERE t.strikes IS NOT NULL
        """
        with gex_db.gex_data_connection() as conn:
            conn.execute(
                f"COPY ({select_sql}) TO '{parquet_file.as_posix()}' (FORMAT 'parquet', COMPRESSION 'zstd')"
            )


def process_historical_imports():
    """Main function to process historical import queue."""
    importer = GEXHistoryImporter()
    importer.process_queue()


if __name__ == "__main__":
    process_historical_imports()
