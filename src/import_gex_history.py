"""
Historical GEX data import processor.

Processes queued historical data import jobs from gex_history_queue,
downloads JSON data, parses with full schema support, and stores
snapshots in DuckDB with strikes exported to Parquet.
"""

import json
from pathlib import Path
from typing import Optional

import requests
import polars as pl

from .lib.gex_history_queue import gex_history_queue
from .lib.gex_database import gex_db
from .lib.logging_config import setup_logging
from .models.api_models import GEXPayload

# Setup logging
logger = setup_logging()


class GEXHistoryImporter:
    """Handles historical GEX data import processing."""

    def __init__(self):
        """Initialize the importer."""
        self.data_dir = Path("data")
        self.source_dir = self.data_dir / "source" / "gexbot"
        self.parquet_dir = self.data_dir / "parquet" / "gex"
        self.source_dir.mkdir(parents=True, exist_ok=True)

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

        # Parse and import the data
        imported_count = self._import_data(local_file, ticker, endpoint)

        # Mark job as completed
        gex_history_queue.mark_job_completed(job_id)

        logger.info(f"Completed job {job_id}: imported {imported_count} snapshots")

    def _download_file(self, url: str, ticker: str, endpoint: str) -> Path:
        """Download file from URL to local storage."""
        filename = f"{ticker}_{endpoint}_history.json"
        local_path = self.source_dir / filename

        logger.info(f"Downloading {url} to {local_path}")
        response = requests.get(url, timeout=120)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"Downloaded {len(response.content)} bytes")
        return local_path

    def _import_data(self, file_path: Path, ticker: str, endpoint: str) -> int:
        """Import data from downloaded file."""
        logger.info(f"Importing data from {file_path}")

        # Read and parse JSON
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, dict) and 'data' in data:
            records = data['data']
        elif isinstance(data, list):
            records = data
        else:
            records = [data]

        imported_count = 0

        for record in records:
            try:
                # Parse record using our full schema model
                payload = GEXPayload(**record)

                # Store snapshot in database
                self._store_snapshot(payload)

                # Export strikes to Parquet if present
                if payload.strikes:
                    self._export_strikes_to_parquet(payload)

                imported_count += 1

            except Exception as e:
                logger.warning(f"Failed to import record: {e}")
                continue

        logger.info(f"Imported {imported_count} snapshots from {len(records)} records")
        return imported_count

    def _store_snapshot(self, payload: GEXPayload) -> None:
        """Store GEX snapshot in database."""
        # Handle net_gex - use net_gex_vol if available, otherwise net_gex
        net_gex_value = payload.net_gex_vol if payload.net_gex_vol is not None else payload.net_gex

        # Handle max_priors - convert list to JSON string if needed
        max_priors_value = payload.max_priors
        if isinstance(max_priors_value, list):
            max_priors_value = json.dumps(max_priors_value)

        with gex_db.gex_data_connection() as conn:
            conn.execute("""
                INSERT INTO gex_snapshots (
                    timestamp, ticker, spot_price, zero_gamma, net_gex,
                    min_dte, sec_min_dte, major_pos_vol, major_pos_oi,
                    major_neg_vol, major_neg_oi, sum_gex_vol, sum_gex_oi,
                    delta_risk_reversal, max_priors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                payload.timestamp, payload.ticker, payload.spot_price,
                payload.zero_gamma, net_gex_value, payload.min_dte,
                payload.sec_min_dte, payload.major_pos_vol, payload.major_pos_oi,
                payload.major_neg_vol, payload.major_neg_oi, payload.sum_gex_vol,
                payload.sum_gex_oi, payload.delta_risk_reversal, max_priors_value
            ])

    def _export_strikes_to_parquet(self, payload: GEXPayload) -> None:
        """Export strike data to Parquet."""
        if not payload.strikes:
            return

        # Convert strikes to Polars DataFrame
        strikes_data = []
        for strike in payload.strikes:
            strikes_data.append({
                'timestamp': payload.timestamp.isoformat(),
                'ticker': payload.ticker,
                'strike': strike.strike,
                'gamma': strike.gamma_now,
                'oi_gamma': strike.oi_gamma,
                'priors': json.dumps(strike.history) if strike.history else None
            })

        df = pl.DataFrame(strikes_data)

        # Determine Parquet path
        year = payload.timestamp.year
        month = f"{payload.timestamp.month:02d}"
        endpoint_clean = str(payload.endpoint or 'unknown').replace('/', '_').replace(' ', '_')

        parquet_dir = self.parquet_dir / f"year={year}" / f"month={month}" / payload.ticker / endpoint_clean
        parquet_dir.mkdir(parents=True, exist_ok=True)
        parquet_file = parquet_dir / "strikes.parquet"

        # Write to Parquet
        df.write_parquet(str(parquet_file))
        logger.info(f"Exported {len(strikes_data)} strikes to {parquet_file}")


def process_historical_imports():
    """Main function to process historical import queue."""
    importer = GEXHistoryImporter()
    importer.process_queue()


if __name__ == "__main__":
    process_historical_imports()