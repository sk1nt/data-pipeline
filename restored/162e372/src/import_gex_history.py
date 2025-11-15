"""
Historical GEX data import processor.

Processes queued historical data import jobs from gex_history_queue,
downloads JSON data, parses with full schema support, and stores
snapshots in DuckDB with strikes exported to Parquet.
"""

import json
import os
import re
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict, List, Union, Iterable
from decimal import Decimal
from datetime import datetime
import datetime as dt

import requests
import ijson
import polars as pl

from .lib.gex_history_queue import gex_history_queue
from .lib.gex_database import gex_db
from .lib.logging_config import setup_logging
from .models.api_models import GEXPayload, GEXStrike

# Setup logging
logger = setup_logging()
CHUNK_SIZE = 5000


def convert_decimals(obj: Any) -> Any:
    """Recursively convert Decimal objects to floats for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals(item) for item in obj]
    else:
        return obj


@dataclass
class ProcessedRecord:
    payload: GEXPayload
    raw_record: Dict[str, Any]
    endpoint: str
    timestamp_epoch: int
    net_gex_value: Optional[float]
    max_priors_serialized: Optional[str]


class GEXHistoryImporter:
    """Handles historical GEX data import processing."""

    def __init__(self):
        """Initialize the importer."""
        self.data_dir = Path("data")
        self.source_dir = self.data_dir / "source" / "gexbot"
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = CHUNK_SIZE
        self.parquet_dir = self.data_dir / "parquet" / "gex"
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self._strike_buffers: Dict[
            tuple[str, str, dt.date], List[Dict[str, Any]]
        ] = defaultdict(list)

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
        filename = self._build_filename(url, ticker, endpoint)
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

            with open(local_path, 'wb') as f:
                f.write(response.content)

        try:
            size = local_path.stat().st_size
        except Exception:
            size = None
        logger.info(f"Downloaded {size if size is not None else 'unknown'} bytes")
        return local_path

    def _build_filename(self, url: str, ticker: str, endpoint: str) -> str:
        """Construct filename that preserves the download date if available."""
        date_match = re.search(r"/(\d{4}-\d{2}-\d{2})_", url)
        if date_match:
            day = date_match.group(1)
        else:
            day = dt.datetime.utcnow().strftime("%Y-%m-%d")
        safe_endpoint = (endpoint or "gex_zero").replace("/", "_").replace(" ", "_")
        safe_ticker = (ticker or "UNKNOWN").replace("/", "_").replace(" ", "_")
        return f"{day}_{safe_ticker}_{safe_endpoint}_history.json"

    @staticmethod
    def _safe_name(value: str) -> str:
        return (value or "unknown").replace("/", "_").replace(" ", "_")

    def _import_data(self, file_path: Path, ticker: str, endpoint: str) -> int:
        """Import data from downloaded file."""
        logger.info(f"Importing data from {file_path}")

        imported_count = 0
        error_logged = False

        path_locks: Dict[Path, threading.Lock] = defaultdict(threading.Lock)
        staging_table = f"staging_raw_{ticker.lower()}_{int(datetime.utcnow().timestamp())}"
        try:
            self._load_into_staging(file_path, staging_table)
            processed_batch = self._prepare_from_staging(staging_table, endpoint)
            if processed_batch:
                self._store_snapshots_batch(processed_batch)
                for processed in processed_batch:
                    self._buffer_strikes(processed)
                imported_count += len(processed_batch)
        finally:
            self._drop_staging(staging_table)

        logger.info(f"Imported {imported_count} snapshots from {file_path}")
        self._flush_buffered_strikes(path_locks)
        return imported_count

    def _load_into_staging(self, file_path: Path, table_name: str) -> None:
        with gex_db.gex_data_connection() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(
                f"""
                CREATE TABLE {table_name} AS
                SELECT * FROM read_json_auto('{file_path}', maximum_depth=4)
                """
            )

    def _prepare_from_staging(self, table_name: str, endpoint: str) -> List[ProcessedRecord]:
        processed: List[ProcessedRecord] = []
        with gex_db.gex_data_connection() as conn:
            rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()
            columns = [desc[0] for desc in conn.description]
        for row in rows:
            record = dict(zip(columns, row))
            record_copy = {k: v for k, v in record.items() if k != 'strikes'}
            if 'max_priors' in record_copy and isinstance(record_copy['max_priors'], list):
                record_copy['max_priors'] = json.dumps(record_copy['max_priors'])
            if 'timestamp' in record_copy and isinstance(record_copy['timestamp'], int):
                record_copy['timestamp'] = datetime.fromtimestamp(record_copy['timestamp'])
            payload = GEXPayload(**record_copy)
            strikes = []
            if 'strikes' in record and isinstance(record['strikes'], list):
                for s in record['strikes']:
                    if isinstance(s, list) and len(s) >= 4:
                        strikes.append(
                            GEXStrike(
                                strike=float(s[0]),
                                gamma_now=float(s[1]),
                                vanna=float(s[2]) if len(s) > 2 else None,
                                history=s[3] if len(s) > 3 and isinstance(s[3], list) else None,
                            )
                        )
            payload.strikes = strikes if strikes else None
            payload_endpoint = endpoint or payload.endpoint or "gex_zero"
            payload.endpoint = payload_endpoint
            net_gex_value = payload.net_gex_vol if payload.net_gex_vol is not None else payload.net_gex
            max_priors_serialized = payload.max_priors
            if isinstance(max_priors_serialized, list):
                max_priors_serialized = json.dumps(max_priors_serialized)
            try:
                timestamp_epoch = int(payload.timestamp.timestamp())
            except Exception:
                timestamp_epoch = int(datetime.utcnow().timestamp())
            processed.append(
                ProcessedRecord(
                    payload=payload,
                    raw_record=record,
                    endpoint=payload_endpoint,
                    timestamp_epoch=timestamp_epoch,
                    net_gex_value=net_gex_value,
                    max_priors_serialized=max_priors_serialized,
                )
            )
        return processed

    def _drop_staging(self, table_name: str) -> None:
        with gex_db.gex_data_connection() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")

    def _store_snapshots_batch(self, batch: List[ProcessedRecord]) -> None:
        if not batch:
            return
        rows = []
        for item in batch:
            payload = item.payload
            rows.append(
                [
                    payload.timestamp,
                    payload.ticker,
                    payload.spot_price,
                    payload.zero_gamma,
                    item.net_gex_value,
                    payload.min_dte,
                    payload.sec_min_dte,
                    payload.major_pos_vol,
                    payload.major_pos_oi,
                    payload.major_neg_vol,
                    payload.major_neg_oi,
                    payload.sum_gex_vol,
                    payload.sum_gex_oi,
                    payload.delta_risk_reversal,
                    item.max_priors_serialized,
                ]
            )

        with gex_db.gex_data_connection() as conn:
            conn.executemany(
                """
                INSERT INTO gex_snapshots (
                    timestamp, ticker, spot_price, zero_gamma, net_gex,
                    min_dte, sec_min_dte, major_pos_vol, major_pos_oi,
                    major_neg_vol, major_neg_oi, sum_gex_vol, sum_gex_oi,
                    delta_risk_reversal, max_priors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )



    def _buffer_strikes(self, processed: ProcessedRecord) -> None:
        payload = processed.payload
        strikes = payload.strikes or []
        if not strikes:
            return
        endpoint = processed.endpoint or "gex_zero"
        trade_day = payload.timestamp.date()
        key = (payload.ticker, endpoint, trade_day)
        entries = []
        for strike in strikes:
            entries.append(
                {
                    "timestamp": payload.timestamp.isoformat(),
                    "ticker": payload.ticker,
                    "endpoint": endpoint,
                    "strike": strike.strike,
                    "gamma": strike.gamma_now,
                    "oi_gamma": strike.oi_gamma,
                    "priors": json.dumps(strike.history) if strike.history else None,
                }
            )
        self._strike_buffers[key].extend(entries)

    def _flush_buffered_strikes(
        self, path_locks: Dict[Path, threading.Lock]
    ) -> None:
        if not self._strike_buffers:
            return

        for key, rows in self._strike_buffers.items():
            ticker, endpoint, trade_day = key
            endpoint_clean = endpoint.replace("/", "_").replace(" ", "_")
            year = trade_day.year
            month = f"{trade_day.month:02d}"
            day_str = trade_day.strftime("%Y%m%d")
            target_dir = (
                self.parquet_dir
                / f"{year}"
                / month
                / ticker
                / endpoint_clean
            ) / day_str
            target_dir.mkdir(parents=True, exist_ok=True)
            parquet_file = target_dir / "strikes.parquet"

            df = pl.DataFrame(rows)
            lock = path_locks[parquet_file]
            with lock:
                if parquet_file.exists():
                    existing = pl.read_parquet(parquet_file)
                    df = (
                        pl.concat([existing, df], how="vertical_relaxed")
                        .unique(
                            subset=["timestamp", "ticker", "endpoint", "strike"],
                            keep="last",
                        )
                    )
                df.write_parquet(str(parquet_file), compression="zstd")
                logger.info(
                    "Wrote %s strike rows to %s", len(df), parquet_file
                )

        self._strike_buffers.clear()



def process_historical_imports():
    """Main function to process historical import queue."""
    importer = GEXHistoryImporter()
    importer.process_queue()


if __name__ == "__main__":
    process_historical_imports()
