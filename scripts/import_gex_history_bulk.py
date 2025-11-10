#!/usr/bin/env python3
"""
Bulk importer for historical gex_full snapshots downloaded from hist.gex.bot.

This script downloads or processes historical GEX (Gamma Exposure) data from GEXBot.com,
parses the JSON/JSONL records, and imports them into the local SQLite database for analysis.

Usage Examples:
    # Download from URL and import
    python scripts/import_gex_history_bulk.py --url "https://hist.gex.bot/...json?...sig=..." --ticker SPX

    # Import from existing file
    python scripts/import_gex_history_bulk.py --file outputs/gex_bridge/history/2025-10-24_SPX_classic_gex_full.json

    # Override endpoint label
    python scripts/import_gex_history_bulk.py --file data.json --endpoint-label gex_zero

The script supports:
- Downloading large history blobspy from signed URLs
- Processing compressed (.gz) and uncompressed JSON/JSONL files
- Handling different record formats (array of records or line-delimited)
- Persisting snapshots to SQLite with proper metadata
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import from data-pipeline module
import importlib.util
spec = importlib.util.spec_from_file_location("data_pipeline", PROJECT_ROOT / "data-pipeline.py")
data_pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_pipeline)

SnapshotStore = data_pipeline.SnapshotStore
persist_snapshot_to_db = data_pipeline.persist_snapshot_to_db

LOG = logging.getLogger("gex_history_import")
HISTORY_DIR = PROJECT_ROOT / "outputs" / "gex_bridge" / "history"


def download_history_blob(url: str, destination: Path) -> Path:
    """
    Download a history blob from the given URL to the specified destination.

    Args:
        url: The signed URL to download from.
        destination: Local file path to save the downloaded file.

    Returns:
        The path to the downloaded file.

    Raises:
        requests.HTTPError: If the download fails.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloading %s -> %s", url, destination)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
    LOG.info("Download complete: %.2f MB", destination.stat().st_size / (1024 * 1024))
    return destination


def iter_history_records(path: Path) -> Iterator[dict]:
    """
    Iterate over records in a JSON or JSONL history file.

    Supports both array-based JSON and line-delimited JSONL formats.
    Handles compressed (.gz) files automatically.

    Args:
        path: Path to the history file.

    Yields:
        Individual record dictionaries.

    Raises:
        ValueError: If the JSON structure is unexpected.
    """
    LOG.info("Reading %s", path)
    if path.suffix.endswith(".gz"):
        import gzip

        opener = gzip.open  # type: ignore[attr-defined]
        text_mode = "rt"
    else:
        opener = open
        text_mode = "r"

    with opener(path, text_mode, encoding="utf-8") as handle:
        first_char = handle.read(1)
        handle.seek(0)
        if not first_char:
            return
        if first_char == "[":
            payload = json.load(handle)
            if isinstance(payload, dict):
                payload = (
                    payload.get("snapshots")
                    or payload.get("records")
                    or payload.get("data")
                    or payload.get("payloads")
                    or []
                )
            if not isinstance(payload, list):
                raise ValueError("Unexpected JSON structure in history blob")
            for record in payload:
                if record:
                    yield record
        else:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def extract_payload(record: dict, default_kind: str = "snapshot"):
    """
    Extract key fields from a raw record dictionary.

    Args:
        record: The raw record dict.
        default_kind: Default payload kind if not specified.

    Returns:
        Tuple of (ticker, payload_kind, payload, received_at, endpoint).
    """
    payload = record.get("payload") if isinstance(record, dict) else None
    if not isinstance(payload, dict):
        payload = record
    ticker = record.get("ticker") or payload.get("ticker")
    payload_kind = (
        record.get("kind")
        or record.get("type")
        or payload.get("kind")
        or payload.get("type")
        or default_kind
    )
    received_at = record.get("received_at") or payload.get("received_at")
    endpoint = record.get("endpoint") or payload.get("endpoint")
    return ticker, payload_kind, payload, received_at, endpoint


def import_history(
    records: Iterable[dict],
    default_ticker: Optional[str] = None,
    forced_endpoint: Optional[str] = None,
) -> None:
    """
    Import records into the SnapshotStore and persist to database.

    Args:
        records: Iterable of record dictionaries.
        default_ticker: Fallback ticker if missing in record.
        forced_endpoint: Override endpoint label for all records.
    """
    store = SnapshotStore()
    total = 0
    imported_tickers = set()
    imported_endpoints = set()
    for idx, record in enumerate(records, start=1):
        ticker, payload_kind, payload, received_at, endpoint = extract_payload(record)
        ticker = ticker or default_ticker
        if not ticker:
            LOG.warning("Skipping record %s: missing ticker", idx)
            continue
        payload_kind = (payload_kind or "snapshot").lower()
        if payload_kind not in {"snapshot", "majors", "maxchange"}:
            LOG.debug("Skipping record %s (%s)", idx, payload_kind)
            continue
        effective_endpoint = forced_endpoint or endpoint
        snapshot_obj = store.update(ticker, payload_kind, payload, received_at, {"endpoint": effective_endpoint})
        persist_snapshot_to_db(snapshot_obj.ticker, snapshot_obj, payload_kind, payload, effective_endpoint)
        imported_tickers.add(ticker)
        imported_endpoints.add(effective_endpoint)
        total += 1
        if total % 500 == 0:
            LOG.info("Processed %s records...", total)
    LOG.info("Import complete: %s records stored", total)

    # Export strike data to Parquet for each ticker/endpoint
    import duckdb
    import polars as pl
    from datetime import datetime
    db_path = str(PROJECT_ROOT / "data" / "gex_data.db")
    con = duckdb.connect(db_path)
    for ticker in imported_tickers:
        for endpoint in imported_endpoints:
            # Find all timestamps for this ticker/endpoint
            try:
                df = con.execute(f"SELECT * FROM gex_bridge_strikes WHERE ticker = ? AND endpoint = ?", [ticker, endpoint]).df()
                if df.empty:
                    continue
                # Extract year/month from first timestamp
                ts = df["timestamp"].iloc[0]
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                else:
                    dt = datetime.utcfromtimestamp(int(ts))
                year = dt.year
                month = f"{dt.month:02d}"
                endpoint_clean = str(endpoint).replace('/', '_').replace(' ', '_')
                parquet_dir = PROJECT_ROOT / "data" / "parquet" / "gex" / f"{year}" / f"{month:02d}" / ticker / endpoint_clean
                parquet_dir.mkdir(parents=True, exist_ok=True)
                parquet_file = parquet_dir / "strikes.parquet"
                pl.from_pandas(df).write_parquet(str(parquet_file))
                LOG.info(f"Exported Parquet: {parquet_file} ({len(df)} rows)")
            except Exception as e:
                LOG.warning(f"Failed Parquet export for {ticker}/{endpoint}: {e}")
    con.close()


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Import historical gex_full snapshots into SQLite")
    parser.add_argument("--url", help="Signed history blob URL")
    parser.add_argument("--file", help="Existing history JSON/JSONL file")
    parser.add_argument("--ticker", help="Fallback ticker if missing in records")
    parser.add_argument("--output", help="Override download location")
    parser.add_argument("--endpoint-label", help="Override endpoint label (e.g. gex_full, gex_zero)")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the script.
    """
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    if not args.url and not args.file:
        raise SystemExit("Provide either --url or --file")

    if args.url:
        filename = args.output or args.url.split("/")[-1].split("?")[0]
        destination = Path(filename)
        if not destination.is_absolute():
            destination = HISTORY_DIR / destination
        blob_path = download_history_blob(args.url, destination)
    else:
        blob_path = Path(args.file).expanduser().resolve()
        if not blob_path.exists():
            raise SystemExit(f"File not found: {blob_path}")

    records = iter_history_records(blob_path)
    import_history(records, default_ticker=args.ticker, forced_endpoint=args.endpoint_label)


if __name__ == "__main__":
    main()
