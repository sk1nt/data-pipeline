import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import duckdb
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_history_queue():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.lib.gex_history_queue import gex_history_queue as queue_helper

    return queue_helper


gex_history_queue = _load_history_queue()


def download_file(url: str, out_path: Path) -> None:
    """Download a file from a URL to a local path."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def import_to_duckdb_and_parquet(
    file_path: Path,
    duckdb_path: Path,
    parquet_path: Path,
    table_name: str = "strikes",
    batch_size: int = 50,
) -> None:
    """Import file to DuckDB and export strikes to Parquet using streaming JSON processing."""
    LOG = logging.getLogger("import_gex_history")

    import ijson
    import pandas as pd
    import json
    from decimal import Decimal

    def convert_decimals(obj):
        """Recursively convert Decimal objects to floats."""
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, list):
            return [convert_decimals(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_decimals(value) for key, value in obj.items()}
        else:
            return obj

    LOG.info(
        f"Processing JSON file with streaming: {file_path} in batches of {batch_size}"
    )

    con = duckdb.connect(str(duckdb_path))

    # Drop table if exists to allow re-imports
    con.execute("DROP TABLE IF EXISTS strikes")

    # Create table with explicit schema to avoid type inference issues
    create_table_sql = f"""
    CREATE TABLE {table_name} (
        timestamp BIGINT,
        timestamp_ms BIGINT,
        timestamp_iso VARCHAR,
        ticker VARCHAR,
        min_dte INTEGER,
        sec_min_dte INTEGER,
        spot DOUBLE,
        zero_gamma DOUBLE,
        major_pos_vol DOUBLE,
        major_pos_oi DOUBLE,
        major_neg_vol DOUBLE,
        major_neg_oi DOUBLE,
        strikes VARCHAR,  -- Store as JSON string
        sum_gex_vol DOUBLE,
        sum_gex_oi DOUBLE,
        delta_risk_reversal DOUBLE,
        max_priors VARCHAR  -- Store as JSON string
    )
    """
    con.execute(create_table_sql)
    LOG.info("Created table with explicit schema")

    batch_data = []
    column_order = [
        "timestamp",
        "timestamp_ms",
        "timestamp_iso",
        "ticker",
        "min_dte",
        "sec_min_dte",
        "spot",
        "zero_gamma",
        "major_pos_vol",
        "major_pos_oi",
        "major_neg_vol",
        "major_neg_oi",
        "strikes",
        "sum_gex_vol",
        "sum_gex_oi",
        "delta_risk_reversal",
        "max_priors",
    ]
    batch_count = 0
    total_processed = 0

    # Use ijson for streaming JSON parsing
    with open(file_path, "rb") as f:
        # Parse items in the JSON array
        for record in ijson.items(f, "item"):
            # Convert Decimal objects to floats recursively
            record = convert_decimals(record)

            ts = record.get("timestamp")
            if isinstance(ts, str):
                ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                ts_epoch = int(ts_dt.timestamp())
            else:
                ts_epoch = int(ts)
                ts_dt = datetime.fromtimestamp(ts_epoch, tz=timezone.utc)
            record["timestamp"] = ts_epoch
            record["timestamp_ms"] = ts_epoch * 1000 + int(ts_dt.microsecond / 1000)
            record["timestamp_iso"] = ts_dt.isoformat()

            # Convert complex types to JSON strings
            if "strikes" in record:
                record["strikes"] = json.dumps(record["strikes"])
            if "max_priors" in record:
                record["max_priors"] = json.dumps(record["max_priors"])

            batch_data.append(record)

            # Process batch when it reaches batch_size
            if len(batch_data) >= batch_size:
                batch_count += 1
                LOG.info(
                    f"Processing batch {batch_count} ({len(batch_data)} records, total: {total_processed + len(batch_data)})"
                )

                # Convert to DataFrame
                batch_df = pd.DataFrame(batch_data, columns=column_order)
                con.register("batch_df", batch_df)

                # Insert batch with explicit column order
                selected_cols = ", ".join(column_order)
                con.execute(
                    f"INSERT INTO {table_name} ({selected_cols}) SELECT {selected_cols} FROM batch_df"
                )

                total_processed += len(batch_data)
                batch_data = []  # Reset for next batch

    # Process any remaining records
    if batch_data:
        batch_count += 1
        LOG.info(
            f"Processing final batch {batch_count} ({len(batch_data)} records, total: {total_processed + len(batch_data)})"
        )

        batch_df = pd.DataFrame(batch_data, columns=column_order)
        con.register("batch_df", batch_df)
        selected_cols = ", ".join(column_order)
        con.execute(
            f"INSERT INTO {table_name} ({selected_cols}) SELECT {selected_cols} FROM batch_df"
        )

        total_processed += len(batch_data)

    LOG.info(f"Completed processing {total_processed} records")
    LOG.info(f"Exporting to Parquet: {parquet_path}")

    # Export the entire table to Parquet
    con.execute(f"COPY {table_name} TO '{parquet_path}' (FORMAT PARQUET)")
    con.close()


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    LOG = logging.getLogger("import_gex_history")
    parser = argparse.ArgumentParser(
        description="Download GEX history blob and import into DuckDB/Parquet."
    )
    parser.add_argument("url", help="Signed URL for the history blob")
    parser.add_argument("ticker", help="Ticker symbol, e.g. NQ_NDX")
    parser.add_argument("endpoint", help="Endpoint/kind for the dataset, e.g. gex_zero")
    parser.add_argument(
        "--queue-id", type=int, help="Optional queue id for status tracking"
    )
    args = parser.parse_args()

    queue_id = args.queue_id
    url = args.url
    ticker = args.ticker
    endpoint = args.endpoint

    duckdb_file = PROJECT_ROOT / "data" / "gex_data.db"

    def infer_trade_date() -> datetime:
        url_match = re.search(r"/(\d{4}-\d{2}-\d{2})_", url)
        if url_match:
            return datetime.fromisoformat(url_match.group(1))
        endpoint_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", endpoint)
        if endpoint_match:
            return datetime(
                int(endpoint_match.group(1)),
                int(endpoint_match.group(2)),
                int(endpoint_match.group(3)),
            )
        return datetime.utcnow()

    trade_date = infer_trade_date()
    year, month = str(trade_date.year), f"{trade_date.month:02d}"
    day_str = trade_date.strftime("%Y-%m-%d")

    # Include endpoint (e.g., gex_zero) in directory and filename for clarity
    endpoint_clean = endpoint.replace("/", "_").replace(" ", "_")
    raw_dir = PROJECT_ROOT / "data" / "source" / "gexbot" / ticker / endpoint_clean
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_name = Path(urlsplit(url).path).name or f"{day_str}.json"
    local_file = raw_dir / source_name
    LOG.info(
        "Preparing to download: url=%s, ticker=%s, endpoint=%s, trade_date=%s, local_file=%s",
        url,
        ticker,
        endpoint,
        day_str,
        local_file,
    )
    parquet_dir = (
        PROJECT_ROOT
        / "data"
        / "parquet"
        / "gex"
        / year
        / month
        / ticker
        / endpoint_clean
        / day_str
    )
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_file = parquet_dir / "strikes.parquet"
    (PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)

    try:
        if queue_id is not None:
            LOG.info(f"Marking queue {queue_id} as started")
            gex_history_queue.mark_job_started(queue_id)
        LOG.info(f"Downloading {url} to {local_file}...")
        download_file(url, local_file)
        LOG.info(
            f"Download complete. File exists: {local_file.exists()} Size: {local_file.stat().st_size if local_file.exists() else 'N/A'}"
        )
        LOG.info(f"Importing {local_file} to DuckDB and Parquet...")
        import_to_duckdb_and_parquet(
            local_file, duckdb_file, parquet_file, batch_size=50
        )
        LOG.info(
            f"Done. Parquet file: {parquet_file} Exists: {parquet_file.exists()} Size: {parquet_file.stat().st_size if parquet_file.exists() else 'N/A'}"
        )
        if queue_id is not None:
            LOG.info(f"Marking queue {queue_id} as completed")
            gex_history_queue.mark_job_completed(queue_id)
    except Exception as exc:
        LOG.error(f"Error during import: {exc}", exc_info=True)
        if queue_id is not None:
            gex_history_queue.mark_job_failed(queue_id, str(exc))
        raise


if __name__ == "__main__":
    main()
