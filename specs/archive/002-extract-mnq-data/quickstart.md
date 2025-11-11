# Quickstart: MNQ Tick & Depth Extraction

## Prerequisites
- Python 3.11
- Install dependencies: `pip install -r requirements.txt`

## Usage

### Extract 70 days of MNQ tick and depth data

```bash
python scripts/orchestrator.py --input_dir <SCID_DIR> --output_dir <PARQUET_DIR> --days 70 --batch_size 6
```

- Tick data is written to DuckDB and daily Parquet files.
- Depth data is written to daily Parquet files, with metadata in DuckDB.

### API Trigger (GEX History)

- POST to `http://localhost:8877/gex_history_url` with JSON body to trigger GEX data import.

## Output
- Parquet files: `<PARQUET_DIR>/ticks_<date>.parquet`, `<PARQUET_DIR>/depth_<date>.parquet`
- DuckDB: `<OUTPUT_DIR>/tick_metadata.duckdb`, `<OUTPUT_DIR>/depth_metadata.duckdb`

## Error Handling
- All errors are logged to `logs/`.
- Extraction is robust to malformed records and interruptions.