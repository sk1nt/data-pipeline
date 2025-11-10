# Quickstart: Migrate Data Pipeline Functionality

**Date**: November 9, 2025  
**Feature**: 003-data-pipeline-migration  
**Purpose**: Guide for running the migrated data pipeline server

## Prerequisites

- Python 3.11
- Required dependencies: FastAPI, Pydantic, polars, duckdb, requests, etc.
- Data directory structure: `data/` with subdirectories for database and files

## Installation

1. Ensure the migrated `data_pipeline.py` is in the project root or `src/` directory
2. Install dependencies:
   ```bash
   pip install fastapi pydantic polars duckdb requests
   ```

## Running the Server

1. Start the server:
   ```bash
   python src/data_pipeline.py --host 127.0.0.1 --port 8877
   ```

2. The server will start on `http://127.0.0.1:8877`

## Testing Endpoints

### GEX Data Capture
```bash
curl -X POST http://127.0.0.1:8877/gex \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "SPX",
    "timestamp": 1731177600,
    "spot": 4500.0,
    "zero_gamma": 0.5,
    "strikes": [[4400, 0.1], [4500, 0.2]]
  }'
```

Expected response: `{"status": "ok", "ticker": "SPX"}`

### Historical Data Import
```bash
curl -X POST http://127.0.0.1:8877/gex_history_url \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/history.json",
    "ticker": "SPX",
    "endpoint": "gex_zero"
  }'
```

Expected response: Queued import with ID and paths

### Universal Webhook
```bash
curl -X POST http://127.0.0.1:8877/uw \
  -H "Content-Type: application/json" \
  -d '{"topic": "test", "payload": {"data": "example"}}'
```

Expected response: `{"status": "received"}`

## Data Locations

- **Database**: `data/gex_data.db` (DuckDB)
- **Parquet Files**: `data/parquet/gex/YYYY/MM/{ticker}/{endpoint}/strikes.parquet`
- **Staged Downloads**: `data/source/gexbot/{ticker}_{endpoint}_history.json`
- **SQLite Metadata**: `data/gex_history.db`

## Monitoring

- Check server logs for processing status
- Verify files are created in expected locations
- Monitor database for imported data

## Troubleshooting

- Ensure `data/` directory exists and is writable
- Check for import errors in logs
- Verify network connectivity for downloads
- Confirm all dependencies are installed
