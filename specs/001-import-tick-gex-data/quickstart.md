# Quickstart: Import GEX Data

## Overview

This guide covers importing comprehensive GEX (Gamma Exposure) data for NQ_NDX futures contracts from legacy-source outputs. All GEX fields are imported without filtering and stored in both DuckDB for fast querying and Parquet files for long-term analytics. Tick and market depth data import is deferred until data completeness is verified.

## Prerequisites

- **Python**: 3.11 or higher
- **Dependencies**: polars, duckdb, redis, fastapi, pydantic, slowapi, uvicorn
- **Data Source**: Access to legacy-source outputs directory containing JSON and CSV files
- **Storage**: Sufficient disk space (expect ~2x data size for DuckDB + Parquet)
- **Memory**: Minimum 8GB RAM recommended for large imports

## Installation

### 1. Clone and Setup Project
```bash
cd /home/rwest/projects/data-pipeline
pip install -r requirements.txt
```

### 2. Verify Dependencies
```bash
python -c "import polars, duckdb, redis, fastapi, pydantic; print('All dependencies installed')"
```

### 3. Prepare Data Directory
Ensure legacy-source GEX outputs are available:
```bash
ls -la projects/legacy-source/outputs/gex_bridge/history/
# Should contain NQ_NDX GEX JSON files
```

## Running the Import

### Option 1: Command Line Import
```bash
# Dry run to validate data without importing
python src/import_data.py --dry-run --source projects/legacy-source/outputs/gex_bridge/history

# Full import with progress logging
python src/import_data.py --source projects/legacy-source/outputs/gex_bridge/history --verbose

# Import specific data types only
python src/import_data.py --source projects/legacy-source/outputs/gex_bridge/history --types gex
```

### Option 2: API-Based Import
Start the API server:
```bash
cd src
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Import via REST API:
```bash
curl -X POST http://localhost:8000/import \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "projects/legacy-source/outputs/gex_bridge/history",
    "data_types": ["gex"]
  }'
```

## Verification Steps

### 1. Check Import Logs
Monitor the import process:
```bash
tail -f logs/import_$(date +%Y%m%d).log
```

Expected output includes:
- Files discovered and processed
- Records imported per data type
- Validation errors (should be <0.1%)
- Performance metrics (target: <2 hours for full import)

### 2. Query Imported Data
Use the API to verify data:

**GEX Snapshots:**
```bash
curl "http://localhost:8000/gex?symbol=NQ_NDX&limit=5"
```

### 3. Validate Data Integrity
Compare against source files:
```bash
# Check GEX field completeness
python -c "
import duckdb
conn = duckdb.connect('data/market_data.db')
result = conn.execute('SELECT COUNT(*) as total, COUNT(min_dte) as min_dte_count FROM gex_snapshots').fetchone()
print(f'GEX records: {result[0]}, with min_dte: {result[1]}')
"
```

### 4. Performance Validation
Verify import performance meets targets:
```bash
# Check import duration from lineage
curl "http://localhost:8000/lineage?limit=1" | jq '.[] | .duration_seconds'
# Should be <7200 seconds (2 hours) for full import
```

## Data Storage Locations

After successful import, data is available in:

- **DuckDB**: `data/market_data.db` (fast querying, real-time access)
- **Parquet**: `data/parquet/` directory (analytics, archival)
  - `gex/2024/01/*.parquet`

## Troubleshooting

### Common Issues

**Import Fails with Permission Errors:**
```bash
# Ensure proper permissions on data directories
chmod -R 755 projects/legacy-source/outputs/gex_bridge/history/
chmod -R 755 data/
```

**Memory Issues During Large Imports:**
```bash
# Reduce batch size or process in chunks
python src/import_data.py --source ../legacy-source/outputs --batch-size 1000
```

**Validation Errors >0.1%:**
- Check source data quality
- Review error logs for patterns
- Contact data provider for corrupted files

**Slow Import Performance:**
- Ensure SSD storage for data directory
- Increase system memory if possible
- Check for competing processes

### Recovery Procedures

**Resume Failed Import:**
```bash
# Check lineage for completed files
curl "http://localhost:8000/lineage?status=failed"

# Resume with --resume flag
python src/import_data.py --source projects/legacy-source/outputs/gex_bridge/history --resume
```

**Re-import Specific Data Type:**
```bash
# Clear and re-import GEX data only
python src/import_data.py --source projects/legacy-source/outputs/gex_bridge/history --types gex --clear-existing
```

### Monitoring and Maintenance

**Regular Health Checks:**
```bash
# Daily validation script
python scripts/validate_import.py

# Check data freshness
python -c "
import duckdb
conn = duckdb.connect('data/market_data.db')
latest = conn.execute('SELECT MAX(timestamp) FROM gex_snapshots').fetchone()[0]
print(f'Latest GEX data: {latest}')
"
```

**Backup Strategy:**
```bash
# Backup Parquet files (primary archival format)
tar -czf backup_$(date +%Y%m%d).tar.gz data/parquet/

# Backup DuckDB (can be regenerated from Parquet if needed)
cp data/market_data.db backups/
```

## Performance Benchmarks

- **Full Import**: <2 hours for typical dataset
- **Query Latency**: <100ms for time-range queries
- **Data Integrity**: >99.9% successful imports
- **Storage Efficiency**: ~50% space savings vs raw JSON/CSV

## Next Steps

- Explore the API documentation at `http://localhost:8000/docs`
- Set up automated imports via cron jobs
- Configure monitoring dashboards for data pipeline health
- Implement downstream analytics using the imported data</content>
<parameter name="filePath">/home/rwest/projects/data-pipeline/specs/001-import-tick-gex-data/quickstart.md
