# CLI Contract: MNQ Data Extraction

## Command Interface

### extract-mnq-data

Triggers the extraction of MNQ historical tick and depth data from SierraChart SCID files.

**Usage:**
```bash
python -m src.cli.extract_mnq_data [OPTIONS]
```

**Options:**
- `--scid-dir PATH`: Path to SierraChart Data directory containing SCID files (required)
- `--symbol TEXT`: Trading symbol to extract (default: MNQ)
- `--days-back INTEGER`: Number of days back to extract (default: 70)
- `--output-dir PATH`: Directory for output files (default: data/)
- `--dry-run`: Validate extraction without writing data
- `--verbose`: Enable detailed logging

**Exit Codes:**
- 0: Success
- 1: Invalid arguments
- 2: File access error
- 3: Parsing error
- 4: Storage error

## Functional Contracts

### FR-001: Extract Tick Data
**Input:** SCID file path, symbol, date range  
**Output:** Tick records inserted into DuckDB  
**Preconditions:** SCID file exists and is readable  
**Postconditions:** All valid tick records stored with unique timestamps

### FR-002: Extract Depth Data
**Input:** SCID file path, symbol, date range  
**Output:** Depth snapshots written to Parquet, metadata in DuckDB  
**Preconditions:** SCID file contains depth data  
**Postconditions:** Depth data synchronized with tick timestamps

### FR-003: CLI Trigger
**Input:** Command-line arguments  
**Output:** Extraction process execution  
**Preconditions:** Valid arguments provided  
**Postconditions:** Process completes with success/failure status

### FR-004: Error Handling
**Input:** Corrupted or missing files  
**Output:** Logged errors, continued processing  
**Preconditions:** Files may be corrupted  
**Postconditions:** Maximum data extracted despite errors

## Data Contracts

### Tick Data Schema
```json
{
  "timestamp": "datetime (UTC)",
  "price": "float",
  "volume": "integer",
  "tick_type": "string (optional)"
}
```

### Depth Data Schema
```json
{
  "timestamp": "datetime (UTC)",
  "bid_price_1": "float",
  "bid_size_1": "integer",
  "ask_price_1": "float",
  "ask_size_1": "integer",
  "...": "additional levels"
}
```

## Error Contracts

- **FileNotFoundError**: When SCID directory or files don't exist
- **ParseError**: When SCID file format is invalid
- **StorageError**: When DuckDB/Parquet write fails
- **ValidationError**: When data integrity checks fail