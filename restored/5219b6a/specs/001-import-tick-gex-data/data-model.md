# Data Model: Import GEX Data

**Date**: 2025-11-08
**Plan**: /specs/001-import-tick-gex-data/plan.md

## Overview

The data model supports importing and storing comprehensive GEX (Gamma Exposure) data for NQ_NDX futures contracts. All data is stored in both DuckDB for fast querying and Parquet files for long-term analytics. Tick and market depth data models are deferred until data completeness is verified.

## Core Entities

### GEXSnapshot
Represents a complete gamma exposure snapshot for a futures contract at a specific point in time.

**Fields**:
- `timestamp` (datetime): When the snapshot was taken (required, indexed)
- `ticker` (string): Futures contract symbol, e.g., "NQ_NDX" (required, indexed)
- `spot_price` (float): Underlying spot price (required, > 0)
- `zero_gamma` (float): Zero gamma value (required)
- `net_gex` (float): Net gamma exposure calculated from volume GEX (required)
- `min_dte` (integer): Minimum days to expiration (optional)
- `sec_min_dte` (integer): Secondary minimum days to expiration (optional)
- `major_pos_vol` (float): Major positive volume gamma (optional)
- `major_pos_oi` (float): Major positive open interest gamma (optional)
- `major_neg_vol` (float): Major negative volume gamma (optional)
- `major_neg_oi` (float): Major negative open interest gamma (optional)
- `sum_gex_vol` (float): Sum of volume-based gamma exposure (optional)
- `sum_gex_oi` (float): Sum of open interest gamma exposure (optional)
- `delta_risk_reversal` (float): Delta risk reversal metric (optional)
- `max_priors` (array): Maximum prior values by expiration bucket (optional)
- `strike_data` (array): Array of strike information (required, 1+ items)

**Relationships**:
- One-to-many with StrikeData entities
- Referenced by DataLineage records

**Validation Rules**:
- Timestamp must be valid datetime in past
- Ticker must be valid futures symbol (NQ_NDX, etc.)
- Spot price must be positive
- Strike data array must contain at least one strike
- All gamma values must be numeric (can be negative)

**Indexes**:
- Primary: (timestamp, ticker)
- Secondary: timestamp (for time range queries)
- Secondary: ticker (for symbol filtering)

### StrikeData
Represents gamma exposure data for a specific strike price within a GEX snapshot.

**Fields**:
- `strike` (float): Strike price (required, > 0)
- `gamma` (float): Volume-based gamma exposure at this strike (required)
- `oi_gamma` (float): Open interest gamma exposure at this strike (required)
- `priors` (array): Prior values for this strike (optional, defaults to [0,0,0,0,0])

**Relationships**:
- Belongs to GEXSnapshot
- No direct external references

**Validation Rules**:
- Strike price must be positive
- Gamma values must be numeric
- Priors array must contain exactly 5 numeric values if present

### DataLineage
Tracks the origin and processing history of imported data records.

**Fields**:
- `import_id` (string): Unique identifier for the import operation (required, primary key)
- `source_file` (string): Path to the source file (required)
- `record_type` (string): Type of data imported (gex) (required)
- `records_imported` (integer): Number of records successfully imported (required)
- `import_timestamp` (datetime): When the import occurred (required)
- `status` (string): Import status (success, partial, failed) (required)
- `errors` (array): List of error messages if any (optional)

**Relationships**:
- References imported records by import_id

**Validation Rules**:
- Import ID must be unique
- Record type must be valid enum value
- Records imported must be >= 0
- Status must be valid enum value

**Indexes**:
- Primary: import_id
- Secondary: import_timestamp (for chronological queries)
- Secondary: record_type (for filtering by data type)

## Data Flow

1. **Import Phase**: Raw data files → Validation → DuckDB storage
2. **Export Phase**: DuckDB data → Validation → Parquet files in data directory
3. **Query Phase**: Applications query DuckDB for real-time access, Parquet for analytics

## Storage Strategy

- **DuckDB**: Primary storage for fast querying during import and validation
- **Parquet**: Long-term storage in data directory for analytics and archival
- **Synchronization**: Data flows from DuckDB to Parquet after successful validation

## Performance Considerations

- All timestamp fields indexed for efficient time range queries
- Symbol fields indexed for efficient filtering
- Strike data stored as JSON strings in DuckDB for flexibility
- Parquet files partitioned by date and symbol for optimal query performance</content>
<parameter name="filePath">/home/rwest/projects/data-pipeline/specs/001-import-tick-gex-data/data-model.md