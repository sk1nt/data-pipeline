# Data Model: Migrate Data Pipeline Functionality

**Date**: November 9, 2025  
**Feature**: 003-data-pipeline-migration  
**Purpose**: Define the data structures and relationships for migrated data pipeline functionality

## Entities

### GEX Payload
Represents a snapshot of Gamma Exposure data for a financial instrument.

**Fields**:
- `timestamp` (datetime): Snapshot timestamp
- `ticker` (string): Financial instrument symbol (e.g., "SPX", "NQ_NDX")
- `spot_price` (float): Current spot price
- `zero_gamma` (float): Zero gamma value
- `net_gex` (float): Net gamma exposure value
- `min_dte` (integer, optional): Minimum days to expiration
- `sec_min_dte` (integer, optional): Secondary minimum days to expiration
- `major_pos_vol` (float, optional): Major positive volume strike
- `major_pos_oi` (float, optional): Major positive OI strike
- `major_neg_vol` (float, optional): Major negative volume strike
- `major_neg_oi` (float, optional): Major negative OI strike
- `sum_gex_vol` (float, optional): Sum of GEX volume
- `sum_gex_oi` (float, optional): Sum of GEX open interest
- `delta_risk_reversal` (float, optional): Delta risk reversal value
- `max_priors` (string): JSON string of prior maximum values

**Validation Rules**:
- `ticker` must be non-empty string
- `timestamp` must be valid datetime
- `spot_price`, `zero_gamma` must be valid floats

**Relationships**:
- Stored in DuckDB table `gex_snapshots`
- Strike details stored in `gex_strikes` table
- Imported from historical JSON files

### Historical Data
Collection of GEX snapshots downloaded from external URLs.

**Fields**:
- `url` (string): Source URL for download
- `ticker` (string): Associated ticker
- `endpoint` (string): Data endpoint
- `status` (string): Import status ("pending", "started", "completed", "failed")
- `attempts` (integer): Number of import attempts
- `last_error` (string, optional): Last error message
- `created_at` (string): Creation timestamp
- `updated_at` (string): Last update timestamp
- `snapshots` (array): Array of GEX payload objects

**Validation Rules**:
- `url` must be valid HTTP/HTTPS URL
- `status` must be one of allowed values
- `snapshots` must be valid array of GEX payloads

**Relationships**:
- Managed through `gex_history_queue` table
- Downloaded files staged in `data/source/gexbot/`
- Imported to DuckDB and Parquet

### Webhook Payload
Generic payload from external webhook sources.

**Fields**:
- `topic` (string): Webhook topic (e.g., "gex", "option_trades_super_algo")
- `event_type` (string): Event type
- `payload` (object): Event-specific data
- `received_at` (string): Receipt timestamp
- `source_ip` (string): Source IP address
- `user_agent` (string): User agent string

**Validation Rules**:
- `topic` must be non-empty
- `payload` must be valid JSON object

**Relationships**:
- Stored in `universal_webhooks` table
- Processed based on topic (GEX data vs. other events)

## Data Flow

1. **GEX Payloads**: Received via POST /gex, validated, stored in gex_data.db DuckDB
2. **Historical Data**: Requested via POST /gex_history_url, queued in gex_data.db, downloaded to staging, imported to gex_data.db DuckDB/Parquet
3. **Webhook Payloads**: Received via POST /uw, stored in gex_data.db DuckDB, processed by topic

## Storage

- **DuckDB gex_data.db**: Real-time data (`gex_snapshots`, `gex_strikes`, `universal_webhooks`, `option_trades_events`)
- **DuckDB gex_data.db**: History import metadata and queue (`gex_history_queue`)
- **Parquet**: Historical data exports (`data/parquet/gexbot/<ticker>/<endpoint>/<YYYYMMDD>.strikes.parquet`)
- **JSON**: Staged downloads (`data/source/gexbot/ticker_endpoint_history.json`)

## Constraints

- All timestamps stored as Unix seconds
- Financial values stored as floats with appropriate precision
- Data integrity maintained through foreign key relationships in DuckDB
- Only `data/` directory modified during operation
