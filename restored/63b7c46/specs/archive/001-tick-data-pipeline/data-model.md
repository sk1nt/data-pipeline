# Data Model: Financial Tick Data Pipeline

**Feature**: Financial Tick Data Pipeline  
**Date**: 2025-11-07  
**Purpose**: Define data entities, relationships, and validation rules for the tick data pipeline system.

## Entities

### Tick Data
Represents individual price/volume updates from financial markets.

**Fields**:
- `symbol` (string, required): Financial instrument identifier (e.g., "AAPL", "BTC/USD")
- `timestamp` (datetime, required): When the tick occurred, microsecond precision
- `price` (decimal, required): Trade price or bid/ask price
- `volume` (integer, optional): Trade volume or order size
- `tick_type` (enum: trade, bid, ask, required): Type of market data
- `source` (string, required): Data source identifier (sierra_chart, gexbot, tastyttrade)

**Validation Rules**:
- Timestamp must be within last 24 hours for real-time data
- Price must be positive decimal
- Volume must be non-negative integer
- Symbol must match known financial instruments
- No duplicate timestamps per symbol within same source

**Relationships**:
- Belongs to Data Source
- Used in Enriched Data aggregation

### Enriched Data
Aggregated and processed tick data at various time intervals (1s to 4h).

**Fields**:
- `symbol` (string, required): Financial instrument identifier
- `interval_start` (datetime, required): Start of aggregation interval
- `interval_end` (datetime, required): End of aggregation interval
- `open_price` (decimal, required): First price in interval
- `high_price` (decimal, required): Highest price in interval
- `low_price` (decimal, required): Lowest price in interval
- `close_price` (decimal, required): Last price in interval
- `total_volume` (integer, required): Sum of volumes in interval
- `vwap` (decimal, optional): Volume-weighted average price

**Validation Rules**:
- Interval must be valid (1s, 1m, 1h, 4h)
- OHLC prices must satisfy: low <= open, close <= high
- Volume must be non-negative
- No gaps in time series for each symbol

**Relationships**:
- Aggregated from Tick Data
- Queried by AI Models

### AI Model
Represents external AI systems that query the pipeline.

**Fields**:
- `model_id` (string, required): Unique identifier for the AI model
- `name` (string, optional): Human-readable name
- `access_permissions` (json, required): Allowed query types and data scopes
- `api_key_hash` (string, required): Hashed API key for authentication
- `created_at` (datetime, required): Registration timestamp
- `last_access` (datetime, optional): Last query timestamp
- `query_count` (integer, default 0): Total queries made

**Validation Rules**:
- model_id must be unique
- access_permissions must include allowed symbols and query types
- api_key_hash must be valid SHA-256 hash

**Relationships**:
- Makes Queries
- Has Query History

### Query History
Audit log of AI model queries.

**Fields**:
- `query_id` (uuid, required): Unique query identifier
- `model_id` (string, required): AI model that made the query
- `query_type` (enum: realtime, historical, required): Type of query
- `parameters` (json, required): Query parameters (symbols, time range, etc.)
- `timestamp` (datetime, required): When query was made
- `response_time_ms` (integer, required): Query execution time
- `data_points_returned` (integer, required): Number of data points in response

**Validation Rules**:
- response_time_ms must be positive
- data_points_returned must be non-negative
- timestamp must be current or recent

**Relationships**:
- Belongs to AI Model

### Data Source
Represents external data providers.

**Fields**:
- `source_id` (string, required): Unique source identifier
- `name` (string, required): Human-readable name
- `type` (enum: sierra_chart, gexbot, tastyttrade, required): Source type
- `status` (enum: active, inactive, error, required): Current operational status
- `last_update` (datetime, optional): Last successful data ingestion
- `error_count` (integer, default 0): Consecutive error count

**Validation Rules**:
- source_id must be unique
- status must reflect actual connectivity

**Relationships**:
- Provides Tick Data

### Service Status
Operational status of pipeline components.

**Fields**:
- `service_name` (string, required): Component name (ingestion, processing, api, ui)
- `current_status` (enum: healthy, degraded, down, required): Current health status
- `last_update_time` (datetime, required): Last status check
- `uptime_percentage` (decimal, optional): Uptime in last 24 hours
- `error_message` (string, optional): Last error description

**Validation Rules**:
- last_update_time must be recent (< 5 minutes old)
- uptime_percentage must be between 0 and 100

**Relationships**:
- Monitored by UI

## State Transitions

### Tick Data Lifecycle
1. **Ingested**: Raw data from sources, validated for basic integrity
2. **Processed**: Filtered for quality, enriched with metadata
3. **Cached**: Stored in Redis for configurable time (default 1 hour)
4. **Archived**: Compressed to DuckDB/Parquet for long-term storage

### AI Model Access
1. **Registered**: Model registered with API key and permissions
2. **Authenticated**: API key validated on each request
3. **Authorized**: Permissions checked against requested data
4. **Queried**: Data retrieved and logged
5. **Rate Limited**: If quota exceeded, temporarily blocked

## Data Integrity Constraints

- No duplicate ticks within same symbol/source/timestamp
- Time series continuity maintained (gap detection runs daily)
- Referential integrity between entities
- Audit trail for all AI queries
- Automatic cleanup of expired cache data