# Data Model: MNQ Historical Data Extraction

## Entities

### Tick Record
Represents individual trade ticks extracted from SCID files.

**Fields:**
- `timestamp`: datetime (UTC) - Trade timestamp with millisecond precision
- `price`: float - Trade price
- `volume`: int - Trade volume
- `tick_type`: str - Trade direction (buy/sell if available)

**Validation Rules:**
- Timestamp must be within the 70-day extraction window
- Price must be positive float
- Volume must be positive integer
- Records must be unique by timestamp

**Relationships:**
- Linked to Depth Snapshot by timestamp (optional, for synchronized analysis)

### Depth Snapshot
Represents market depth at a point in time, stored in Parquet format.

**Fields:**
- `timestamp`: datetime (UTC) - Snapshot timestamp
- `bid_price_1`: float - Best bid price
- `bid_size_1`: int - Best bid size
- `ask_price_1`: float - Best ask price
- `ask_size_1`: int - Best ask size
- `bid_price_2`: float - Second best bid price (optional)
- `bid_size_2`: int - Second best bid size (optional)
- ... (up to available depth levels)

**Validation Rules:**
- Timestamp must be within the 70-day extraction window
- Bid prices must be less than ask prices
- All sizes must be non-negative integers
- Records must be unique by timestamp

**Relationships:**
- Linked to Tick Record by timestamp (optional, for synchronized analysis)

### SCID File
Source data file containing compressed tick and depth data.

**Fields:**
- `file_path`: str - Absolute path to SCID file
- `symbol`: str - Trading symbol (MNQ)
- `date_range`: tuple(datetime, datetime) - Date range covered by file

**Validation Rules:**
- File must exist and be readable
- File size must be divisible by record size (40 bytes)
- Header must match SCID format

### Parquet File
Columnar storage for depth data analytics.

**Fields:**
- `file_path`: str - Path to Parquet file
- `symbol`: str - Trading symbol
- `date_range`: tuple(datetime, datetime) - Date range covered
- `record_count`: int - Number of depth snapshots

**Validation Rules:**
- File must be valid Parquet format
- Schema must match depth snapshot structure

### Tick Parquet File
Columnar storage for tick data analytics.

**Fields:**
- `file_path`: str - Path to Parquet file
- `symbol`: str - Trading symbol
- `date_range`: tuple(datetime, datetime) - Date range covered
- `record_count`: int - Number of tick records

**Validation Rules:**
- File must be valid Parquet format
- Schema must match tick record structure

**Relationships:**
- Each tick Parquet file stores Tick Records for a single day
- Metadata table in DuckDB links tick Parquet files to record counts and date

### Metadata (DuckDB)
Metadata table links Parquet files and tick/depth counts for each day.

**Fields:**
- `date`: date
- `tick_count`: int
- `depth_count`: int
- `parquet_path`: str

**Validation Rules:**
- Counts must match actual records written
- Parquet path must exist after extraction

**Relationships:**
- Each metadata record links a Parquet file to its extraction date and record counts

## Data Flow

1. SCID files are parsed using custom binary logic
2. Tick records are inserted directly into DuckDB
3. Depth snapshots are collected and written to Parquet files
4. Metadata about Parquet files is stored in DuckDB for indexing
5. Data integrity validation runs after extraction

## State Transitions

- **Raw SCID**: Initial state of source files
- **Parsed Records**: Extracted tick and depth data in memory
- **Validated Data**: Records passing integrity checks
- **Stored Data**: Persisted in DuckDB and Parquet

## Performance Considerations

- Expected volume: 100,000-500,000 records per day
- Storage: DuckDB for fast time-series queries, Parquet for analytical depth data
- Indexing: Timestamp-based indexing in DuckDB for efficient range queries