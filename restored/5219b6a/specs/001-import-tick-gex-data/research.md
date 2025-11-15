# Research Findings: Import Tick and GEX Data

**Date**: 2025-11-08
**Plan**: /specs/001-import-tick-gex-data/plan.md

## Research Tasks Completed

### 1. GEX Data Structure Analysis
**Task**: Research complete GEX JSON structure from legacy-source outputs to identify all fields that need to be imported

**Findings**:
  - Core fields: timestamp, ticker, spot, zero_gamma, sum_gex_vol, sum_gex_oi
  - Additional fields: min_dte, sec_min_dte, major_pos_vol, major_pos_oi, major_neg_vol, major_neg_oi, delta_risk_reversal, max_priors
  - Strike data: Array of [strike_price, vol_gex, oi_gex, priors] tuples

### 2. Parquet Storage Implementation
**Task**: Research best practices for storing large datasets in both DuckDB and Parquet formats

**Findings**:
  - Import data first to DuckDB for immediate querying and validation
  - Export validated data to Parquet files in data directory
  - Use PyArrow for Parquet operations

### 3. Data Validation Strategies
**Task**: Research validation approaches for large-scale financial data imports

**Findings**:
  - Schema validation using Pydantic models
  - Statistical validation (timestamp continuity, value ranges)
  - Cross-reference validation against source data
  - Duplicate detection and handling

### 4. Performance Optimization for Large Imports
**Task**: Research techniques for importing thousands of files with millions of records within 2-hour time limit

**Findings**:
  - File-level parallelism using concurrent.futures
  - Batch database inserts (1000-5000 records per batch)
  - Memory-efficient JSON streaming for large files
  - Progress tracking and resumable imports

### 5. Data Lineage Implementation
**Task**: Research approaches for tracking data lineage across import processes

**Findings**:
  - LineageTracker class for recording import operations
  - Metadata storage linking source files to database records
  - Import timestamps and success/failure status tracking

## Resolved Technical Unknowns

All technical unknowns from the implementation plan have been resolved:


## File Format Documentation

### MarketDepth Data Files (.depth)

**File Location**: `/mnt/c/SierraChart/Data/MarketDepthData/[CONTRACT].[YYYY-MM-DD].depth`

**Purpose**: Contains incremental market depth updates for order book reconstruction.

**File Structure**:
  - `FileTypeUniqueHeaderID`: "SCDD" (0x44444353)
  - `HeaderSize`: Size of header in bytes
  - `RecordSize`: Size of each record (24 bytes)
  - `Version`: File format version
  - `Reserve[48]`: Unused padding

  - `DateTime`: SCDateTimeMS (64-bit microseconds since Dec 30, 1899 UTC)
  - `Command`: Operation type (1-7)
    - `COMMAND_CLEAR_BOOK = 1`: Clear entire order book
    - `COMMAND_ADD_BID_LEVEL = 2`: Add bid price level
    - `COMMAND_ADD_ASK_LEVEL = 3`: Add ask price level
    - `COMMAND_MODIFY_BID_LEVEL = 4`: Modify existing bid level
    - `COMMAND_MODIFY_ASK_LEVEL = 5`: Modify existing ask level
    - `COMMAND_DELETE_BID_LEVEL = 6`: Remove bid level
    - `COMMAND_DELETE_ASK_LEVEL = 7`: Remove ask level
  - `Flags`: Bit flags (e.g., `FLAG_END_OF_BATCH = 0x01`)
  - `NumOrders`: Number of orders at this price level
  - `Price`: Price level (float32)
  - `Quantity`: Total quantity at price level (uint32)
  - `Reserved`: Unused padding (uint32)

**Key Characteristics**:

### SCID Intraday Data Files (.scid)

**File Location**: `/mnt/c/SierraChart/Data/[SYMBOL].scid`

**Purpose**: Contains time-series price and volume data for intraday charts.

**File Structure**:
  - `FileTypeUniqueHeaderID[4]`: "SCID"
  - `HeaderSize`: Size of header in bytes
  - `RecordSize`: Size of each record (40 bytes)
  - `Version`: File format version (currently 1)
  - `Unused1`: Not used
  - `UTCStartIndex`: Should be 0
  - `Reserve[36]`: Unused padding

  - `DateTime`: SCDateTimeMS (64-bit microseconds since Dec 30, 1899 UTC)
  - `Open`: Opening price (float32)
  - `High`: High price (float32)
  - `Low`: Low price (float32)
  - `Close`: Closing/trade price (float32)
  - `NumTrades`: Number of trades in period (uint32)
  - `TotalVolume`: Total volume (uint32)
  - `BidVolume`: Volume at bid or lower (uint32)
  - `AskVolume`: Volume at ask or higher (uint32)

**Key Characteristics**:
  - `Open = 0`: Indicates single trade with bid/ask data
  - `High = Ask price`: Bid/ask prices stored in High/Low
  - `Low = Bid price`: Bid/ask prices stored in High/Low
  - `Close = Trade price`: Actual transaction price
  - `FIRST_SUB_TRADE_OF_UNBUNDLED_TRADE = -1.99900095e+37`
  - `LAST_SUB_TRADE_OF_UNBUNDLED_TRADE = -1.99900197e+37`

**References**:

## Next Steps

<parameter name="filePath">/home/rwest/projects/data-pipeline/specs/001-import-tick-gex-data/research.md