# Feature Specification: Extract MNQ Historical Data

**Feature Branch**: `002-extract-mnq-data`  
**Created**: 2025-11-08  
**Status**: Draft  
**Input**: User description: "Add a historical tick extraction method for MNQ from the SierraChart Data folder that extracts 70 days back and writes directly to duckdb from SCID file, it should also pull market depth for the same timeline."

## Clarifications

### Session 2025-11-08

- Q: How should market depth data be stored? → A: MarketDepthData should be stored in parquet files similar to gex with metadata in duckdb
- Q: How will users trigger the data extraction process? → A: Command-line interface (CLI) script
- Q: What method/library will be used to parse SCID files? → A: Custom Python parsing of SCID binary format
- Q: What are the uniqueness constraints for tick records and depth snapshots? → A: Unique by timestamp - Records deduplicated based on timestamp
- Q: What timezone are the timestamps in SCID files, and how should they be handled? → A: UTC timezone - Timestamps are in UTC and stored as-is
- Q: What is the expected daily data volume for MNQ tick and depth data? → A: 100,000-500,000 records per day - Based on typical futures trading volume, but tick records and depth snapshots contain different information, one is spot and the other is flow data.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Extract MNQ Tick Data (Priority: P1)

Data analyst needs to extract historical tick data for MNQ futures from SierraChart SCID files to analyze market microstructure and trading patterns.

**Why this priority**: This is the core functionality requested - extracting tick data is essential for market analysis.

**Independent Test**: Can be fully tested by verifying tick data is extracted from SCID files and stored in DuckDB with correct timestamps, prices, and volumes.

**Acceptance Scenarios**:

1. **Given** SierraChart Data folder contains MNQ SCID files, **When** extraction runs for 70 days back, **Then** tick data is written to DuckDB with timestamp, price, volume, and tick type fields
2. **Given** SCID file has corrupted data, **When** extraction runs, **Then** system logs errors and continues processing other files
3. **Given** extraction completes successfully, **When** querying DuckDB, **Then** tick data shows continuous time series without gaps

---

### User Story 2 - Extract MNQ Market Depth (Priority: P1)

Data analyst needs to extract market depth data for MNQ futures from SierraChart for the same 70-day period to analyze order book dynamics.

**Why this priority**: Market depth is equally important for comprehensive market analysis and was specifically requested alongside tick data.

**Independent Test**: Can be fully tested by verifying depth data is extracted and stored in DuckDB with bid/ask levels, sizes, and timestamps matching the tick data timeline.

**Acceptance Scenarios**:

1. **Given** SierraChart Data folder contains depth data, **When** extraction runs for 70 days back, **Then** market depth snapshots are written to Parquet files with metadata in DuckDB, with timestamp, bid prices/sizes, ask prices/sizes
2. **Given** depth data extraction completes, **When** querying DuckDB metadata, **Then** depth data timestamps align with tick data timestamps
3. **Given** depth data has missing levels, **When** extraction runs, **Then** system handles variable depth levels appropriately

---

### User Story 3 - Data Integrity Validation (Priority: P2)

Data analyst needs assurance that extracted data maintains integrity and can be used for reliable analysis.

**Why this priority**: Data quality is critical for analysis but secondary to the core extraction functionality.

**Independent Test**: Can be fully tested by running validation checks on extracted data for completeness, accuracy, and consistency.

**Acceptance Scenarios**:

1. **Given** extraction completes, **When** validation runs, **Then** all records have valid timestamps within the 70-day window
2. **Given** extracted data, **When** checking for duplicates, **Then** no duplicate records exist based on timestamp and price
3. **Given** tick and depth data, **When** cross-referencing timestamps, **Then** data is synchronized within acceptable time tolerance

### Edge Cases

- What happens when SCID files are missing for some dates in the 70-day period?
- How does system handle SCID files with different data formats or versions?
- What happens when disk space is insufficient for the extracted data?
- Timestamps are assumed to be in UTC; no timezone conversion needed
- What happens when SierraChart Data folder path is invalid or inaccessible?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST extract tick data from MNQ SCID files in SierraChart Data folder
- **FR-002**: System MUST extract data for exactly 70 days back from current date
- **FR-003**: System MUST write tick data directly to DuckDB with fields: timestamp, price, volume, tick_type
- **FR-004**: System MUST extract market depth data for the same 70-day timeline
- **FR-005**: System MUST write depth data to Parquet files with metadata in DuckDB, similar to GEX strike data storage
- **FR-006**: System MUST handle SCID file parsing errors gracefully without stopping extraction
- **FR-007**: System MUST validate data integrity after extraction
- **FR-008**: System MUST log extraction progress and any errors encountered
- **FR-009**: System MUST provide a CLI script to trigger the extraction process
- **FR-010**: System MUST implement custom Python binary parsing logic for SCID files
- **FR-011**: System MUST handle timestamps in UTC timezone

### Key Entities *(include if feature involves data)*

- **Tick Record**: Represents individual trade ticks with timestamp, price, volume, and trade direction; unique by timestamp
- **Depth Snapshot**: Represents market depth at a point in time with multiple bid/ask levels; unique by timestamp
- **SCID File**: SierraChart Intraday Data file containing compressed tick and depth data
- **Parquet File**: Columnar storage format for depth data, similar to GEX strike data storage
- **Data Volume**: Expected 100,000-500,000 records per day for MNQ tick and depth data

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: System extracts tick data for 70 consecutive days with less than 1% data loss
- **SC-002**: System extracts depth data synchronized with tick data timestamps within 1-second tolerance
- **SC-003**: Extraction process completes within 30 minutes for 70 days of data
- **SC-004**: Extracted data passes integrity validation with 99.9% accuracy
- **SC-005**: System handles file parsing errors without crashing and processes 95% of available files successfully
