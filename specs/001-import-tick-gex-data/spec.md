# Feature Specification: Import GEX Data

**Feature Branch**: `001-import-tick-gex-data`
**Created**: 2025-11-08
**Status**: Draft
**Input**: User description: "import tick and gex data from previous implementation, scanning folder ../legacy-source/outputs. Piece together as much information as possible and migrate to project. There are several SQLite databases as well. We want the baseline tick data, full gex_zero data for NQ_NDX and market depth data for MNQ to be used with MNQ and NQ"

**Updated**: 2025-11-08 - Scope narrowed to focus on GEX data import only from projects/legacy-source/outputs/gex_bridge/history. Tick and depth data import deferred until data completeness is verified.

## Clarifications

### Session 2025-11-08

- Q: How should duplicate records be handled during the import process? → A: Skip duplicates after verification. Verification of tick data via sampling /mnt/c/SierraChart/Data SCID files.
- User provided: all fields will be imported from market data unless agreed upon, do not filter gex_zero fields for import or live data.
- Q: What are the expected data volumes for the import (approximate number of files and total records)? → A: Thousands of files, millions of records
- Q: Are there any compliance or regulatory requirements for the imported market data? → A: None - Standard market data
- Q: Should GEX import include all available fields from source data or only a filtered subset? → A: Import all available fields from GEX JSON files (including min_dte, major_pos_vol, sum_gex_oi, delta_risk_reversal, max_priors, etc.)
- Q: How should imported GEX data be stored (DuckDB only, Parquet only, or both)? → A: Store in both DuckDB database and Parquet files in data directory
- Q: When should data be stored in each format (DuckDB vs Parquet)? → A: Import data first to DuckDB for immediate querying and validation, then export to Parquet files in data directory for long-term storage

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Import GEX Data for NQ_NDX (Priority: P1)

A data engineer imports comprehensive GEX (Gamma Exposure) data for NQ_NDX from the previous legacy-source implementation to enable gamma-based trading analysis.

**Why this priority**: GEX data is essential for understanding market positioning and making informed trading decisions for futures contracts.

**Independent Test**: Can be fully tested by verifying GEX data import, data integrity validation, and successful querying of gamma exposure metrics.

**Acceptance Scenarios**:

1. **Given** GEX data files exist in projects/legacy-source/outputs/gex_bridge/history/, **When** the import process runs, **Then** all GEX snapshots are successfully imported with correct timestamps and gamma values.
2. **Given** imported GEX data, **When** querying for a specific time range, **Then** complete gamma exposure data is returned without gaps.
3. **Given** GEX strike data, **When** analyzing gamma distribution, **Then** all strike prices and corresponding gamma values are accurately preserved.

---

### User Story 2 - Import Baseline Tick Data (Priority: Deferred - P2)

**Status**: Deferred - Tick data sources are incomplete and require verification before import.

Baseline tick data exists in ../legacy-source/outputs as well

**Why this priority**: Historical tick data provides the foundation for backtesting and validating trading strategies.

**Independent Test**: Can be fully tested by verifying tick data import, timestamp continuity, and successful querying of price/volume data.

**Acceptance Scenarios**:

1. **Given** CSV tick data files exist in ../legacy-source/outputs/data/raw/tastytrade_stream/, **When** the import process runs, **Then** all tick records for MNQ and NQ are successfully imported.
2. **Given** imported tick data, **When** querying for price movements, **Then** bid/ask/last prices and volumes are accurately preserved.
3. **Given** tick data time series, **When** checking for gaps, **Then** no significant data gaps are found in the imported dataset.

---

### User Story 3 - Import Market Depth Data for MNQ (Priority: Deferred - P3)

**Status**: Deferred - Market depth data sources are incomplete and require verification before import.

A data engineer imports market depth data specifically for MNQ futures to support order book analysis and market microstructure studies.

**Why this priority**: Market depth data provides insights into liquidity and order flow patterns for MNQ trading.

**Independent Test**: Can be fully tested by verifying depth data import and successful querying of bid/ask levels.

**Acceptance Scenarios**:

1. **Given** market depth JSONL files exist, **When** the import process runs, **Then** MNQ depth data is successfully imported with all bid/ask levels.
2. **Given** imported depth data, **When** analyzing liquidity, **Then** order book depth and spread information is available.
3. **Given** depth data over time, **When** studying market impact, **Then** order book changes are properly timestamped and sequenced.

---

### Edge Cases

- What happens when GEX data files are corrupted or incomplete?  They can be downloaded again using the /gex_history_url server
- How does system handle duplicate tick data from multiple sources? (Skip after verification)
- What occurs if market depth data has timestamp inconsistencies? They should be compared and reviewed, but may be normalized to 1s
- How are SQLite database schema changes handled during migration? Not sure, once sets of data are migrated and verified and duplicates in particular can be deleted from the source
- What happens when target storage runs out of space during import? Run the data migration in daily sets, once verified, delete from source files and database

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST scan projects/legacy-source/outputs/gex_bridge/history folder and identify all relevant GEX data files (GEX JSON files for NQ_NDX)
- **FR-002**: System MUST import all fields from complete gex_zero data for NQ_NDX from JSON files, including timestamps, spot prices, gamma values, strike data, min_dte, sec_min_dte, major_pos_vol, major_pos_oi, major_neg_vol, major_neg_oi, sum_gex_vol, sum_gex_oi, delta_risk_reversal, max_priors, and all other available fields (do not filter gex_zero fields)
- **FR-003**: [DEFERRED] System MUST import all fields from baseline tick data for MNQ and NQ from TastyTrade CSV streams, including bid/ask/last prices and volumes
- **FR-004**: [DEFERRED] System MUST import all fields from market depth data for MNQ from JSONL files preserving order book structure
- **FR-005**: [DEFERRED] System MUST migrate relevant data from SQLite databases (gex_history.db, trade_history.db) preserving schema and relationships
- **FR-006**: System MUST validate GEX data integrity during import checking for duplicates, gaps, and corruption, and skip verified duplicates
- **FR-007**: System MUST provide progress reporting and error handling for GEX data imports
- **FR-008**: System MUST store imported GEX data in both DuckDB database (for fast querying) and Parquet files in the data directory (for long-term storage and analytics), importing first to DuckDB for immediate querying and validation, then exporting to Parquet files
- **FR-009**: System MUST create data lineage records linking imported GEX data to source files and timestamps
- **FR-010**: [DEFERRED] System MUST verify tick data integrity via sampling against /mnt/c/SierraChart/Data SCID files before import

### Data Volume Assumptions

- Expected data volumes: Thousands of files, millions of records

### Non-Functional Requirements

- Compliance: None - Standard market data

### Key Entities *(include if feature involves data)*

- **GEX Snapshot**: Gamma exposure data with timestamp, ticker, spot_price, zero_gamma, net_gex, strike data, min_dte, sec_min_dte, major_pos_vol, major_pos_oi, major_neg_vol, major_neg_oi, sum_gex_vol, sum_gex_oi, delta_risk_reversal, max_priors, and all other available fields
- **Data Lineage**: Import metadata linking source GEX files to imported records

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Successfully import 100% of NQ_NDX GEX data files without data loss or corruption
- **SC-002**: [DEFERRED] Import tick data for MNQ and NQ covering at least 30 days of trading activity
- **SC-003**: [DEFERRED] Successfully migrate all relevant SQLite database records preserving relationships
- **SC-004**: GEX data integrity checks pass with <0.1% error rate
- **SC-005**: GEX import process completes within 2 hours for full dataset migration
- **SC-006**: All imported GEX data is queryable through the project's API within 10ms response times
