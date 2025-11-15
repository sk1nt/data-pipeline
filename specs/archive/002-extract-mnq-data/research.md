# Research Findings: MNQ Historical Data Extraction

## SCID File Format and Parsing

**Decision**: Use custom Python binary parsing with struct module for SCID files  
**Rationale**: SCID uses a proprietary binary format with no existing Python libraries. The struct module provides efficient, low-level binary parsing suitable for high-volume tick data processing.  
**Alternatives Considered**: 
- Third-party libraries (none available for SCID)
- Export to CSV first (adds complexity and potential data loss)
- SierraChart API (not suitable for historical batch extraction)

## Tick Data Handling

**Decision**: Parse individual trades and handle bundled trades (CME repeating groups)  
**Rationale**: MNQ data includes both single trades and bundled groups. Proper handling ensures complete data extraction with accurate volume aggregation.  
**Alternatives Considered**: 
- Skip bundled trades (would lose significant volume data)
- Treat bundles as single records (incorrect volume reporting)

## Timestamp Processing

**Decision**: Convert SCID timestamps to UTC datetime with millisecond precision and counter  
**Rationale**: SCID timestamps include microseconds since 1899-12-30 epoch plus 3-digit counter for sub-millisecond trades. Full precision needed for accurate sequencing.  
**Alternatives Considered**: 
- Truncate to seconds (loses precision for high-frequency data)
- Ignore counter (breaks ordering for rapid trades)

## Market Depth Data

**Decision**: Parse separate .scdd files for depth data with incremental updates  
**Rationale**: Depth data is stored in companion .scdd files with ADD/MODIFY/DELETE commands. Incremental parsing ensures efficient storage of full order book snapshots.  
**Alternatives Considered**: 
- Store only full snapshots (wasteful for unchanged levels)
- Ignore depth data (misses critical order book information)

## Data Storage Strategy

**Decision**: Store tick data in DuckDB, depth data in Parquet with metadata in DuckDB  
**Rationale**: DuckDB excels at time-series queries for ticks. Parquet provides columnar compression for analytical depth data, with DuckDB metadata for indexing.  
**Alternatives Considered**: 
- Store all data in DuckDB (less efficient for depth analytics)
- Store all data in Parquet (slower for time-series queries)

## Performance Optimization

**Decision**: Process files sequentially with memory-efficient streaming  
**Rationale**: MNQ data volumes (100k-500k records/day) require streaming to avoid memory issues. Sequential processing maintains data integrity.  
**Alternatives Considered**: 
- Load entire files into memory (fails for large datasets)
- Parallel processing (risks data ordering issues)

## Error Handling

**Decision**: Log errors and continue processing for corrupted records/files  
**Rationale**: Data quality issues are common in financial data. Graceful error handling ensures maximum data extraction while maintaining process reliability.  
**Alternatives Considered**: 
- Stop on first error (would fail entire extraction)
- Skip entire files (loses potentially good data)