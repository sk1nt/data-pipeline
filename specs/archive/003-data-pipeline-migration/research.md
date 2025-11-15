# Research Findings: Data Pipeline Migration

**Feature**: 003-data-pipeline-migration  
**Date**: November 9, 2025  
**Research Focus**: Original data-pipeline.py functionality and migration requirements

## Original Implementation Analysis

### Decision: Preserve all core functionality from data-pipeline.py
**Rationale**: The original server provides comprehensive GEX data capture, historical import queuing, and universal webhook handling. All features must be maintained to ensure identical behavior in the migrated version.

**Alternatives Considered**:
- Selective migration (only core endpoints) - Rejected because user specified "full functionality"
- Refactoring to modern async patterns - Deferred to maintain compatibility
- Removing unused features (Discord integration) - Rejected to preserve original behavior

### Server Architecture
**Decision**: Maintain ThreadingHTTPServer with custom request handler
**Rationale**: The original uses threading for concurrent request handling and background processing. This architecture supports the real-time requirements and background I/O operations.

**Alternatives Considered**:
- Switch to asyncio/FastAPI async - Rejected due to complexity of migrating threading logic
- Use standard FastAPI - Rejected because original uses custom threading server

### Database Operations
**Decision**: Migrate from SQLite to DuckDB with identical schema
**Rationale**: DuckDB provides better performance for analytical queries while maintaining SQL compatibility. Schema includes gex_bridge_snapshots, gex_bridge_strikes, gex_history_queue, universal_webhooks, and option_trades_events tables.

**Alternatives Considered**:
- Keep SQLite - Rejected due to user specification for DuckDB
- Use different schema - Rejected to maintain data compatibility

### Data Persistence Patterns
**Decision**: Preserve in-memory store + disk persistence + background threading
**Rationale**: The original uses an in-memory GEXStore for fast access, with background threads for disk I/O (JSONL files, consolidated snapshots). This pattern ensures low-latency responses while maintaining data durability.

**Alternatives Considered**:
- Pure in-memory with periodic dumps - Rejected due to real-time requirements
- Synchronous I/O - Rejected due to performance impact

### Endpoint Behavior
**Decision**: Maintain exact endpoint responses and validation logic
**Rationale**: /gex, /gex_history_url, and /uw endpoints have specific validation, processing, and response behaviors that must be preserved for compatibility.

**Alternatives Considered**:
- Simplify validation - Rejected to maintain original behavior
- Change response formats - Rejected for compatibility

### File Handling
**Decision**: Update paths but preserve file formats and atomic writes
**Rationale**: Original uses JSONL for depth events, JSON for snapshots, and specific directory structures. Atomic writes prevent corruption during concurrent access.

**Alternatives Considered**:
- Change file formats - Rejected to maintain compatibility
- Different directory structure - Rejected due to existing conventions

### Background Processing
**Decision**: Maintain DepthPersistor thread and API polling threads
**Rationale**: Background threads handle depth event buffering, file I/O, and periodic API data fetching. This ensures real-time data availability without blocking request handling.

**Alternatives Considered**:
- Use async tasks - Rejected due to threading complexity
- Remove background processing - Rejected for functionality preservation

## Technology Best Practices

### FastAPI Integration
**Decision**: Use FastAPI for the migrated server implementation
**Rationale**: FastAPI provides modern async support, automatic OpenAPI generation, and Pydantic integration, making it suitable for the API endpoints while maintaining performance.

**Alternatives Considered**:
- Keep ThreadingHTTPServer - Rejected due to modernization benefits
- Use Flask - Rejected due to less robust validation

### DuckDB Usage
**Decision**: Use DuckDB for all database operations with connection pooling
**Rationale**: DuckDB excels at analytical queries on time-series data and integrates well with Polars. Connection management prevents resource leaks.

**Alternatives Considered**:
- Use SQLAlchemy ORM - Rejected due to performance overhead
- Direct SQL - Rejected for maintainability

### Polars Integration
**Decision**: Use Polars for data processing and Parquet operations
**Rationale**: Polars provides fast DataFrame operations and native Parquet support, ideal for the financial data processing requirements.

**Alternatives Considered**:
- Pandas - Rejected due to performance advantages of Polars
- Pure Python - Rejected for data processing complexity

### Error Handling
**Decision**: Implement comprehensive error handling with retries and logging
**Rationale**: Original uses retry decorators and detailed logging. This ensures reliability in production environments.

**Alternatives Considered**:
- Basic try/except - Rejected for robustness
- External monitoring - Deferred as enhancement

### Configuration Management
**Decision**: Support environment variables and CLI arguments
**Rationale**: Original uses env vars for flexibility. This allows deployment-specific configuration.

**Alternatives Considered**:
- Config files only - Rejected for containerization compatibility
- Hardcoded values - Rejected for flexibility

## Migration Strategy

### Decision: Incremental migration with compatibility testing
**Rationale**: Migrate functionality module by module, testing each endpoint against original behavior to ensure identical operation.

**Alternatives Considered**:
- Big bang migration - Rejected due to risk
- Rewrite from scratch - Rejected due to complexity

### Directory Structure
**Decision**: Adapt to current workspace structure while preserving relative paths
**Rationale**: Use data/ for persistence, src/ for code, maintaining the original's organization within the new environment.

**Alternatives Considered**:
- Completely new structure - Rejected to minimize changes
- Keep original paths - Rejected due to workspace conventions

## Implementation Notes

- Remove legacy-specific dependencies (depth module, webhook_schemas)
- Adapt database connections to use existing DuckDB setup
- Preserve all validation logic and error responses
- Maintain background thread patterns for performance
- Update import paths and configuration defaults
- Ensure all original CLI arguments and env vars are supported