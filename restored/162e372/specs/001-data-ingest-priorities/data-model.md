# Data Model: High-Speed GEX Data Ingest Priorities

**Feature**: High-Speed GEX Data Ingest Priorities with Firm Guidelines and Methodology
**Date**: November 9, 2025
**Version**: 1.0

## Overview

This document defines the data structures, relationships, and validation rules for the high-speed GEX data ingest priority system. The model supports automatic priority assignment, real-time processing guarantees, and comprehensive monitoring.

## Core Entities

### PriorityRequest

Represents a request to ingest GEX data with priority assignment.

**Fields**:
- `request_id`: UUID - Unique identifier for the request
- `source_url`: str - URL of the GEX data source
- `data_type`: GEXDataType - Type of GEX data (historical, real-time, snapshot)
- `market_symbol`: str - Underlying market symbol (e.g., "NDX", "SPX")
- `requested_at`: datetime - Timestamp when request was submitted
- `priority_score`: float - Calculated priority score (0.0-1.0)
- `priority_level`: PriorityLevel - Assigned priority level (CRITICAL, HIGH, MEDIUM, LOW)
- `estimated_processing_time`: timedelta - Expected processing duration
- `metadata`: dict - Additional request metadata

**Validation Rules**:
- `source_url` must be valid HTTP/HTTPS URL
- `priority_score` must be between 0.0 and 1.0
- `requested_at` cannot be in the future
- `market_symbol` must be valid optionable symbol

**Relationships**:
- One-to-many with ProcessingJob
- Many-to-one with DataSource

### ProcessingJob

Represents an active or completed data processing job.

**Fields**:
- `job_id`: UUID - Unique job identifier
- `request_id`: UUID - Associated priority request
- `status`: JobStatus - Current job status (QUEUED, PROCESSING, COMPLETED, FAILED)
- `started_at`: datetime - When processing began
- `completed_at`: datetime - When processing finished
- `processing_duration`: timedelta - Actual processing time
- `records_processed`: int - Number of GEX records processed
- `data_size_bytes`: int - Size of processed data
- `error_message`: str - Error details if failed
- `retry_count`: int - Number of retry attempts

**Validation Rules**:
- `started_at` must be after `requested_at` from PriorityRequest
- `completed_at` must be after `started_at` if set
- `processing_duration` calculated as `completed_at - started_at`
- `retry_count` cannot exceed maximum retry limit (3)

**State Transitions**:
- QUEUED → PROCESSING (when job starts)
- PROCESSING → COMPLETED (on success)
- PROCESSING → FAILED (on error, with retry logic)
- FAILED → QUEUED (on retry)

### DataSource

Represents a GEX data source with reliability metrics.

**Fields**:
- `source_id`: UUID - Unique source identifier
- `base_url`: str - Base URL for the data source
- `name`: str - Human-readable source name
- `reliability_score`: float - Historical reliability (0.0-1.0)
- `last_successful_fetch`: datetime - Last successful data fetch
- `total_requests`: int - Total requests made
- `successful_requests`: int - Successful requests
- `average_response_time`: timedelta - Average response time
- `is_active`: bool - Whether source is currently active

**Validation Rules**:
- `reliability_score` = `successful_requests / total_requests`
- `base_url` must be valid base URL
- `average_response_time` updated with exponential moving average

**Relationships**:
- One-to-many with PriorityRequest

### PriorityRule

Defines rules for automatic priority assignment.

**Fields**:
- `rule_id`: UUID - Unique rule identifier
- `name`: str - Human-readable rule name
- `description`: str - Rule description
- `condition`: str - Python expression for rule condition
- `priority_score`: float - Score assigned when condition met
- `is_active`: bool - Whether rule is currently active
- `created_at`: datetime - When rule was created
- `updated_at`: datetime - When rule was last updated

**Validation Rules**:
- `condition` must be valid Python expression
- `priority_score` between 0.0 and 1.0
- `updated_at` >= `created_at`

**Examples**:
```python
# High priority for current day data
condition: "data_type == 'real_time' and market_symbol in ['NDX', 'SPX']"
priority_score: 0.9

# Medium priority for high volume strikes
condition: "open_interest > 10000 and volume > 5000"
priority_score: 0.7
```

### GEXSnapshot

Represents a processed GEX data snapshot.

**Fields**:
- `snapshot_id`: UUID - Unique snapshot identifier
- `job_id`: UUID - Processing job that created this snapshot
- `market_symbol`: str - Underlying symbol
- `snapshot_date`: date - Date of the snapshot
- `snapshot_time`: time - Time of the snapshot
- `strikes`: List[GEXStrike] - GEX strike data
- `total_open_interest`: int - Sum of all open interest
- `total_volume`: int - Sum of all volume
- `processed_at`: datetime - When snapshot was processed
- `data_quality_score`: float - Quality assessment (0.0-1.0)

**Validation Rules**:
- `strikes` cannot be empty
- `snapshot_date` cannot be in the future
- `data_quality_score` based on data completeness and consistency

### GEXStrike

Individual strike data within a snapshot.

**Fields**:
- `strike_price`: float - Option strike price
- `call_open_interest`: int - Call open interest
- `put_open_interest`: int - Put open interest
- `call_volume`: int - Call volume
- `put_volume`: int - Put volume
- `call_bid`: float - Call bid price
- `call_ask`: float - Call ask price
- `put_bid`: float - Put bid price
- `put_ask`: float - Put ask price
- `gamma`: float - Gamma exposure
- `delta`: float - Delta exposure
- `theta`: float - Theta exposure
- `vega`: float - Vega exposure

**Validation Rules**:
- `strike_price` > 0
- Open interest and volume >= 0
- Bid/ask prices >= 0
- Greeks (gamma, delta, theta, vega) are valid float values

## Enums

### GEXDataType
- `HISTORICAL`: Historical GEX data
- `REAL_TIME`: Real-time streaming data
- `SNAPSHOT`: Point-in-time snapshot

### PriorityLevel
- `CRITICAL`: <30 second processing guarantee
- `HIGH`: <5 minute processing
- `MEDIUM`: <30 minute processing
- `LOW`: Best effort processing

### JobStatus
- `QUEUED`: Waiting for processing
- `PROCESSING`: Currently being processed
- `COMPLETED`: Successfully processed
- `FAILED`: Processing failed

## Data Flow

1. **PriorityRequest** created from API submission
2. **PriorityRules** evaluated to calculate `priority_score`
3. **ProcessingJob** created and queued based on priority
4. **DataSource** reliability affects priority calculation
5. **GEXSnapshot** created from successful processing
6. **GEXStrike** data stored within snapshots

## Validation Schemas

### Pydantic Models

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime, date, time
from enum import Enum

class GEXDataType(str, Enum):
    HISTORICAL = "historical"
    REAL_TIME = "real_time"
    SNAPSHOT = "snapshot"

class PriorityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class PriorityRequest(BaseModel):
    request_id: UUID
    source_url: str = Field(..., regex=r"https?://.+")
    data_type: GEXDataType
    market_symbol: str = Field(..., min_length=1, max_length=10)
    requested_at: datetime
    priority_score: float = Field(..., ge=0.0, le=1.0)
    priority_level: PriorityLevel
    estimated_processing_time: timedelta
    metadata: dict = Field(default_factory=dict)

    @validator('requested_at')
    def requested_at_not_future(cls, v):
        if v > datetime.utcnow():
            raise ValueError('requested_at cannot be in the future')
        return v
```

## Storage Strategy

### DuckDB Tables

- `priority_requests`: Core request storage with indexing on `priority_score`
- `processing_jobs`: Job tracking with foreign key to `priority_requests`
- `data_sources`: Source reliability metrics
- `priority_rules`: Active rules for priority calculation
- `gex_snapshots`: Processed snapshot metadata
- `gex_strikes`: Individual strike data (partitioned by snapshot_date)

### Redis Keys

- `priority_queue:{level}`: Sorted sets for priority-based queuing
- `job_status:{job_id}`: Job status with TTL
- `source_cache:{source_id}`: Source reliability data
- `priority_rules`: Cached active rules

### Parquet Files

- `data/gex/{symbol}/{date}/snapshot_{timestamp}.parquet`: Processed GEX data
- Partitioned by symbol and date for efficient querying

## Performance Considerations

- **Indexing**: Priority score and request time indexed for fast queue operations
- **Partitioning**: GEX data partitioned by symbol/date for parallel processing
- **Caching**: Hot data cached in Redis with appropriate TTL
- **Archiving**: Old processing jobs archived after 30 days

## Migration Strategy

1. Create new tables alongside existing schema
2. Migrate existing import jobs to new priority system
3. Update import scripts to use priority queue
4. Validate priority assignment accuracy before full deployment