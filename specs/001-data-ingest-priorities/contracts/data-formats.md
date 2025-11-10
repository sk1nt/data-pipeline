# GEX Data Exchange Formats

**Feature**: High-Speed GEX Data Ingest Priorities
**Version**: 1.0
**Date**: November 9, 2025

## Overview

This document specifies the data exchange formats for GEX (Gamma Exposure) data used in the priority-based ingestion system. All formats are JSON-based for compatibility with existing pipeline components.

## Core Data Structures

### GEXSnapshot

Complete GEX snapshot data structure.

**JSON Schema**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["snapshot_id", "market_symbol", "snapshot_date", "snapshot_time", "strikes"],
  "properties": {
    "snapshot_id": {
      "type": "string",
      "format": "uuid",
      "description": "Unique snapshot identifier"
    },
    "market_symbol": {
      "type": "string",
      "minLength": 1,
      "maxLength": 10,
      "description": "Underlying market symbol (e.g., NDX, SPX)"
    },
    "snapshot_date": {
      "type": "string",
      "format": "date",
      "description": "Date of the snapshot (YYYY-MM-DD)"
    },
    "snapshot_time": {
      "type": "string",
      "format": "time",
      "description": "Time of the snapshot (HH:MM:SS)"
    },
    "strikes": {
      "type": "array",
      "items": { "$ref": "#/definitions/GEXStrike" },
      "minItems": 1,
      "description": "Array of strike data"
    },
    "total_open_interest": {
      "type": "integer",
      "minimum": 0,
      "description": "Sum of all call and put open interest"
    },
    "total_volume": {
      "type": "integer",
      "minimum": 0,
      "description": "Sum of all call and put volume"
    },
    "processed_at": {
      "type": "string",
      "format": "date-time",
      "description": "When this snapshot was processed"
    },
    "data_quality_score": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Quality assessment score"
    },
    "metadata": {
      "type": "object",
      "description": "Additional snapshot metadata",
      "additionalProperties": true
    }
  },
  "definitions": {
    "GEXStrike": {
      "type": "object",
      "required": ["strike_price", "call_open_interest", "put_open_interest"],
      "properties": {
        "strike_price": {
          "type": "number",
          "minimum": 0,
          "description": "Option strike price"
        },
        "call_open_interest": {
          "type": "integer",
          "minimum": 0,
          "description": "Call option open interest"
        },
        "put_open_interest": {
          "type": "integer",
          "minimum": 0,
          "description": "Put option open interest"
        },
        "call_volume": {
          "type": "integer",
          "minimum": 0,
          "description": "Call option volume"
        },
        "put_volume": {
          "type": "integer",
          "minimum": 0,
          "description": "Put option volume"
        },
        "call_bid": {
          "type": "number",
          "minimum": 0,
          "description": "Call option bid price"
        },
        "call_ask": {
          "type": "number",
          "minimum": 0,
          "description": "Call option ask price"
        },
        "put_bid": {
          "type": "number",
          "minimum": 0,
          "description": "Put option bid price"
        },
        "put_ask": {
          "type": "number",
          "minimum": 0,
          "description": "Put option ask price"
        },
        "gamma": {
          "type": "number",
          "description": "Gamma exposure for this strike"
        },
        "delta": {
          "type": "number",
          "description": "Delta exposure for this strike"
        },
        "theta": {
          "type": "number",
          "description": "Theta exposure for this strike"
        },
        "vega": {
          "type": "number",
          "description": "Vega exposure for this strike"
        }
      }
    }
  }
}
```

**Example**:
```json
{
  "snapshot_id": "123e4567-e89b-12d3-a456-426614174000",
  "market_symbol": "NDX",
  "snapshot_date": "2025-11-09",
  "snapshot_time": "16:00:00",
  "strikes": [
    {
      "strike_price": 16000.0,
      "call_open_interest": 1250,
      "put_open_interest": 980,
      "call_volume": 450,
      "put_volume": 320,
      "call_bid": 125.50,
      "call_ask": 126.25,
      "put_bid": 45.75,
      "put_ask": 46.25,
      "gamma": 0.0023,
      "delta": 0.45,
      "theta": -0.012,
      "vega": 0.089
    }
  ],
  "total_open_interest": 2230,
  "total_volume": 770,
  "processed_at": "2025-11-09T16:05:30Z",
  "data_quality_score": 0.98,
  "metadata": {
    "source": "gexbot",
    "version": "v2.1"
  }
}
```

### PriorityRequest

Request format for priority-based ingestion.

**JSON Schema**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["source_url", "data_type", "market_symbol"],
  "properties": {
    "source_url": {
      "type": "string",
      "format": "uri",
      "pattern": "^https?://",
      "description": "URL of the GEX data source"
    },
    "data_type": {
      "type": "string",
      "enum": ["historical", "real_time", "snapshot"],
      "description": "Type of GEX data"
    },
    "market_symbol": {
      "type": "string",
      "minLength": 1,
      "maxLength": 10,
      "description": "Underlying market symbol"
    },
    "metadata": {
      "type": "object",
      "description": "Additional request metadata",
      "properties": {
        "expected_records": {
          "type": "integer",
          "minimum": 1,
          "description": "Expected number of records"
        },
        "data_quality": {
          "type": "string",
          "enum": ["low", "medium", "high"],
          "description": "Expected data quality"
        },
        "priority_hint": {
          "type": "string",
          "enum": ["low", "normal", "high", "critical"],
          "description": "Priority hint for manual override"
        }
      },
      "additionalProperties": true
    }
  }
}
```

**Example**:
```json
{
  "source_url": "https://api.gexbot.com/ndx/gex-history?date=2025-11-09",
  "data_type": "historical",
  "market_symbol": "NDX",
  "metadata": {
    "expected_records": 15000,
    "data_quality": "high",
    "priority_hint": "high"
  }
}
```

### ProcessingStatus

Status response format for ingestion jobs.

**JSON Schema**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["request_id", "status", "priority_level", "submitted_at"],
  "properties": {
    "request_id": {
      "type": "string",
      "format": "uuid",
      "description": "Unique request identifier"
    },
    "status": {
      "type": "string",
      "enum": ["QUEUED", "PROCESSING", "COMPLETED", "FAILED"],
      "description": "Current processing status"
    },
    "priority_level": {
      "type": "string",
      "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
      "description": "Assigned priority level"
    },
    "submitted_at": {
      "type": "string",
      "format": "date-time",
      "description": "When request was submitted"
    },
    "started_at": {
      "type": "string",
      "format": "date-time",
      "description": "When processing started"
    },
    "completed_at": {
      "type": "string",
      "format": "date-time",
      "description": "When processing completed"
    },
    "records_processed": {
      "type": "integer",
      "minimum": 0,
      "description": "Number of records processed"
    },
    "estimated_completion": {
      "type": "string",
      "format": "date-time",
      "description": "Estimated completion time"
    },
    "error_message": {
      "type": "string",
      "description": "Error message if failed"
    },
    "progress_percentage": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 100.0,
      "description": "Processing progress percentage"
    }
  }
}
```

**Example**:
```json
{
  "request_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "PROCESSING",
  "priority_level": "HIGH",
  "submitted_at": "2025-11-09T10:30:00Z",
  "started_at": "2025-11-09T10:30:15Z",
  "records_processed": 7500,
  "estimated_completion": "2025-11-09T10:35:00Z",
  "progress_percentage": 50.0
}
```

## Data Validation Rules

### GEX Data Quality Checks

1. **Completeness**: All required fields present
2. **Consistency**: Strike prices are properly ordered
3. **Reasonableness**: Open interest and volume within expected ranges
4. **Timeliness**: Snapshot times are reasonable for market hours

### Priority Validation

1. **Source URL**: Must be valid HTTPS URL from trusted domains
2. **Market Symbol**: Must be known optionable symbol
3. **Data Type**: Must match expected format for the source
4. **Metadata**: Optional fields must conform to schema if present

## Compression and Serialization

### Recommended Formats

- **Storage**: Parquet with Snappy compression for analytical queries
- **Transport**: JSON with gzip compression for API responses
- **Streaming**: Protocol Buffers for real-time data streams

### Size Estimates

- **Single Strike**: ~200 bytes JSON
- **Full Snapshot**: ~2-5 MB for major indices (10,000+ strikes)
- **Compressed Parquet**: ~30-50% of JSON size

## Versioning

- **v1.0**: Initial release with core GEX fields
- **Future**: Add implied volatility, rho, and additional Greeks as needed

## Migration Notes

Existing GEX data can be converted to this format by:
1. Adding snapshot metadata fields
2. Reordering strikes by price
3. Calculating total aggregates
4. Adding data quality scores based on existing validation