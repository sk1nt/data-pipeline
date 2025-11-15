# Quickstart: High-Speed GEX Data Ingest Priorities

**Feature**: High-Speed GEX Data Ingest Priorities with Firm Guidelines and Methodology
**Version**: 1.0
**Date**: November 9, 2025

## Overview

This guide provides a quick start for developers to set up and use the high-speed GEX data ingest priority system. The system provides automatic priority assignment and real-time processing guarantees for critical market data.

## Prerequisites

- **Python**: 3.11+
- **Dependencies**: Polars, DuckDB, FastAPI, Pydantic, Redis
- **System**: Linux server with 4GB+ RAM
- **Network**: Access to GEX data sources and Redis instance

## Installation

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd data-pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```bash
# Database
DUCKDB_PATH=data/priority_system.db

# Redis
REDIS_URL=redis://localhost:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8000

# Data directories
DATA_DIR=data/
PARQUET_DIR=data/parquet/
```

### 3. Initialize Database

```bash
# Run database migrations
python -m src.cli init-db

# Load default priority rules
python -m src.cli load-rules --file specs/001-data-ingest-priorities/config/default-rules.json
```

## Basic Usage

### Starting the API Server

```bash
# Start the FastAPI server
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Server will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

### Submitting a Priority Ingest Request

```bash
# Using curl
curl -X POST "http://localhost:8000/api/v1/ingest/priority" \
  -H "Content-Type: application/json" \
  -d '{
    "source_url": "https://api.gexbot.com/ndx/gex-history?date=2025-11-09",
    "data_type": "historical",
    "market_symbol": "NDX",
    "metadata": {
      "expected_records": 15000,
      "data_quality": "high"
    }
  }'
```

**Response**:
```json
{
  "request_id": "123e4567-e89b-12d3-a456-426614174000",
  "priority_level": "HIGH",
  "estimated_processing_time": "00:05:00",
  "queue_position": 2
}
```

### Checking Request Status

```bash
# Get status by request ID
curl "http://localhost:8000/api/v1/ingest/priority/123e4567-e89b-12d3-a456-426614174000"
```

**Response**:
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

### Monitoring Queue Status

```bash
# Get current queue status
curl "http://localhost:8000/api/v1/ingest/priority/queue"
```

**Response**:
```json
{
  "total_queued": 8,
  "by_priority": {
    "CRITICAL": 1,
    "HIGH": 3,
    "MEDIUM": 3,
    "LOW": 1
  },
  "oldest_request": "2025-11-09T10:25:00Z",
  "processing_capacity": 4
}
```

## Python Client Usage

### Basic Client Setup

```python
import requests
from typing import Dict, Any

class GEXPriorityClient:
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.session = requests.Session()

    def submit_ingest_request(self, source_url: str, data_type: str,
                            market_symbol: str, metadata: Dict[str, Any] = None) -> Dict:
        """Submit a GEX data ingest request with priority assignment."""
        payload = {
            "source_url": source_url,
            "data_type": data_type,
            "market_symbol": market_symbol,
            "metadata": metadata or {}
        }

        response = self.session.post(f"{self.base_url}/ingest/priority", json=payload)
        response.raise_for_status()
        return response.json()

    def get_request_status(self, request_id: str) -> Dict:
        """Get the status of an ingest request."""
        response = self.session.get(f"{self.base_url}/ingest/priority/{request_id}")
        response.raise_for_status()
        return response.json()

    def get_queue_status(self) -> Dict:
        """Get current priority queue status."""
        response = self.session.get(f"{self.base_url}/ingest/priority/queue")
        response.raise_for_status()
        return response.json()
```

### Usage Example

```python
# Initialize client
client = GEXPriorityClient()

# Submit a high-priority request
result = client.submit_ingest_request(
    source_url="https://api.gexbot.com/spx/gex-realtime",
    data_type="real_time",
    market_symbol="SPX",
    metadata={"priority_hint": "critical", "expected_records": 20000}
)

print(f"Request submitted: {result['request_id']}")
print(f"Priority: {result['priority_level']}")
print(f"Queue position: {result['queue_position']}")

# Monitor progress
import time
request_id = result['request_id']

while True:
    status = client.get_request_status(request_id)
    print(f"Status: {status['status']}, Progress: {status.get('progress_percentage', 0):.1f}%")

    if status['status'] in ['COMPLETED', 'FAILED']:
        break

    time.sleep(5)
```

## Configuration

### Priority Rules

Default priority rules are loaded automatically. You can customize them by editing the rules file:

```json
[
  {
    "name": "Critical Real-time Data",
    "description": "Highest priority for real-time SPX/NDX data",
    "condition": "data_type == 'real_time' and market_symbol in ['SPX', 'NDX']",
    "priority_score": 0.95,
    "is_active": true
  },
  {
    "name": "High Volume Strikes",
    "description": "High priority for strikes with significant open interest",
    "condition": "metadata.get('open_interest_threshold', 0) > 50000",
    "priority_score": 0.8,
    "is_active": true
  }
]
```

### Performance Tuning

```bash
# Environment variables for performance
export PROCESSING_WORKERS=4          # Number of parallel processing workers
export REDIS_POOL_SIZE=20            # Redis connection pool size
export BATCH_SIZE=1000              # Records per processing batch
export CACHE_TTL_SECONDS=300        # Redis cache TTL
```

## Troubleshooting

### Common Issues

1. **Request Rejected**: Check source URL format and market symbol validity
2. **Slow Processing**: Verify Redis connection and increase worker count
3. **Memory Issues**: Reduce batch size or increase system RAM
4. **Queue Backlog**: Check processing capacity and scale workers

### Debug Commands

```bash
# Check system status
python -m src.cli status

# View active requests
python -m src.cli list-requests --status PROCESSING

# Clear failed requests
python -m src.cli clear-failed

# View priority rules
python -m src.cli list-rules
```

### Logs

Logs are written to `logs/priority_system.log`. Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
```

## Next Steps

1. **Integration**: Integrate with existing GEX import workflows
2. **Monitoring**: Set up dashboards for priority queue metrics
3. **Scaling**: Configure multiple processing workers
4. **Testing**: Run the test suite with `pytest tests/`

## Support

- **API Documentation**: http://localhost:8000/docs
- **Logs**: `logs/priority_system.log`
- **Configuration**: `.env` file
- **Database**: `data/priority_system.db`