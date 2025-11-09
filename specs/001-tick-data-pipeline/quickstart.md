# Quickstart: Financial Tick Data Pipeline

**Feature**: Financial Tick Data Pipeline  
**Date**: 2025-11-07  
**Purpose**: Get the tick data pipeline up and running quickly for development and testing.

## Prerequisites

- Python 3.11+
- Redis server
- Access to data sources (Sierra Chart, gexbot API credentials, TastyTrade DXClient)

## Installation

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd data-pipeline
   git checkout 001-tick-data-pipeline
   python -m venv .venv
   source .venv/bin/activate
   pip install polars duckdb redis fastapi uvicorn
   ```

2. **Configure environment**:
   ```bash
   cp backend/config.example.yaml backend/config.yaml
   # Edit config.yaml with your API keys and settings
   ```

3. **Start Redis**:
   ```bash
   redis-server redis/redis.conf
   ```

## Running the Pipeline

1. **Start the backend API**:
   ```bash
   cd backend
   python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Open the monitoring UI**:
   - Open `frontend/index.html` in your browser
   - Or serve with a simple HTTP server:
     ```bash
     cd frontend
     python -m http.server 3000
     ```
   - Visit http://localhost:3000

3. **Start data ingestion** (in separate terminals):
   ```bash
   # Sierra Chart ingestion
   python backend/src/services/ingestion_sierra.py

   # Gexbot API ingestion
   python backend/src/services/ingestion_gexbot.py

   # TastyTrade ingestion
   python backend/src/services/ingestion_tastyttrade.py
   ```

## Testing the API

1. **Register an AI model** (admin operation):
   ```bash
   curl -X POST http://localhost:8000/api/v1/models \
     -H "Content-Type: application/json" \
     -d '{"model_id": "test-model", "name": "Test AI Model", "permissions": {"symbols": ["AAPL"], "query_types": ["realtime", "historical"]}}'
   ```
   Returns API key in response.

2. **Query real-time data**:
   ```bash
   curl "http://localhost:8000/api/v1/ticks/realtime?symbols=AAPL" \
     -H "X-API-Key: YOUR_API_KEY"
   ```

3. **Query historical data**:
   ```bash
   curl "http://localhost:8000/api/v1/ticks/historical?symbols=AAPL&start_time=2025-11-06T00:00:00Z&end_time=2025-11-07T00:00:00Z&interval=1m" \
     -H "X-API-Key: YOUR_API_KEY"
   ```

## Monitoring

- **UI Dashboard**: View service statuses and recent data samples at http://localhost:3000
- **API Status**: GET /api/v1/status for programmatic monitoring
- **Logs**: Check backend logs for ingestion and processing status

## Data Locations

- **Raw tick data**: `./data/source/{source}/`
- **Enriched data**: `./data/enriched/`
- **Database files**: `./data/tick_data.db`, `./data/my_trades.db`
- **Redis cache**: `./redis/`

## Troubleshooting

- **No data appearing**: Check data source connections and API credentials
- **Slow queries**: Verify Redis is running and configured correctly
- **UI not loading**: Ensure backend API is running on port 8000
- **Ingestion errors**: Check logs for specific source connection issues

## Next Steps

- Run the test suite: `python -m pytest backend/tests/`
- Configure production settings in `backend/config.yaml`
- Set up monitoring and alerting for production deployment
- Review security settings for AI model access