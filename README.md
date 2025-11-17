# Financial Tick Data Pipeline

A high-performance data pipeline platform designed for financial tick data, providing secure real-time and historical data access for AI models used in trading and backtesting applications.

## Features

- **Real-time Data Ingestion**: Collect tick data from multiple sources (Sierra Chart, gexbot API, TastyTrade DXClient)
- **Data Quality Assurance**: Filter and validate data for gaps and accuracy
- **Memory + Disk Storage**: Keep recent data in Redis cache, compress historical data to DuckDB/Parquet
- **Secure AI Access**: API key authentication for AI models to query data
- **Real-time Trading Support**: Sub-10ms query latency for trading decisions
- **Backtesting Support**: Historical data queries for strategy development
- **Monitoring Dashboard**: Web UI for service status and data sampling

## Architecture

- **Backend**: Python 3.11 with FastAPI, Polars, DuckDB, Redis
- **Frontend**: Vanilla HTML/CSS/JavaScript monitoring dashboard
- **Storage**: Redis for hot data, DuckDB for historical data
- **Data Flow**: Sources → Validation → Memory Cache → Compression → Disk Storage

## Quick Start

See [specs/001-tick-data-pipeline/quickstart.md](specs/001-tick-data-pipeline/quickstart.md) for detailed setup instructions.

### Prerequisites

- Python 3.11+
- Redis server
- Data source credentials

## Script flags and common options

Some of the helper scripts and the orchestrator can emit a canonical `ts_ms`
column (epoch milliseconds) for timestamps. Use the `--convert-timestamp-to-ms`
flag to enable this and `--timestamp-tz` to specify the timezone the source
timestamps should be interpreted with.

- `--convert-timestamp-to-ms`: Add a `ts_ms` BIGINT column when writing depth/tick Parquet
- `--timestamp-tz <TZ>`: The timezone name (e.g., `UTC`, `America/New_York`) to
   treat naive timestamps as. When converting Parquet with `convert_parquet_to_ts_ms.py`
   the `--timestamp-tz` value is used in the DuckDB `AT TIME ZONE` expression so that
   `ts_ms` aligns with the intended timezone. Likewise, `verify_timestamps.py` can
   take the same `--timestamp-tz` flag to ensure comparisons use the same timezone.

Example:
```bash
python3 scripts/orchestrator.py --start 2025-11-11 --end 2025-11-12 --workers 2 \
   --scid-dir /mnt/c/SierraChart/Data --depth-dir /mnt/c/SierraChart/Data/MarketDepthData \
   --depth-prefix MNQZ25_FUT_CME --convert-timestamp-to-ms --timestamp-tz America/New_York
```
### Installation

```bash
pip install -e .
```

### Running

```bash
# Start Redis (local instance)
redis-server redis/redis.conf &

# Start the unified pipeline (services + control API)
python data-pipeline.py --host 0.0.0.0 --port 8877

# Monitor services / restart feeds
# Visit http://localhost:8877/status.html
```

### Schwab Streaming Service

The Schwab streamer ingests tick + level 2 market data directly into the trading bus (Redis pub/sub).

1. Set the following environment variables in `.env`:
   - `SCHWAB_ENABLED=true`
   - `SCHWAB_CLIENT_ID`, `SCHWAB_CLIENT_SECRET`, `SCHWAB_REFRESH_TOKEN`
   - `SCHWAB_ACCOUNT_ID` (optional, for downstream routing)
   - `SCHWAB_SYMBOLS=MNQ,MES,SPY,QQQ,VIX` (customize symbols)
   - `SCHWAB_TICK_CHANNEL` / `SCHWAB_LEVEL2_CHANNEL` if you need custom Redis targets
2. Install dependencies: `pip install -e .`
3. Generate Schwab OAuth tokens (one-time, or whenever consent expires):
   ```bash
   python scripts/schwab_oauth_helper.py
   ```
   This prints the Schwab consent URL, guides you through login/MFA, and saves the new `SCHWAB_REFRESH_TOKEN` to `.env`.
4. (Optional) Keep tokens fresh automatically:
   ```bash
   python scripts/schwab_token_rotator.py --access-interval-minutes 25 --refresh-interval-days 6
   ```
   The rotator stores the latest access/refresh pair in `data/schwab_tokens.json` and rewrites `SCHWAB_REFRESH_TOKEN` in `.env` every six days. Leave it running alongside your data services so the streamer always has a valid token.
5. Run the streamer:

```bash
python scripts/run_schwab_streamer.py
```

Use `--dry-run` to validate configuration without opening the websocket.

## API Documentation

### Real-time Queries
```http
GET /api/v1/ticks/realtime?symbols=AAPL&limit=100
Authorization: Bearer <api-key>
```

### Historical Queries
```http
GET /api/v1/ticks/historical?symbols=AAPL&start_time=2025-01-01T00:00:00Z&end_time=2025-01-02T00:00:00Z&interval=1h
Authorization: Bearer <api-key>
```

### System Status
```http
GET /api/v1/status
```

## Development

### Testing
```bash
pytest backend/tests/
```

### Linting
```bash
ruff check .
ruff format .
```

## Project Structure

```
backend/
├── src/
│   ├── api/          # FastAPI routes
│   ├── models/       # Pydantic models
│   ├── services/     # Business logic
│   └── config.py     # Configuration
└── tests/            # Test suites

frontend/
└── src/              # Monitoring UI

data/
├── source/           # Raw data from sources
├── enriched/         # Processed data
├── tick_data.db      # DuckDB database
└── my_trades.db      # Trade data

redis/                # Redis configuration

specs/                # Feature specifications and plans
```

## Contributing

1. Follow the specification-driven development process
2. Write tests before implementation (TDD)
3. Ensure all code passes linting and tests
4. Update documentation as needed

## License

[Add license information]
