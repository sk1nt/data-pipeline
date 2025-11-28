# Quickstart Guide

## Prerequisites
- Python 3.11+
- Tastytrade account (sandbox and production)
- Discord bot token
- Redis instance
- DuckDB database

## Installation
```bash
pip install tastytrade discord.py pydantic python-dotenv redis duckdb
```

## Configuration
Create `.env` file:
```bash
DISCORD_TOKEN=your_bot_token
TASTYTRADE_CLIENT_SECRET=your_client_secret
TASTYTRADE_REFRESH_TOKEN=your_refresh_token
TASTYTRADE_USE_SANDBOX=true
REDIS_URL=redis://localhost:6379
DATABASE_URL=duckdb:///data/orders.db
```

## Running the Services
1. Start Redis: `redis-server`
2. Start Discord bot: `python src/discord_bot/main.py`
3. Start API server: `python src/api/main.py`
4. Start order processor: `python src/services/order_processor.py`

## Testing
- Use sandbox environment for initial testing
- Run unit tests: `pytest tests/unit/`
- Run integration tests: `pytest tests/integration/`

## Production Deployment
1. Set `TASTYTRADE_USE_SANDBOX=false`
2. Update production credentials
3. Run full test suite
4. Deploy with monitoring enabled