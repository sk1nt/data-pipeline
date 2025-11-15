# Quickstart: Schwab Real-Time GEX Support

**Feature**: Schwab Real-Time GEX Support
**Date**: November 10, 2025
**Purpose**: Developer guide for setting up and using Schwab real-time GEX calculations

## Prerequisites

- Python 3.11+
- Redis server (for caching)
- Schwab Developer Account with API credentials
- DuckDB (automatically installed)

## Installation

### 1. Clone and Setup

```bash
cd /home/rwest/projects/data-pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```bash
# Schwab API Credentials
SCHWAB_APP_KEY=your_app_key_here
SCHWAB_APP_SECRET=your_app_secret_here
SCHWAB_REDIRECT_URI=http://localhost:8000/callback

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Database Configuration
DUCKDB_PATH=data/schwab_gex.db

# Service Configuration
SERVICE_PORT=8000
LOG_LEVEL=INFO
```

### 3. Schwab Developer Setup

1. Visit [Schwab Developer Portal](https://developer.schwab.com/)
2. Create a new application
3. Configure OAuth 2.0 redirect URI: `http://localhost:8000/callback`
4. Copy App Key and App Secret to `.env` file

## Quick Start

### 1. Start the Service

```bash
cd src
python -m uvicorn main:app --reload --port 8000
```

### 2. Health Check

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-10T10:00:00Z",
  "version": "1.0.0",
  "uptime_seconds": 45.2,
  "services": {
    "schwab_api": "disconnected",
    "redis_cache": "available",
    "duckdb_storage": "available"
  }
}
```

### 3. Authenticate with Schwab

Visit: `http://localhost:8000/auth/schwab`

This will redirect you through Schwab's OAuth flow. After authentication, you'll be redirected back to the application.

### 4. Get GEX Data

```bash
# Get current GEX for SPY
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     http://localhost:8000/api/v1/gex/SPY
```

Example response:
```json
{
  "symbol": "SPY",
  "calculation_timestamp": "2025-11-10T10:05:00Z",
  "data_timestamp": "2025-11-10T10:04:55Z",
  "spot_price": 450.25,
  "total_gamma": 1250000.50,
  "gamma_flip_price": 448.75,
  "max_gamma_price": 452.10,
  "call_gamma": 980000.25,
  "put_gamma": 270000.25,
  "net_gamma": 710000.00,
  "options_count": 245,
  "confidence_score": 0.92
}
```

## Basic Usage Examples

### Python Client

```python
import requests
import os

class SchwabGEXClient:
    def __init__(self, base_url="http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.token = os.getenv("JWT_TOKEN")

    def get_headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    def get_gex(self, symbol: str):
        """Get current GEX calculation for a symbol"""
        response = requests.get(
            f"{self.base_url}/gex/{symbol}",
            headers=self.get_headers()
        )
        return response.json()

    def get_market_data(self, symbol: str):
        """Get real-time market data"""
        response = requests.get(
            f"{self.base_url}/market/{symbol}",
            headers=self.get_headers()
        )
        return response.json()

    def get_options_chain(self, symbol: str, expiration_date=None):
        """Get options chain data"""
        params = {}
        if expiration_date:
            params["expiration_date"] = expiration_date

        response = requests.get(
            f"{self.base_url}/options/{symbol}",
            headers=self.get_headers(),
            params=params
        )
        return response.json()

# Usage
client = SchwabGEXClient()
gex_data = client.get_gex("SPY")
market_data = client.get_market_data("SPY")
options = client.get_options_chain("SPY")
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');

class SchwabGEXClient {
    constructor(baseURL = 'http://localhost:8000/api/v1', token) {
        this.client = axios.create({
            baseURL,
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
    }

    async getGEX(symbol) {
        const response = await this.client.get(`/gex/${symbol}`);
        return response.data;
    }

    async getMarketData(symbol) {
        const response = await this.client.get(`/market/${symbol}`);
        return response.data;
    }

    async getOptionsChain(symbol, expirationDate = null) {
        const params = expirationDate ? { expiration_date: expirationDate } : {};
        const response = await this.client.get(`/options/${symbol}`, { params });
        return response.data;
    }
}

// Usage
const client = new SchwabGEXClient('http://localhost:8000/api/v1', process.env.JWT_TOKEN);

async function example() {
    try {
        const gex = await client.getGEX('SPY');
        console.log('GEX for SPY:', gex);

        const market = await client.getMarketData('SPY');
        console.log('Market data:', market);
    } catch (error) {
        console.error('Error:', error.response.data);
    }
}

example();
```

## Common Tasks

### Monitor Connection Status

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/api/v1/connection/status
```

### Check Cache Performance

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     http://localhost:8000/api/v1/cache/stats
```

### Get Historical GEX Data

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     "http://localhost:8000/api/v1/gex/SPY/history?start_date=2025-11-10T09:00:00Z&interval=5m"
```

## Troubleshooting

### Connection Issues

**Problem**: Schwab API connection fails
```
Error: Authentication failed
```

**Solutions**:
1. Verify credentials in `.env` file
2. Check Schwab Developer Portal for application status
3. Ensure redirect URI matches exactly
4. Try re-authenticating: `http://localhost:8000/auth/schwab`

### Data Quality Issues

**Problem**: GEX calculations show low confidence
```
"confidence_score": 0.45
```

**Solutions**:
1. Check options data availability
2. Verify market data freshness
3. Ensure sufficient options in chain (minimum 10 strikes)
4. Check for data source errors in logs

### Performance Issues

**Problem**: Slow response times
```
Response time > 2 seconds
```

**Solutions**:
1. Check Redis connection: `redis-cli ping`
2. Monitor cache hit rate: `/api/v1/cache/stats`
3. Verify DuckDB performance: check disk I/O
4. Review Schwab API rate limits

### Common Error Codes

| Error Code | Description | Solution |
|------------|-------------|----------|
| 401 | Unauthorized | Check JWT token validity |
| 404 | Symbol not found | Verify symbol format (e.g., SPY, AAPL) |
| 429 | Rate limited | Implement exponential backoff |
| 503 | Service unavailable | Check Schwab API status and connection |

## Development Workflow

### Running Tests

```bash
cd src
python -m pytest tests/ -v
```

### Code Quality Checks

```bash
# Linting
ruff check .

# Type checking
mypy .

# Formatting
black .
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-gex-endpoint`
2. Add tests in `tests/`
3. Implement in `src/`
4. Update API contracts in `contracts/`
5. Update documentation
6. Run full test suite
7. Create pull request

## Next Steps

- Explore the [API Documentation](http://localhost:8000/docs) (Swagger UI)
- Review the [Data Models](data-model.md) for detailed schema information
- Check the [Implementation Plan](plan.md) for roadmap details
- Join the development discussions in project issues

## Support

- **Documentation**: See `docs/` directory
- **Issues**: Create GitHub issues for bugs/features
- **Discussions**: Use GitHub Discussions for questions
- **Logs**: Check application logs in `logs/app.log`