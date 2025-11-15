# Data Models: Schwab Real-Time GEX Support

**Feature**: Schwab Real-Time GEX Support
**Date**: November 10, 2025
**Purpose**: Define data structures and validation schemas for Schwab API integration and GEX calculations

## Overview

This document defines the data models for integrating with Schwab's trading API, handling real-time market data, and performing in-memory GEX calculations. All models use Pydantic for validation and type safety.

## Core Entities

### SchwabConnection

Represents an authenticated connection to Schwab's API with session management.

```python
class SchwabConnection(BaseModel):
    connection_id: UUID
    app_key: str
    app_secret: str
    access_token: Optional[str]
    refresh_token: Optional[str]
    token_expires_at: Optional[datetime]
    refresh_expires_at: Optional[datetime]
    status: ConnectionStatus  # CONNECTED, DISCONNECTED, AUTHENTICATING, ERROR
    last_connected_at: Optional[datetime]
    reconnect_attempts: int = 0
    max_reconnect_attempts: int = 5
    created_at: datetime
    updated_at: datetime
```

**Validation Rules**:
- `app_key` and `app_secret` must be non-empty strings
- `token_expires_at` must be in the future when present
- `reconnect_attempts` cannot exceed `max_reconnect_attempts`

**Relationships**:
- One-to-many with MarketDataStream
- One-to-many with SchwabRequest

### MarketData

Real-time market data received from Schwab streams.

```python
class MarketData(BaseModel):
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    bid: Optional[float]
    ask: Optional[float]
    bid_size: Optional[int]
    ask_size: Optional[int]
    last_trade_price: Optional[float]
    last_trade_size: Optional[int]
    day_high: Optional[float]
    day_low: Optional[float]
    day_open: Optional[float]
    prev_close: Optional[float]
    data_source: str = "schwab"
    quality: DataQuality  # REALTIME, DELAYED, STALE
```

**Validation Rules**:
- `symbol` must be valid ticker format (1-5 uppercase letters, optional numbers)
- `price` must be positive
- `volume` must be non-negative
- `timestamp` must be current or recent (within 5 minutes)

### OptionData

Options chain data for GEX calculations.

```python
class OptionData(BaseModel):
    option_symbol: str
    underlying_symbol: str
    strike_price: float
    expiration_date: date
    option_type: OptionType  # CALL, PUT
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: Optional[float]
    delta: Optional[float]
    gamma: Optional[float]
    theta: Optional[float]
    vega: Optional[float]
    rho: Optional[float]
    timestamp: datetime
    data_source: str = "schwab"
```

**Validation Rules**:
- `strike_price` must be positive
- `expiration_date` must be in the future
- Greeks values must be reasonable ranges (-1.0 to 1.0 for delta, etc.)
- `option_symbol` must follow OCC format

### GEXCalculation

Result of gamma exposure calculations.

```python
class GEXCalculation(BaseModel):
    calculation_id: UUID
    symbol: str
    spot_price: float
    total_gamma: float
    gamma_flip_price: Optional[float]
    max_gamma_price: Optional[float]
    call_gamma: float
    put_gamma: float
    net_gamma: float
    calculation_timestamp: datetime
    data_timestamp: datetime
    options_count: int
    calculation_method: str = "black_scholes"
    confidence_score: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = {}
```

**Validation Rules**:
- `total_gamma` should balance with `call_gamma + put_gamma`
- `confidence_score` must be between 0.0 and 1.0
- `data_timestamp` should be recent (within 5 minutes of calculation)

### CacheEntry

Metadata for cached data items.

```python
class CacheEntry(BaseModel):
    key: str
    data_type: CacheDataType  # MARKET_DATA, OPTIONS_CHAIN, GEX_RESULT
    created_at: datetime
    expires_at: Optional[datetime]
    last_accessed_at: datetime
    access_count: int = 0
    data_size_bytes: int
    source: str = "schwab"
    quality: DataQuality
    tags: List[str] = []
```

**Validation Rules**:
- `expires_at` must be after `created_at` when present
- `data_size_bytes` must be positive
- `access_count` must be non-negative

## Data Flow Models

### MarketDataStream

Configuration for real-time data streams.

```python
class MarketDataStream(BaseModel):
    stream_id: UUID
    connection_id: UUID
    symbols: List[str]
    stream_type: StreamType  # QUOTES, OPTIONS, NEWS
    status: StreamStatus  # ACTIVE, PAUSED, ERROR
    last_message_at: Optional[datetime]
    message_count: int = 0
    error_count: int = 0
    created_at: datetime
```

### SchwabRequest

API request tracking for rate limiting and monitoring.

```python
class SchwabRequest(BaseModel):
    request_id: UUID
    connection_id: UUID
    endpoint: str
    method: str
    request_time: datetime
    response_time: Optional[datetime]
    status_code: Optional[int]
    response_size_bytes: Optional[int]
    error_message: Optional[str]
    rate_limit_remaining: Optional[int]
    rate_limit_reset_at: Optional[datetime]
```

## Validation Schemas

### Business Rules

1. **Data Freshness**: Market data older than 5 minutes is considered stale
2. **Option Chain Completeness**: Must have at least 10 strikes for reliable GEX calculation
3. **Gamma Calculation Bounds**: Total gamma should be within reasonable bounds for underlying asset
4. **Connection Health**: More than 3 consecutive connection failures triggers circuit breaker

### Cross-Entity Validation

1. **Temporal Consistency**: GEX calculation timestamp should be after all input data timestamps
2. **Symbol Matching**: Option data underlying symbol must match GEX calculation symbol
3. **Data Source Integrity**: All related data should come from the same source (Schwab)

## Database Schema

### DuckDB Tables

```sql
-- Historical market data
CREATE TABLE market_data (
    symbol VARCHAR,
    price DOUBLE,
    volume INTEGER,
    timestamp TIMESTAMP,
    data_source VARCHAR DEFAULT 'schwab'
);

-- Options data
CREATE TABLE options_data (
    option_symbol VARCHAR PRIMARY KEY,
    underlying_symbol VARCHAR,
    strike_price DOUBLE,
    expiration_date DATE,
    option_type VARCHAR,
    bid DOUBLE,
    ask DOUBLE,
    last_price DOUBLE,
    volume INTEGER,
    open_interest INTEGER,
    gamma DOUBLE,
    timestamp TIMESTAMP
);

-- GEX calculations
CREATE TABLE gex_calculations (
    calculation_id VARCHAR PRIMARY KEY,
    symbol VARCHAR,
    spot_price DOUBLE,
    total_gamma DOUBLE,
    calculation_timestamp TIMESTAMP,
    data_timestamp TIMESTAMP,
    options_count INTEGER
);
```

### Redis Keys

```
# Real-time market data
market:{symbol} → MarketData JSON

# Options chains
options:{symbol}:{expiration} → List of OptionData JSON

# GEX results
gex:{symbol} → GEXCalculation JSON

# Cache metadata
cache:meta:{key} → CacheEntry JSON

# Stream status
stream:{stream_id} → MarketDataStream JSON
```

## Performance Considerations

### Memory Optimization

- Use `__slots__` in Pydantic models for reduced memory footprint
- Implement data compression for large option chains
- Use memory-mapped structures for read-heavy operations

### Caching Strategy

- Market data: 5-minute TTL with automatic refresh
- Options data: 1-minute TTL during market hours
- GEX results: 30-second TTL with invalidation on price changes

### Indexing Strategy

- Symbol-based indexing for fast lookups
- Time-based indexing for historical queries
- Expiration-based indexing for options data

## Migration Strategy

### Data Migration

1. **Initial Load**: Import historical options data from existing sources
2. **Incremental Updates**: Use Schwab streams for real-time updates
3. **Data Validation**: Cross-check with existing data sources for accuracy

### API Migration

1. **Gradual Rollout**: Start with read-only operations
2. **Parallel Operation**: Run alongside existing data sources
3. **Cutover Plan**: Switch to Schwab as primary source after validation

## Monitoring and Observability

### Key Metrics

- **Data Freshness**: Age of cached market data
- **Cache Hit Rate**: Percentage of requests served from cache
- **GEX Calculation Latency**: Time to compute gamma exposure
- **API Error Rate**: Percentage of failed Schwab API calls
- **Connection Uptime**: Percentage of time connected to Schwab

### Alerting Thresholds

- Data freshness > 5 minutes
- Cache hit rate < 95%
- GEX calculation > 200ms
- API error rate > 5%
- Connection uptime < 99%