# Services Agent Instructions

## Domain Scope
This folder contains the core service layer for the data pipeline. Sub-agents working here should focus on:

### Market Data Ingestion (Priority: High)
- `schwab_streamer.py` - Schwab WebSocket streaming
- `tastytrade_streamer.py` - TastyTrade DXLink streaming
- `tastytrade_client.py` - TastyTrade REST client

**Constraints:**
- Never modify OAuth token handling without explicit approval
- Maintain backward compatibility with existing Redis channel schemas
- All streaming services must implement `start()`, `stop()`, `is_running` interface

### GEX Data Services
- `gexbot_poller.py` - GEXBot API polling
- Related: `src/importers/gex_importer.py`

**Constraints:**
- Polling intervals are market-hours aware (RTH vs off-hours)
- Sierra Chart output format must remain stable

### Redis TimeSeries
- `redis_timeseries.py` - TimeSeries operations
- `redis_flush_worker.py` - Flush to persistent storage
- `lookup_service.py` - Query interface

**Constraints:**
- Key naming convention: `ts:{type}:{field}:{symbol}:{source}`
- Never delete TimeSeries keys without flush confirmation

### UW Message Processing
- `uw_message_service.py` - Unusual Whales parsing
- `market_agg_alert_service.py` - Alert routing

**Constraints:**
- Phoenix message format: `[joinRef, ref, topic, eventType, payload]`
- Maintain Redis key TTLs for cache management

## Testing Requirements
- Unit tests in `src/services/tests/`
- Integration tests require Redis connection
- Mock external APIs (Schwab, TastyTrade, GEXBot) in tests

## Common Patterns
```python
# Service lifecycle pattern
class MyService:
    def __init__(self, settings: MySettings, redis_client: RedisClient):
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
    
    def start(self) -> None: ...
    async def stop(self) -> None: ...
    
    @property
    def is_running(self) -> bool: ...
    
    def status(self) -> Dict[str, Any]: ...
```

## Do NOT
- Hardcode API keys or secrets
- Block the event loop with synchronous I/O
- Modify `src/config.py` without updating docs
