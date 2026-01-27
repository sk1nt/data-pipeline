# API Agent Instructions

## Domain Scope
This folder contains the FastAPI REST/WebSocket API layer. Sub-agents working here should focus on:

### Components
- `app.py` - FastAPI application factory
- `gex_api.py` - GEX data endpoints
- `tick_api.py` - Tick data endpoints
- `depth_api.py` - Depth data endpoints
- `health.py` - Health check endpoints
- `routes/admin.py` - Admin operations

### Related (Main Entrypoint)
- `data-pipeline.py` (root) - Contains inline endpoints that should be migrated here

## Endpoint Standards

### Response Format
```python
# Success response
{"status": "ok", "data": {...}, "count": N}

# Error response
{"status": "error", "detail": "message", "code": "ERROR_CODE"}
```

### Route Naming
- `GET /api/gex/{symbol}` - Fetch GEX data
- `GET /api/ticks/{symbol}` - Fetch tick data
- `POST /admin/flush` - Trigger operations
- `GET /health` - Basic health
- `GET /health/detailed` - Dependency checks

### Pydantic Models
All request/response schemas in `src/models/api_models.py`:
```python
class GEXResponse(BaseModel):
    symbol: str
    sum_gex_vol: float
    timestamp: datetime
    
class ErrorResponse(BaseModel):
    status: Literal["error"]
    detail: str
    code: str
```

## WebSocket Patterns
```python
@app.websocket("/ws/{topic}")
async def websocket_handler(websocket: WebSocket, topic: str):
    await websocket.accept()
    # Subscribe to Redis pub/sub
    # Forward messages to client
    # Handle disconnection gracefully
```

## Migration Tasks
Move from `data-pipeline.py` to here:
- `/control/{service}/start|stop|restart` → `routes/control.py`
- `/lookup/*` → `routes/lookup.py`
- `/sc`, `/ws/sc` → `routes/sierra_chart.py`
- `/gex_history_url` → `routes/gex.py`
- `/uw` → `routes/uw.py`

## Testing
- Contract tests in `tests/contract/`
- Use `TestClient` from FastAPI
- Mock Redis/DuckDB dependencies

## Do NOT
- Add business logic directly in route handlers
- Skip input validation
- Return raw exceptions to clients
