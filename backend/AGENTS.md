# Backend Agent Instructions

## Domain Scope
This folder contains the secondary backend services layer. Sub-agents working here should focus on:

### Services
- `services/auth_service.py` - Authentication helpers
- `services/duckdb_service.py` - DuckDB query wrapper
- `services/redis_service.py` - Redis utilities
- `services/enriched_service.py` - Enriched data queries
- `services/tick_service.py` - Tick data access

### API Layer
- `api/main.py` - FastAPI app
- `api/status.py` - Status endpoints
- `api/ticks.py` - Tick endpoints

## Relationship to `src/`
This backend layer was created for specific features and may have overlap with `src/`. When working here:

1. **Check `src/services/` first** - Prefer extending existing services
2. **Avoid duplication** - If similar code exists in `src/`, refactor to shared lib
3. **Migration target** - Eventually consolidate into `src/`

## Consolidation Priority
Services to merge with `src/`:
- `duckdb_service.py` → `src/db/duckdb_utils.py`
- `redis_service.py` → `src/lib/redis_client.py`
- `auth_service.py` → `src/services/auth_service.py`

## Testing
- Tests in `backend/tests/`
- Mirror the source structure: `tests/services/test_duckdb_service.py`

## Do NOT
- Create new services here without checking `src/` first
- Duplicate Redis connection logic
- Add new API endpoints (use `src/api/` instead)
