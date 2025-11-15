# Feature 005: Unified Real-Time Ingestion + RedisTimeSeries Buffer

**Owner:** Codex assistant  
**Created:** 2025-11-11  
**Status:** In Progress

## Goal
Run all real-time services via `data-pipeline.py`, buffering one full trading day of TastyTrade (trades + summarized depth) and GEXBot data inside RedisTimeSeries, then flushing to DuckDB/Parquet every 10 minutes. Add guardrails and daily compressed backups once ingestion is stable.

## Scope
- TastyTrade DXLink streamer managed by `data-pipeline.py`
- GEXBot poller (NQ_NDX, ES_SPX, SPY, QQQ, SPX, NDX) polling every 60s
- RedisTimeSeries cache with ~24h retention
- 10-minute batch flush to DuckDB + Parquet
- Future guardrails & backups (plan defined; implementation after ingestion)

## Task Breakdown

### 1. Service Extraction & Config
- [x] Move DXLink ingestion into `src/services/tastytrade_streamer.py` (start/stop API)
- [x] Create `src/services/gexbot_poller.py` with cache for latest snapshot & max-change
- [x] Extend `src/config.Settings` with service toggles + Redis/flush params
- [x] Nightly GEX symbol discovery → store supported-ticker set in Redis (24h TTL)
- [x] Auto-enroll ad-hoc symbols for the current day (persist list in Redis, expires nightly)

### 2. RedisTimeSeries Integration
- [ ] Define key schema + retention in docs and config
- [ ] Define key schema + retention in docs and config
- [x] Write trades/depth updates from streamer into RedisTimeSeries
- [x] Write GEX snapshots/max-change into RedisTimeSeries

### 3. Flush Pipeline (10 min)
- [x] Implement flush worker to read RedisTimeSeries deltas, persist to DuckDB/Parquet
- [ ] Add logging/metrics for flush success/failure

### 4. FastAPI Lifecycle Wiring
- [ ] Update `data-pipeline.py` lifespan to start/stop streamer, poller, flush worker
- [ ] Ensure graceful shutdown (cancel tasks, final flush)

### 5. Guardrails & Backups (post-ingestion)
- [ ] Data quality checker job (volume sanity, missing intervals, GEX coverage)
- [ ] Backup script (tar+xz configs + DB + daily Parquet) with retention policy

### 6. Testing & Docs
- [ ] Tests for services + flush workflow
- [ ] Update quickstarts/datasources with RedisTimeSeries + backup instructions
- [ ] Provide operational runbook

## Notes
- GEXBot poller references legacy implementation in `../torch-market/data-pipeline.py` but will now store latest snapshot/max-change in memory/Redis instead of JSON files.
- Redis retention/flush cadence can be tuned later; start with 24h retention and 10-minute flush.
- Guardrails/backups defer until ingestion proves stable.
