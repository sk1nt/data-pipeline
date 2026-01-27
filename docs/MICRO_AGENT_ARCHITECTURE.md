# Micro Agent Architecture - GitHub Issues

This document organizes the data-pipeline codebase into logical micro agent boundaries, each with corresponding GitHub issue templates for tracking and implementation.

---

## Overview: Micro Agent Domains

| Domain | Agent Name | Primary Responsibility |
|--------|------------|----------------------|
| 1 | **Market Data Ingestion** | Real-time streaming from Schwab, TastyTrade |
| 2 | **GEX Data Services** | GEXBot polling, snapshot caching, history imports |
| 3 | **Redis TimeSeries** | In-memory caching, time-series storage, lookups |
| 4 | **Persistence & Flush** | DuckDB/Parquet storage, data lifecycle |
| 5 | **API Gateway** | FastAPI endpoints, webhooks, WebSocket |
| 6 | **Discord Bot** | Alerts, commands, TastyTrade trading integration |
| 7 | **UW Message Processing** | Unusual Whales feed parsing and routing |
| 8 | **Batch Processing** | Historical imports, SCID conversion, orchestration |
| 9 | **Monitoring & Metrics** | Prometheus exporters, health checks |

---

## GitHub Issues

### Epic 1: Market Data Ingestion Agent

```markdown
## Issue #1: [Epic] Market Data Ingestion Micro Agent

### Description
Consolidate all real-time market data streaming into a self-contained micro agent with clear boundaries.

### Components
- `src/services/schwab_streamer.py` - Schwab WebSocket client (1101 lines)
- `src/services/tastytrade_streamer.py` - TastyTrade DXLink streamer (200 lines)
- `src/services/tastytrade_client.py` - TastyTrade REST client
- `src/models/market_data.py` - TickEvent, Level2Event, Level2Quote models
- `src/token_store.py` - OAuth token management

### Responsibilities
- [ ] Stream real-time tick data (trades)
- [ ] Stream Level 2 / depth-of-book data
- [ ] Handle OAuth token refresh lifecycle
- [ ] Normalize data formats across feeds
- [ ] Publish to Redis pub/sub channels

### Interface Contracts
**Input:** OAuth credentials, symbol lists
**Output:** `market_data:tastytrade:trades`, `market_data:schwab:*` Redis channels

### Acceptance Criteria
- [ ] Agent can start/stop independently
- [ ] Reconnection logic handles network failures
- [ ] Token refresh works without manual intervention
- [ ] Metrics exposed: connection status, message counts, latency

### Labels
`micro-agent`, `market-data`, `streaming`, `epic`
```

---

```markdown
## Issue #2: Schwab Streamer Refactor

### Parent Epic
#1 Market Data Ingestion Agent

### Description
Refactor `schwab_streamer.py` (1101 lines) into modular components.

### Tasks
- [ ] Extract `SchwabAuthClient` to `src/services/auth/schwab_auth.py`
- [ ] Extract `SchwabToken` dataclass to `src/models/auth.py`
- [ ] Create `SchwabStreamHandler` for message parsing
- [ ] Add retry/backoff logic as configurable middleware
- [ ] Unit test token refresh cycle

### Files
- `src/services/schwab_streamer.py`
- `src/token_store.py`
- `src/config.py` (Schwab settings)

### Labels
`micro-agent`, `refactor`, `schwab`
```

---

```markdown
## Issue #3: TastyTrade Streamer Enhancement

### Parent Epic
#1 Market Data Ingestion Agent

### Description
Enhance TastyTrade DXLink integration for parity with Schwab streamer.

### Tasks
- [ ] Add depth-of-book streaming (currently optional)
- [ ] Implement reconnection with exponential backoff
- [ ] Add connection health heartbeat
- [ ] Normalize output format to match Schwab events
- [ ] Add session persistence for faster reconnects

### Files
- `src/services/tastytrade_streamer.py`
- `discord-bot/bot/tastytrade_client.py`

### Labels
`micro-agent`, `tastytrade`, `enhancement`
```

---

### Epic 2: GEX Data Services Agent

```markdown
## Issue #4: [Epic] GEX Data Services Micro Agent

### Description
Unified agent for all GEX (Gamma Exposure) data operations.

### Components
- `src/services/gexbot_poller.py` - GEXBot API polling (662 lines)
- `src/importers/gex_importer.py` - Historical GEX file imports
- `src/import_gex_history.py` - Queue-based import processor
- `src/lib/gex_history_queue.py` - Import job queue
- `src/models/gex_snapshot.py` - GEX data models
- `src/models/gex_data.py` - Additional GEX models

### Responsibilities
- [ ] Poll GEXBot API at configurable intervals
- [ ] Cache snapshots in Redis with pub/sub notifications
- [ ] Process historical import queue
- [ ] Write to Sierra Chart format for external tools
- [ ] Maintain symbol lists (dynamic refresh)

### Interface Contracts
**Input:** GEXBot API key, symbol configurations
**Output:** `gex:snapshot:*` Redis keys, `gex:snapshot:stream` pub/sub

### Acceptance Criteria
- [ ] Polling adapts to RTH vs off-hours schedules
- [ ] Historical imports don't block real-time polling
- [ ] Sierra Chart files update atomically

### Labels
`micro-agent`, `gex`, `epic`
```

---

```markdown
## Issue #5: GEXBot Poller Service Isolation

### Parent Epic
#4 GEX Data Services Agent

### Description
Isolate GEXBot polling as a standalone service with clean interfaces.

### Tasks
- [ ] Extract settings to dedicated `GEXBotConfig` class
- [ ] Create abstract `GEXDataSource` interface
- [ ] Implement `GEXBotAPISource` implementing the interface
- [ ] Add circuit breaker for API rate limiting
- [ ] Separate NQ poller logic from main poller

### Current Issues
- Main poller and NQ poller share code but have different schedules
- Symbol exclusion logic is interleaved with polling

### Files
- `src/services/gexbot_poller.py`

### Labels
`micro-agent`, `gex`, `refactor`
```

---

```markdown
## Issue #6: GEX History Import Pipeline

### Parent Epic
#4 GEX Data Services Agent

### Description
Streamline historical GEX data imports with better job tracking.

### Tasks
- [ ] Create `GEXImportJob` model with states (queued, processing, completed, failed)
- [ ] Add import deduplication by URL hash
- [ ] Implement retry logic with exponential backoff
- [ ] Add import progress tracking via Redis
- [ ] Create CLI for manual import operations

### Files
- `src/import_gex_history.py`
- `src/lib/gex_history_queue.py`
- `src/importers/gex_importer.py`
- `src/import_job_store.py`

### Labels
`micro-agent`, `gex`, `pipeline`
```

---

### Epic 3: Redis TimeSeries Agent

```markdown
## Issue #7: [Epic] Redis TimeSeries Micro Agent

### Description
Centralize all Redis time-series operations and caching logic.

### Components
- `src/services/redis_timeseries.py` - TimeSeries client wrapper
- `src/lib/redis_client.py` - Base Redis connection
- `src/services/lookup_service.py` - Trade/depth lookups
- `backend/src/services/redis_service.py` - Alternative Redis helpers

### Responsibilities
- [ ] Manage Redis TimeSeries keys for trades/depth
- [ ] Provide lookup APIs for historical data
- [ ] Handle cross-feed depth comparisons
- [ ] Manage key TTLs and retention policies

### Interface Contracts
**Input:** Normalized market data events
**Output:** `ts:trade:*`, `ts:depth:*` TimeSeries keys

### Acceptance Criteria
- [ ] Consistent key naming across all services
- [ ] Automatic key creation with proper labels
- [ ] Memory-efficient aggregation rules

### Labels
`micro-agent`, `redis`, `timeseries`, `epic`
```

---

```markdown
## Issue #8: Lookup Service Consolidation

### Parent Epic
#7 Redis TimeSeries Agent

### Description
Unify lookup operations across the codebase.

### Tasks
- [ ] Merge `LookupService` with backend Redis services
- [ ] Add caching layer for frequent queries
- [ ] Implement pagination for large result sets
- [ ] Add query optimization hints

### Files
- `src/services/lookup_service.py`
- `backend/src/services/redis_service.py`

### Labels
`micro-agent`, `redis`, `refactor`
```

---

### Epic 4: Persistence & Flush Agent

```markdown
## Issue #9: [Epic] Persistence & Flush Micro Agent

### Description
Handle all durable storage operations: DuckDB, Parquet, data lifecycle.

### Components
- `src/services/redis_flush_worker.py` - TimeSeries → DuckDB/Parquet (919 lines)
- `src/db/duckdb_utils.py` - DuckDB operations
- `src/importers/tick_importer.py` - Tick data imports
- `src/importers/depth_importer.py` - Depth data imports
- `backend/src/services/duckdb_service.py` - Backend DuckDB wrapper
- `backend/src/services/enriched_service.py` - Enriched data queries

### Responsibilities
- [ ] Flush Redis TimeSeries to DuckDB on schedule
- [ ] Write Parquet files for archival
- [ ] Manage data retention and purging
- [ ] Handle GEX snapshot persistence
- [ ] Support tick/depth imports from external files

### Interface Contracts
**Input:** Redis TimeSeries data, import files
**Output:** DuckDB tables, Parquet files in `data/parquet/`

### Acceptance Criteria
- [ ] Flush operations are atomic/transactional
- [ ] Parquet files are partitioned by date
- [ ] Old data purging follows retention policy

### Labels
`micro-agent`, `persistence`, `duckdb`, `parquet`, `epic`
```

---

```markdown
## Issue #10: Redis Flush Worker Refactor

### Parent Epic
#9 Persistence & Flush Agent

### Description
Break down the 919-line flush worker into focused components.

### Tasks
- [ ] Extract `TimeSeriesFlusher` for TS → DuckDB
- [ ] Extract `ParquetWriter` for file operations
- [ ] Extract `GEXSnapshotPersister` for GEX-specific logic
- [ ] Create `FlushScheduler` for timing logic
- [ ] Add flush status/progress API

### Current Pain Points
- Mixed concerns: TS flush, GEX persistence, tick/depth handling
- Hard to test individual flush operations

### Files
- `src/services/redis_flush_worker.py`

### Labels
`micro-agent`, `refactor`, `persistence`
```

---

```markdown
## Issue #11: Data Retention & Purge Service

### Parent Epic
#9 Persistence & Flush Agent

### Description
Implement configurable data retention policies.

### Tasks
- [ ] Create `RetentionPolicy` model (per data type)
- [ ] Implement `PurgeService` for scheduled cleanup
- [ ] Add CLI for manual purge operations
- [ ] Support dry-run mode for purge preview
- [ ] Log purge operations for audit

### Related Scripts
- `scripts/purge_old_data.py`

### Labels
`micro-agent`, `persistence`, `cleanup`
```

---

### Epic 5: API Gateway Agent

```markdown
## Issue #12: [Epic] API Gateway Micro Agent

### Description
Consolidate all HTTP/WebSocket endpoints into a unified gateway.

### Components
- `data-pipeline.py` - Main FastAPI app (900+ lines)
- `src/api/app.py` - Secondary API surface
- `src/api/gex_api.py` - GEX endpoints
- `src/api/tick_api.py` - Tick data endpoints
- `src/api/depth_api.py` - Depth data endpoints
- `src/api/health.py` - Health checks
- `src/api/routes/admin.py` - Admin operations
- `backend/src/api/` - Backend API layer

### Responsibilities
- [ ] Route requests to appropriate services
- [ ] Handle authentication/authorization
- [ ] Expose WebSocket streams
- [ ] Serve status dashboard

### Current Issues
- Two separate FastAPI apps (`data-pipeline.py` and `src/api/app.py`)
- Endpoint logic mixed with service orchestration

### Acceptance Criteria
- [ ] Single entrypoint for all API traffic
- [ ] OpenAPI documentation complete
- [ ] Request/response logging

### Labels
`micro-agent`, `api`, `fastapi`, `epic`
```

---

```markdown
## Issue #13: Merge API Surfaces

### Parent Epic
#12 API Gateway Agent

### Description
Consolidate `data-pipeline.py` endpoints with `src/api/` modules.

### Tasks
- [ ] Move inline endpoints from `data-pipeline.py` to `src/api/routes/`
- [ ] Create `src/api/routes/control.py` for service control
- [ ] Create `src/api/routes/lookup.py` for lookup endpoints
- [ ] Create `src/api/routes/sierra_chart.py` for SC bridge
- [ ] Keep `data-pipeline.py` as pure orchestration entrypoint

### Files
- `data-pipeline.py`
- `src/api/app.py`
- `src/api/routes/`

### Labels
`micro-agent`, `api`, `refactor`
```

---

```markdown
## Issue #14: WebSocket Streaming Consolidation

### Parent Epic
#12 API Gateway Agent

### Description
Create unified WebSocket handler for all streaming endpoints.

### Tasks
- [ ] Create `WebSocketManager` for connection lifecycle
- [ ] Implement topic-based subscription model
- [ ] Add `/ws/trades`, `/ws/depth`, `/ws/gex` endpoints
- [ ] Support reconnection with state replay
- [ ] Add connection metrics

### Current Endpoints
- `/ws/sc` - Sierra Chart GEX stream

### Labels
`micro-agent`, `api`, `websocket`
```

---

### Epic 6: Discord Bot Agent

```markdown
## Issue #15: [Epic] Discord Bot Micro Agent

### Description
Isolate Discord bot as a fully independent service.

### Components
- `discord-bot/bot/trade_bot.py` - Main bot (3868 lines)
- `discord-bot/bot/tastytrade_client.py` - Trading client
- `discord-bot/bot/config.py` - Bot configuration
- `discord-bot/run_discord_bot.py` - Entrypoint
- `src/services/discord_bot_service.py` - Service wrapper

### Responsibilities
- [ ] Process Discord commands (!gex, !status, etc.)
- [ ] Execute trades via TastyTrade
- [ ] Forward alerts from Redis pub/sub
- [ ] Manage user permissions/allowlists

### Current Issues
- 3868 lines in single file
- Mixed concerns: commands, trading, alerts, feeds

### Acceptance Criteria
- [ ] Bot can run as standalone process
- [ ] Commands are modular/pluggable
- [ ] Trading operations are audited

### Labels
`micro-agent`, `discord`, `trading`, `epic`
```

---

```markdown
## Issue #16: Discord Bot Command Modularization

### Parent Epic
#15 Discord Bot Agent

### Description
Split monolithic trade_bot.py into command modules.

### Tasks
- [ ] Create `discord-bot/bot/commands/` directory
- [ ] Extract `gex_commands.py` for GEX lookups
- [ ] Extract `trade_commands.py` for order execution
- [ ] Extract `status_commands.py` for system status
- [ ] Extract `alert_commands.py` for alert management
- [ ] Use discord.py Cogs pattern

### Target Structure
```
discord-bot/bot/
├── commands/
│   ├── __init__.py
│   ├── gex.py
│   ├── trading.py
│   ├── status.py
│   └── alerts.py
├── services/
│   ├── redis_listener.py
│   └── tastytrade_executor.py
└── trade_bot.py (reduced to wiring)
```

### Labels
`micro-agent`, `discord`, `refactor`
```

---

```markdown
## Issue #17: TastyTrade Trading Integration

### Parent Epic
#15 Discord Bot Agent

### Description
Harden TastyTrade order execution with proper safeguards.

### Tasks
- [ ] Add order validation service
- [ ] Implement position limits
- [ ] Add kill switch for emergency stop
- [ ] Create audit log for all trades
- [ ] Add order confirmation workflow

### Files
- `discord-bot/bot/tastytrade_client.py`
- `src/services/automated_options_service.py`
- `src/services/futures_order_service.py`
- `src/services/order_validation.py`

### Labels
`micro-agent`, `discord`, `trading`, `safety`
```

---

### Epic 7: UW Message Processing Agent

```markdown
## Issue #18: [Epic] Unusual Whales Message Agent

### Description
Dedicated agent for processing Unusual Whales data feeds.

### Components
- `src/services/uw_message_service.py` - Message router (254 lines)
- `src/services/market_agg_alert_service.py` - Market alerts
- `src/models/uw_message.py` - Message models

### Responsibilities
- [ ] Parse Phoenix WebSocket messages
- [ ] Route to option trade or market agg handlers
- [ ] Store in Redis for downstream consumers
- [ ] Trigger alerts based on rules

### Interface Contracts
**Input:** Raw Phoenix arrays `[joinRef, ref, topic, eventType, payload]`
**Output:** `uw:market_agg:*`, `uw:option_trade:*` Redis keys

### Labels
`micro-agent`, `uw`, `alerts`, `epic`
```

---

```markdown
## Issue #19: UW Alert Rule Engine

### Parent Epic
#18 Unusual Whales Message Agent

### Description
Implement configurable alert rules for UW data.

### Tasks
- [ ] Create `AlertRule` model (conditions, actions)
- [ ] Implement rule evaluation engine
- [ ] Add rule persistence to Redis/DuckDB
- [ ] Support Discord, webhook, and pubsub actions
- [ ] Add rule management API

### Files
- `src/services/rule_engine.py`
- `src/models/alert.py`

### Labels
`micro-agent`, `uw`, `alerts`
```

---

### Epic 8: Batch Processing Agent

```markdown
## Issue #20: [Epic] Batch Processing Micro Agent

### Description
Consolidate all batch/offline processing scripts.

### Components
- `scripts/orchestrator.py` - Parallel date-range processing
- `scripts/worker_day.py` - Single-day processor
- `scripts/import_scid_ticks.py` - SCID tick imports
- `scripts/export_scid_ticks_to_parquet.py` - SCID → Parquet
- `scripts/enrich_tick_gex.py` - GEX enrichment
- `scripts/export_enriched_bars.py` - Bar aggregation
- `scripts/migrate_*.py` - Data migrations

### Responsibilities
- [ ] Historical data backfills
- [ ] Format conversions (SCID → Parquet)
- [ ] Data enrichment pipelines
- [ ] Schema migrations

### Acceptance Criteria
- [ ] Jobs can run in parallel where safe
- [ ] Progress tracking and resumability
- [ ] Consistent logging and error handling

### Labels
`micro-agent`, `batch`, `pipeline`, `epic`
```

---

```markdown
## Issue #21: Batch Job Framework

### Parent Epic
#20 Batch Processing Agent

### Description
Create a unified framework for batch job execution.

### Tasks
- [ ] Create `BatchJob` base class with lifecycle hooks
- [ ] Implement `JobRunner` with parallelization
- [ ] Add progress persistence for resumability
- [ ] Create `JobReport` for execution summaries
- [ ] Add CLI for job management

### Benefits
- Consistent error handling across scripts
- Resumable jobs after failures
- Unified logging and metrics

### Labels
`micro-agent`, `batch`, `framework`
```

---

```markdown
## Issue #22: SCID Processing Pipeline

### Parent Epic
#20 Batch Processing Agent

### Description
Standardize Sierra Chart SCID file processing.

### Tasks
- [ ] Create `SCIDReader` utility class
- [ ] Implement streaming parser for large files
- [ ] Add validation for SCID integrity
- [ ] Support multiple contract roll schedules
- [ ] Document CME contract windows

### Files
- `scripts/import_scid_ticks.py`
- `scripts/export_scid_ticks_to_parquet.py`
- `scripts/scid_slice_to_parquet.py`

### Labels
`micro-agent`, `batch`, `scid`
```

---

### Epic 9: Monitoring & Metrics Agent

```markdown
## Issue #23: [Epic] Monitoring & Metrics Micro Agent

### Description
Unified observability layer for all pipeline components.

### Components
- `src/metrics_exporter.py` - Prometheus metrics
- `src/services/metrics.py` - Metrics utilities
- `monitoring/metrics_exporter/` - Standalone exporter
- `monitoring/prometheus/` - Prometheus configs
- `monitoring/grafana/` - Dashboard definitions

### Responsibilities
- [ ] Export Prometheus metrics
- [ ] Aggregate service health statuses
- [ ] Provide Grafana dashboards
- [ ] Alert on anomalies

### Acceptance Criteria
- [ ] All micro agents expose `/metrics` endpoint
- [ ] Dashboards cover all data flows
- [ ] Alerting rules for critical failures

### Labels
`micro-agent`, `monitoring`, `observability`, `epic`
```

---

```markdown
## Issue #24: Unified Health Check System

### Parent Epic
#23 Monitoring & Metrics Agent

### Description
Implement comprehensive health checks across all agents.

### Tasks
- [ ] Define `HealthCheck` protocol/interface
- [ ] Implement checks for each micro agent
- [ ] Create aggregated `/health/detailed` endpoint
- [ ] Add dependency health (Redis, DuckDB, APIs)
- [ ] Support liveness vs readiness probes

### Current Endpoints
- `GET /health` - Basic health
- `GET /status` - Detailed status

### Labels
`micro-agent`, `monitoring`, `health`
```

---

## Cross-Cutting Issues

```markdown
## Issue #25: Configuration Management Overhaul

### Description
Consolidate configuration across all micro agents.

### Current State
- `src/config.py` - Main settings (Pydantic)
- `src/config/*.py` - Additional configs
- `discord-bot/bot/config.py` - Bot config
- `.env` - Environment variables

### Tasks
- [ ] Create unified `Config` namespace
- [ ] Add config validation on startup
- [ ] Support config hot-reload where safe
- [ ] Document all configuration options
- [ ] Add config diff tooling

### Labels
`infrastructure`, `configuration`
```

---

```markdown
## Issue #26: Shared Library Extraction

### Description
Extract common utilities to `src/lib/` for reuse across agents.

### Candidates for Extraction
- [ ] Redis connection pooling
- [ ] Logging configuration
- [ ] Datetime utilities (timezone handling)
- [ ] Retry/backoff decorators
- [ ] JSON serialization helpers

### Current Lib Contents
- `src/lib/redis_client.py`
- `src/lib/gex_history_queue.py`

### Labels
`infrastructure`, `library`
```

---

```markdown
## Issue #27: Testing Infrastructure

### Description
Establish testing patterns for micro agent architecture.

### Tasks
- [ ] Create test fixtures for each agent boundary
- [ ] Add integration tests for agent communication
- [ ] Implement contract tests for Redis schemas
- [ ] Add performance benchmarks
- [ ] Set up CI pipeline for all agents

### Current Test Locations
- `tests/` - Root tests
- `src/tests/` - Src tests
- `backend/tests/` - Backend tests
- `discord-bot/tests/` - Bot tests

### Labels
`infrastructure`, `testing`
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- Issue #25: Configuration Management
- Issue #26: Shared Library Extraction
- Issue #27: Testing Infrastructure

### Phase 2: Core Data Flow (Weeks 3-5)
- Issue #1-3: Market Data Ingestion Agent
- Issue #7-8: Redis TimeSeries Agent
- Issue #9-11: Persistence & Flush Agent

### Phase 3: Data Services (Weeks 6-8)
- Issue #4-6: GEX Data Services Agent
- Issue #18-19: UW Message Processing Agent

### Phase 4: User-Facing (Weeks 9-11)
- Issue #12-14: API Gateway Agent
- Issue #15-17: Discord Bot Agent

### Phase 5: Operations (Weeks 12-13)
- Issue #20-22: Batch Processing Agent
- Issue #23-24: Monitoring & Metrics Agent

---

## Agent Communication Patterns

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway                               │
│  (FastAPI - Routes requests, serves WebSocket, status dashboard) │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ Market Data   │  │ GEX Data      │  │ UW Message    │
│ Ingestion     │  │ Services      │  │ Processing    │
│ (Schwab, TT)  │  │ (Polling)     │  │ (Alerts)      │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Redis TimeSeries      │
              │   (Hot data, pub/sub)   │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  Persistence & Flush    │
              │  (DuckDB, Parquet)      │
              └─────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ Discord Bot   │  │ Batch Jobs    │  │ Monitoring    │
│ (Commands)    │  │ (Historical)  │  │ (Metrics)     │
└───────────────┘  └───────────────┘  └───────────────┘
```

---

## Redis Channel Schema

| Channel | Publisher | Subscribers |
|---------|-----------|-------------|
| `market_data:tastytrade:trades` | Market Data Agent | Discord Bot, Persistence |
| `gex:snapshot:stream` | GEX Services | Discord Bot, API Gateway |
| `uw:market_agg:stream` | UW Processing | Discord Bot, Alert Service |
| `uw:option_trade:stream` | UW Processing | Discord Bot, Alert Service |

---

## DuckDB Table Ownership

| Table | Owner Agent | Readers |
|-------|-------------|---------|
| `gex_snapshots` | Persistence | API Gateway, Batch Jobs |
| `tick_data` | Persistence | API Gateway, Enrichment |
| `depth_data` | Persistence | API Gateway, Enrichment |
| `import_jobs` | GEX Services | API Gateway |

---

## Next Steps

1. Review and prioritize issues with team
2. Create GitHub issues from templates above
3. Assign owners to each epic
4. Begin Phase 1 foundation work
5. Establish weekly sync for cross-agent coordination
