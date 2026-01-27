#!/usr/bin/env bash
# Create GitHub Issues for Micro Agent Architecture
# Usage: ./create_github_issues.sh
# Requires: gh CLI authenticated

set -e

REPO="sk1nt/data-pipeline"  # Update with your repo

echo "Creating Micro Agent Architecture Issues..."

# ============================================================================
# EPIC 1: Market Data Ingestion Agent
# ============================================================================

gh issue create --repo "$REPO" \
  --title "[Epic] Market Data Ingestion Micro Agent" \
  --label "epic,micro-agent,market-data,streaming" \
  --body "## Description
Consolidate all real-time market data streaming into a self-contained micro agent with clear boundaries.

## Components
- \`src/services/schwab_streamer.py\` - Schwab WebSocket client (1101 lines)
- \`src/services/tastytrade_streamer.py\` - TastyTrade DXLink streamer (200 lines)
- \`src/services/tastytrade_client.py\` - TastyTrade REST client
- \`src/models/market_data.py\` - TickEvent, Level2Event, Level2Quote models
- \`src/token_store.py\` - OAuth token management

## Responsibilities
- [ ] Stream real-time tick data (trades)
- [ ] Stream Level 2 / depth-of-book data
- [ ] Handle OAuth token refresh lifecycle
- [ ] Normalize data formats across feeds
- [ ] Publish to Redis pub/sub channels

## Interface Contracts
**Input:** OAuth credentials, symbol lists
**Output:** \`market_data:tastytrade:trades\`, \`market_data:schwab:*\` Redis channels

## Acceptance Criteria
- [ ] Agent can start/stop independently
- [ ] Reconnection logic handles network failures
- [ ] Token refresh works without manual intervention
- [ ] Metrics exposed: connection status, message counts, latency"

gh issue create --repo "$REPO" \
  --title "Schwab Streamer Refactor" \
  --label "micro-agent,refactor,schwab" \
  --body "## Parent Epic
Market Data Ingestion Agent

## Description
Refactor \`schwab_streamer.py\` (1101 lines) into modular components.

## Tasks
- [ ] Extract \`SchwabAuthClient\` to \`src/services/auth/schwab_auth.py\`
- [ ] Extract \`SchwabToken\` dataclass to \`src/models/auth.py\`
- [ ] Create \`SchwabStreamHandler\` for message parsing
- [ ] Add retry/backoff logic as configurable middleware
- [ ] Unit test token refresh cycle

## Files
- \`src/services/schwab_streamer.py\`
- \`src/token_store.py\`
- \`src/config.py\`"

gh issue create --repo "$REPO" \
  --title "TastyTrade Streamer Enhancement" \
  --label "micro-agent,tastytrade,enhancement" \
  --body "## Parent Epic
Market Data Ingestion Agent

## Description
Enhance TastyTrade DXLink integration for parity with Schwab streamer.

## Tasks
- [ ] Add depth-of-book streaming (currently optional)
- [ ] Implement reconnection with exponential backoff
- [ ] Add connection health heartbeat
- [ ] Normalize output format to match Schwab events
- [ ] Add session persistence for faster reconnects

## Files
- \`src/services/tastytrade_streamer.py\`
- \`discord-bot/bot/tastytrade_client.py\`"

# ============================================================================
# EPIC 2: GEX Data Services Agent
# ============================================================================

gh issue create --repo "$REPO" \
  --title "[Epic] GEX Data Services Micro Agent" \
  --label "epic,micro-agent,gex" \
  --body "## Description
Unified agent for all GEX (Gamma Exposure) data operations.

## Components
- \`src/services/gexbot_poller.py\` - GEXBot API polling (662 lines)
- \`src/importers/gex_importer.py\` - Historical GEX file imports
- \`src/import_gex_history.py\` - Queue-based import processor
- \`src/lib/gex_history_queue.py\` - Import job queue
- \`src/models/gex_snapshot.py\` - GEX data models

## Responsibilities
- [ ] Poll GEXBot API at configurable intervals
- [ ] Cache snapshots in Redis with pub/sub notifications
- [ ] Process historical import queue
- [ ] Write to Sierra Chart format for external tools
- [ ] Maintain symbol lists (dynamic refresh)

## Interface Contracts
**Input:** GEXBot API key, symbol configurations
**Output:** \`gex:snapshot:*\` Redis keys, \`gex:snapshot:stream\` pub/sub

## Acceptance Criteria
- [ ] Polling adapts to RTH vs off-hours schedules
- [ ] Historical imports don't block real-time polling
- [ ] Sierra Chart files update atomically"

gh issue create --repo "$REPO" \
  --title "GEXBot Poller Service Isolation" \
  --label "micro-agent,gex,refactor" \
  --body "## Parent Epic
GEX Data Services Agent

## Description
Isolate GEXBot polling as a standalone service with clean interfaces.

## Tasks
- [ ] Extract settings to dedicated \`GEXBotConfig\` class
- [ ] Create abstract \`GEXDataSource\` interface
- [ ] Implement \`GEXBotAPISource\` implementing the interface
- [ ] Add circuit breaker for API rate limiting
- [ ] Separate NQ poller logic from main poller

## Files
- \`src/services/gexbot_poller.py\`"

gh issue create --repo "$REPO" \
  --title "GEX History Import Pipeline" \
  --label "micro-agent,gex,pipeline" \
  --body "## Parent Epic
GEX Data Services Agent

## Description
Streamline historical GEX data imports with better job tracking.

## Tasks
- [ ] Create \`GEXImportJob\` model with states (queued, processing, completed, failed)
- [ ] Add import deduplication by URL hash
- [ ] Implement retry logic with exponential backoff
- [ ] Add import progress tracking via Redis
- [ ] Create CLI for manual import operations

## Files
- \`src/import_gex_history.py\`
- \`src/lib/gex_history_queue.py\`
- \`src/importers/gex_importer.py\`"

# ============================================================================
# EPIC 3: Redis TimeSeries Agent
# ============================================================================

gh issue create --repo "$REPO" \
  --title "[Epic] Redis TimeSeries Micro Agent" \
  --label "epic,micro-agent,redis,timeseries" \
  --body "## Description
Centralize all Redis time-series operations and caching logic.

## Components
- \`src/services/redis_timeseries.py\` - TimeSeries client wrapper
- \`src/lib/redis_client.py\` - Base Redis connection
- \`src/services/lookup_service.py\` - Trade/depth lookups
- \`backend/src/services/redis_service.py\` - Alternative Redis helpers

## Responsibilities
- [ ] Manage Redis TimeSeries keys for trades/depth
- [ ] Provide lookup APIs for historical data
- [ ] Handle cross-feed depth comparisons
- [ ] Manage key TTLs and retention policies

## Interface Contracts
**Input:** Normalized market data events
**Output:** \`ts:trade:*\`, \`ts:depth:*\` TimeSeries keys

## Acceptance Criteria
- [ ] Consistent key naming across all services
- [ ] Automatic key creation with proper labels
- [ ] Memory-efficient aggregation rules"

gh issue create --repo "$REPO" \
  --title "Lookup Service Consolidation" \
  --label "micro-agent,redis,refactor" \
  --body "## Parent Epic
Redis TimeSeries Agent

## Description
Unify lookup operations across the codebase.

## Tasks
- [ ] Merge \`LookupService\` with backend Redis services
- [ ] Add caching layer for frequent queries
- [ ] Implement pagination for large result sets
- [ ] Add query optimization hints

## Files
- \`src/services/lookup_service.py\`
- \`backend/src/services/redis_service.py\`"

# ============================================================================
# EPIC 4: Persistence & Flush Agent
# ============================================================================

gh issue create --repo "$REPO" \
  --title "[Epic] Persistence & Flush Micro Agent" \
  --label "epic,micro-agent,persistence,duckdb,parquet" \
  --body "## Description
Handle all durable storage operations: DuckDB, Parquet, data lifecycle.

## Components
- \`src/services/redis_flush_worker.py\` - TimeSeries → DuckDB/Parquet (919 lines)
- \`src/db/duckdb_utils.py\` - DuckDB operations
- \`src/importers/tick_importer.py\` - Tick data imports
- \`src/importers/depth_importer.py\` - Depth data imports
- \`backend/src/services/duckdb_service.py\` - Backend DuckDB wrapper

## Responsibilities
- [ ] Flush Redis TimeSeries to DuckDB on schedule
- [ ] Write Parquet files for archival
- [ ] Manage data retention and purging
- [ ] Handle GEX snapshot persistence
- [ ] Support tick/depth imports from external files

## Interface Contracts
**Input:** Redis TimeSeries data, import files
**Output:** DuckDB tables, Parquet files in \`data/parquet/\`

## Acceptance Criteria
- [ ] Flush operations are atomic/transactional
- [ ] Parquet files are partitioned by date
- [ ] Old data purging follows retention policy"

gh issue create --repo "$REPO" \
  --title "Redis Flush Worker Refactor" \
  --label "micro-agent,refactor,persistence" \
  --body "## Parent Epic
Persistence & Flush Agent

## Description
Break down the 919-line flush worker into focused components.

## Tasks
- [ ] Extract \`TimeSeriesFlusher\` for TS → DuckDB
- [ ] Extract \`ParquetWriter\` for file operations
- [ ] Extract \`GEXSnapshotPersister\` for GEX-specific logic
- [ ] Create \`FlushScheduler\` for timing logic
- [ ] Add flush status/progress API

## Files
- \`src/services/redis_flush_worker.py\`"

gh issue create --repo "$REPO" \
  --title "Data Retention & Purge Service" \
  --label "micro-agent,persistence,cleanup" \
  --body "## Parent Epic
Persistence & Flush Agent

## Description
Implement configurable data retention policies.

## Tasks
- [ ] Create \`RetentionPolicy\` model (per data type)
- [ ] Implement \`PurgeService\` for scheduled cleanup
- [ ] Add CLI for manual purge operations
- [ ] Support dry-run mode for purge preview
- [ ] Log purge operations for audit

## Related Scripts
- \`scripts/purge_old_data.py\`"

# ============================================================================
# EPIC 5: API Gateway Agent
# ============================================================================

gh issue create --repo "$REPO" \
  --title "[Epic] API Gateway Micro Agent" \
  --label "epic,micro-agent,api,fastapi" \
  --body "## Description
Consolidate all HTTP/WebSocket endpoints into a unified gateway.

## Components
- \`data-pipeline.py\` - Main FastAPI app (900+ lines)
- \`src/api/app.py\` - Secondary API surface
- \`src/api/gex_api.py\` - GEX endpoints
- \`src/api/tick_api.py\` - Tick data endpoints
- \`src/api/depth_api.py\` - Depth data endpoints
- \`src/api/health.py\` - Health checks
- \`backend/src/api/\` - Backend API layer

## Responsibilities
- [ ] Route requests to appropriate services
- [ ] Handle authentication/authorization
- [ ] Expose WebSocket streams
- [ ] Serve status dashboard

## Current Issues
- Two separate FastAPI apps
- Endpoint logic mixed with service orchestration

## Acceptance Criteria
- [ ] Single entrypoint for all API traffic
- [ ] OpenAPI documentation complete
- [ ] Request/response logging"

gh issue create --repo "$REPO" \
  --title "Merge API Surfaces" \
  --label "micro-agent,api,refactor" \
  --body "## Parent Epic
API Gateway Agent

## Description
Consolidate \`data-pipeline.py\` endpoints with \`src/api/\` modules.

## Tasks
- [ ] Move inline endpoints from \`data-pipeline.py\` to \`src/api/routes/\`
- [ ] Create \`src/api/routes/control.py\` for service control
- [ ] Create \`src/api/routes/lookup.py\` for lookup endpoints
- [ ] Create \`src/api/routes/sierra_chart.py\` for SC bridge
- [ ] Keep \`data-pipeline.py\` as pure orchestration entrypoint

## Files
- \`data-pipeline.py\`
- \`src/api/app.py\`
- \`src/api/routes/\`"

gh issue create --repo "$REPO" \
  --title "WebSocket Streaming Consolidation" \
  --label "micro-agent,api,websocket" \
  --body "## Parent Epic
API Gateway Agent

## Description
Create unified WebSocket handler for all streaming endpoints.

## Tasks
- [ ] Create \`WebSocketManager\` for connection lifecycle
- [ ] Implement topic-based subscription model
- [ ] Add \`/ws/trades\`, \`/ws/depth\`, \`/ws/gex\` endpoints
- [ ] Support reconnection with state replay
- [ ] Add connection metrics

## Current Endpoints
- \`/ws/sc\` - Sierra Chart GEX stream"

# ============================================================================
# EPIC 6: Discord Bot Agent
# ============================================================================

gh issue create --repo "$REPO" \
  --title "[Epic] Discord Bot Micro Agent" \
  --label "epic,micro-agent,discord,trading" \
  --body "## Description
Isolate Discord bot as a fully independent service.

## Components
- \`discord-bot/bot/trade_bot.py\` - Main bot (3868 lines)
- \`discord-bot/bot/tastytrade_client.py\` - Trading client
- \`discord-bot/bot/config.py\` - Bot configuration
- \`discord-bot/run_discord_bot.py\` - Entrypoint

## Responsibilities
- [ ] Process Discord commands (!gex, !status, etc.)
- [ ] Execute trades via TastyTrade
- [ ] Forward alerts from Redis pub/sub
- [ ] Manage user permissions/allowlists

## Current Issues
- 3868 lines in single file
- Mixed concerns: commands, trading, alerts, feeds

## Acceptance Criteria
- [ ] Bot can run as standalone process
- [ ] Commands are modular/pluggable
- [ ] Trading operations are audited"

gh issue create --repo "$REPO" \
  --title "Discord Bot Command Modularization" \
  --label "micro-agent,discord,refactor" \
  --body "## Parent Epic
Discord Bot Agent

## Description
Split monolithic trade_bot.py into command modules.

## Tasks
- [ ] Create \`discord-bot/bot/commands/\` directory
- [ ] Extract \`gex_commands.py\` for GEX lookups
- [ ] Extract \`trade_commands.py\` for order execution
- [ ] Extract \`status_commands.py\` for system status
- [ ] Use discord.py Cogs pattern

## Target Structure
\`\`\`
discord-bot/bot/
├── commands/
│   ├── __init__.py
│   ├── gex.py
│   ├── trading.py
│   └── status.py
├── services/
│   └── tastytrade_executor.py
└── trade_bot.py (reduced to wiring)
\`\`\`"

gh issue create --repo "$REPO" \
  --title "TastyTrade Trading Integration Hardening" \
  --label "micro-agent,discord,trading,safety" \
  --body "## Parent Epic
Discord Bot Agent

## Description
Harden TastyTrade order execution with proper safeguards.

## Tasks
- [ ] Add order validation service
- [ ] Implement position limits
- [ ] Add kill switch for emergency stop
- [ ] Create audit log for all trades
- [ ] Add order confirmation workflow

## Files
- \`discord-bot/bot/tastytrade_client.py\`
- \`src/services/order_validation.py\`"

# ============================================================================
# EPIC 7: UW Message Processing Agent
# ============================================================================

gh issue create --repo "$REPO" \
  --title "[Epic] Unusual Whales Message Micro Agent" \
  --label "epic,micro-agent,uw,alerts" \
  --body "## Description
Dedicated agent for processing Unusual Whales data feeds.

## Components
- \`src/services/uw_message_service.py\` - Message router (254 lines)
- \`src/services/market_agg_alert_service.py\` - Market alerts
- \`src/models/uw_message.py\` - Message models

## Responsibilities
- [ ] Parse Phoenix WebSocket messages
- [ ] Route to option trade or market agg handlers
- [ ] Store in Redis for downstream consumers
- [ ] Trigger alerts based on rules

## Interface Contracts
**Input:** Raw Phoenix arrays \`[joinRef, ref, topic, eventType, payload]\`
**Output:** \`uw:market_agg:*\`, \`uw:option_trade:*\` Redis keys"

gh issue create --repo "$REPO" \
  --title "UW Alert Rule Engine" \
  --label "micro-agent,uw,alerts" \
  --body "## Parent Epic
Unusual Whales Message Agent

## Description
Implement configurable alert rules for UW data.

## Tasks
- [ ] Create \`AlertRule\` model (conditions, actions)
- [ ] Implement rule evaluation engine
- [ ] Add rule persistence to Redis/DuckDB
- [ ] Support Discord, webhook, and pubsub actions
- [ ] Add rule management API

## Files
- \`src/services/rule_engine.py\`
- \`src/models/alert.py\`"

# ============================================================================
# EPIC 8: Batch Processing Agent
# ============================================================================

gh issue create --repo "$REPO" \
  --title "[Epic] Batch Processing Micro Agent" \
  --label "epic,micro-agent,batch,pipeline" \
  --body "## Description
Consolidate all batch/offline processing scripts.

## Components
- \`scripts/orchestrator.py\` - Parallel date-range processing
- \`scripts/worker_day.py\` - Single-day processor
- \`scripts/import_scid_ticks.py\` - SCID tick imports
- \`scripts/export_scid_ticks_to_parquet.py\` - SCID → Parquet
- \`scripts/enrich_tick_gex.py\` - GEX enrichment

## Responsibilities
- [ ] Historical data backfills
- [ ] Format conversions (SCID → Parquet)
- [ ] Data enrichment pipelines
- [ ] Schema migrations

## Acceptance Criteria
- [ ] Jobs can run in parallel where safe
- [ ] Progress tracking and resumability
- [ ] Consistent logging and error handling"

gh issue create --repo "$REPO" \
  --title "Batch Job Framework" \
  --label "micro-agent,batch,framework" \
  --body "## Parent Epic
Batch Processing Agent

## Description
Create a unified framework for batch job execution.

## Tasks
- [ ] Create \`BatchJob\` base class with lifecycle hooks
- [ ] Implement \`JobRunner\` with parallelization
- [ ] Add progress persistence for resumability
- [ ] Create \`JobReport\` for execution summaries
- [ ] Add CLI for job management"

gh issue create --repo "$REPO" \
  --title "SCID Processing Pipeline" \
  --label "micro-agent,batch,scid" \
  --body "## Parent Epic
Batch Processing Agent

## Description
Standardize Sierra Chart SCID file processing.

## Tasks
- [ ] Create \`SCIDReader\` utility class
- [ ] Implement streaming parser for large files
- [ ] Add validation for SCID integrity
- [ ] Support multiple contract roll schedules
- [ ] Document CME contract windows

## Files
- \`scripts/import_scid_ticks.py\`
- \`scripts/export_scid_ticks_to_parquet.py\`"

# ============================================================================
# EPIC 9: Monitoring & Metrics Agent
# ============================================================================

gh issue create --repo "$REPO" \
  --title "[Epic] Monitoring & Metrics Micro Agent" \
  --label "epic,micro-agent,monitoring,observability" \
  --body "## Description
Unified observability layer for all pipeline components.

## Components
- \`src/metrics_exporter.py\` - Prometheus metrics
- \`src/services/metrics.py\` - Metrics utilities
- \`monitoring/metrics_exporter/\` - Standalone exporter
- \`monitoring/prometheus/\` - Prometheus configs
- \`monitoring/grafana/\` - Dashboard definitions

## Responsibilities
- [ ] Export Prometheus metrics
- [ ] Aggregate service health statuses
- [ ] Provide Grafana dashboards
- [ ] Alert on anomalies

## Acceptance Criteria
- [ ] All micro agents expose \`/metrics\` endpoint
- [ ] Dashboards cover all data flows
- [ ] Alerting rules for critical failures"

gh issue create --repo "$REPO" \
  --title "Unified Health Check System" \
  --label "micro-agent,monitoring,health" \
  --body "## Parent Epic
Monitoring & Metrics Agent

## Description
Implement comprehensive health checks across all agents.

## Tasks
- [ ] Define \`HealthCheck\` protocol/interface
- [ ] Implement checks for each micro agent
- [ ] Create aggregated \`/health/detailed\` endpoint
- [ ] Add dependency health (Redis, DuckDB, APIs)
- [ ] Support liveness vs readiness probes"

# ============================================================================
# Cross-Cutting Issues
# ============================================================================

gh issue create --repo "$REPO" \
  --title "Configuration Management Overhaul" \
  --label "infrastructure,configuration" \
  --body "## Description
Consolidate configuration across all micro agents.

## Current State
- \`src/config.py\` - Main settings (Pydantic)
- \`discord-bot/bot/config.py\` - Bot config
- \`.env\` - Environment variables

## Tasks
- [ ] Create unified \`Config\` namespace
- [ ] Add config validation on startup
- [ ] Support config hot-reload where safe
- [ ] Document all configuration options
- [ ] Add config diff tooling"

gh issue create --repo "$REPO" \
  --title "Shared Library Extraction" \
  --label "infrastructure,library" \
  --body "## Description
Extract common utilities to \`src/lib/\` for reuse across agents.

## Candidates for Extraction
- [ ] Redis connection pooling
- [ ] Logging configuration
- [ ] Datetime utilities (timezone handling)
- [ ] Retry/backoff decorators
- [ ] JSON serialization helpers

## Current Lib Contents
- \`src/lib/redis_client.py\`
- \`src/lib/gex_history_queue.py\`"

gh issue create --repo "$REPO" \
  --title "Testing Infrastructure for Micro Agents" \
  --label "infrastructure,testing" \
  --body "## Description
Establish testing patterns for micro agent architecture.

## Tasks
- [ ] Create test fixtures for each agent boundary
- [ ] Add integration tests for agent communication
- [ ] Implement contract tests for Redis schemas
- [ ] Add performance benchmarks
- [ ] Set up CI pipeline for all agents

## Current Test Locations
- \`tests/\` - Root tests
- \`src/tests/\` - Src tests
- \`backend/tests/\` - Backend tests
- \`discord-bot/tests/\` - Bot tests"

echo "Done! Created all micro agent architecture issues."
