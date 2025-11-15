# Implementation Plan: Schwab Real-Time GEX Support

**Branch**: `001-schwab-realtime-gex` | **Date**: November 10, 2025 | **Spec**: [link](../spec.md)  
**Input**: Feature specification from `/specs/001-schwab-realtime-gex/spec.md`

**Note**: This plan was produced via the `/speckit.plan` workflow outlined in `.github/prompts/speckit.plan.prompt.md`.

## Summary

Integrate Schwab’s trading API to ingest authenticated, real-time equity and options data, keep the current trading day’s market data hot in Redis (or an alternative shared-memory store such as Dragonfly/Memcached) for instant retrieval, and expose ultra-low-latency Gamma Exposure (GEX) calculations through FastAPI endpoints. Research confirms an OAuth2 + WebSocket client, Redis time-series caching, and Polars-based vectorized GEX engine will meet the success criteria (<30s connect, <50 ms cache hit, <100 ms GEX response).

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: FastAPI, Pydantic, Polars, DuckDB, redis-py, websocket-client, httpx, python-dotenv  
**Storage**: Redis time-series/streams for full-day market data, optional Dragonfly/Memcached if Redis unavailable, in-memory LRU cache for computed GEX, DuckDB/Parquet for persistence and replay  
**Testing**: pytest, pytest-asyncio, ruff; contract tests for OpenAPI schema, integration tests for streaming + caching loops  
**Target Platform**: Linux server deployment with Redis and DuckDB co-located (dev) or managed (prod)  
**Project Type**: Backend data pipeline (monorepo with `backend/` services + monitoring UI)  
**Performance Goals**: Connect to Schwab within 30 s, serve cache hits <50 ms, complete GEX calculations <100 ms, sustain 100 concurrent GEX requests with p95 <200 ms  
**Constraints**: Must obey Schwab rate limits (120–500 RPM), auto-recover connections in <10 s, maintain cache freshness <5 min, log all Schwab + cache operations, ensure memory footprint bounded via TTL/segment eviction  
**Scale/Scope**: Stream up to 1000 symbols per WebSocket, store rolling 5 min of tick + options deltas, deliver results to API consumers + internal analytics

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

No custom constitution is defined (`.specify/memory/constitution.md` is template). Default gates applied:

**Pre-Phase 1**
- [x] Feature specification completed with priorities, requirements, and success criteria
- [x] Technical context documented (see section above)
- [x] Repository structure supports backend data pipeline work
- [x] Scope fits performance + complexity expectations

**Post-Phase 1**
- [x] Data models, contracts, and quickstart generated
- [x] Research questions resolved (no remaining NEEDS CLARIFICATION items)
- [x] Agent context updated via `.specify/scripts/bash/update-agent-context.sh copilot`
- [x] Deliverables meet quality bars (validated schemas + OpenAPI)

## Project Structure

### Documentation (this feature)

```text
specs/001-schwab-realtime-gex/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── gex-api.yaml
└── checklists/ (feature-specific QA/regression items)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── api/               # FastAPI routers; add schwab_gex routes + auth callback
│   ├── models/            # Pydantic models (extend with Schwab/GEX schemas)
│   ├── services/
│   │   ├── schwab/        # NEW: OAuth client, WebSocket stream, rate limiting
│   │   └── gex/           # Vectorized GEX calculator & cache orchestrator
│   ├── caching/           # Redis + in-memory cache utilities (new helpers)
│   └── config.py          # ENV wiring for Schwab + Redis settings
└── tests/
    ├── unit/              # Service + model tests (token refresh, GEX math)
    ├── integration/       # Redis + WebSocket streaming loops with fakes
    └── contract/          # OpenAPI + schema validation against gex-api.yaml
```

**Structure Decision**: Extend existing `backend/src` FastAPI service with Schwab-specific clients, caching utilities, and GEX services; keep frontend untouched. Tests stay under `backend/tests` mirroring new modules.

## Complexity Tracking

No constitution violations. Added sub-packages under `backend/src/services` instead of new top-level projects to minimize blast radius.

## Phase 0: Research & Analysis

**Objective**: Resolve Schwab API unknowns, caching strategy, and GEX calculation approach.

**Deliverable**: [`research.md`](../001-schwab-realtime-gex/research.md) (Complete ✅)

**Highlights**:
- OAuth2 PKCE w/ automatic refresh + jittered retries
- WebSocket primary feed w/ REST fallback
- Redis sorted sets + hashes + pub/sub for ticks/options
- Vectorized Polars GEX engine w/ pre-computed greeks
- Circuit breaker + connection health metrics

## Phase 1: Design & Architecture

**Objective**: Capture implementation design artifacts for development handoff.

**Deliverables** (Complete ✅):
- [`data-model.md`](../001-schwab-realtime-gex/data-model.md): SchwabConnection, MarketData, OptionData, GEXCalculation, CacheEntry, telemetry models
- [`contracts/gex-api.yaml`](../001-schwab-realtime-gex/contracts/gex-api.yaml): OpenAPI 3.0.3 covering health, market, options, GEX endpoints
- [`quickstart.md`](../001-schwab-realtime-gex/quickstart.md): Environment, auth, sample clients
- Agent context updated for Copilot (`.github/copilot-instructions.md`)

**Design Focus Areas**:
- OAuth2 handler + refresh daemon (async tasks + redis-stored metadata)
- Streaming pipeline → Redis sorted sets + fan-out channels
- GEX service using Polars frames + in-memory LRU caching
- Observability: structured logging, metrics for connection health + cache hit rate

## Phase 2: Implementation & Testing

**Objective**: Build, test, and verify production readiness.

**Deliverables**:
- Schwab client + WebSocket streamer with retry + heartbeat
- Redis (or equivalent in-memory store) cache module that retains intraday/daily market data with TTL enforcement + pub/sub fan-out
- GEX calculation service + FastAPI routes mapped to OpenAPI contract
- Comprehensive tests (unit, integration, contract) + benchmarks
- `tasks.md` capturing granular implementation tracker

**Implementation Phases**:
1. **Connectivity Foundations**: OAuth2 PKCE helper, secrets loading, token refresh jobs, health endpoints.
2. **Streaming + Caching**: WebSocket consumer, Redis (or alternate in-memory store) ingestion to hold the active trading day, stale-data detection, backpressure + rate-limit guardrails.
3. **GEX Engine + API**: Polars/NumPy GEX computation, concurrency-safe caches, API serialization, error responses.
4. **Reliability & Monitoring**: Circuit breaker, reconnection policies, structured logs, metrics export, resilient startup/shutdown.
5. **Validation**: Unit + property tests for greeks/gamma math, integration tests with mocked Schwab feed, contract tests vs `gex-api.yaml`, soak/perf tests targeting success criteria.

**Status**: ✅ PLANNED - Implementation tasks defined in `tasks.md`, ready for development

## Phase 3: Integration & Deployment

**Objective**: Harden feature for production use.

**Deliverables**:
- Redis/DuckDB sizing + retention policies
- Deployment runbooks (OAuth secret management, rotation)
- Monitoring dashboards + alerts for connection health, cache hit rate, GEX latency
- Ops documentation + rollout checklist
