# Tasks: High-Speed GEX Data Ingest Priorities with Firm Guidelines and Methodology

**Input**: Design documents from `/specs/001-data-ingest-priorities/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are OPTIONAL - not explicitly requested in feature specification. Examples shown for illustration.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

Based on plan.md: Single project structure with `src/`, `tests/` at repository root

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create project directory structure per plan.md
- [x] T002 Initialize Python 3.11 project with Poetry/pip and core dependencies (Polars, DuckDB, FastAPI, Pydantic, Redis)
- [x] T003 [P] Configure ruff linting and pytest testing framework
- [x] T004 [P] Setup environment configuration management (.env files)
- [x] T005 Create basic logging configuration in src/lib/logging.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Setup DuckDB database connection and schema management in src/lib/database.py
- [x] T007 [P] Setup Redis connection and caching utilities in src/lib/redis_client.py
- [x] T008 [P] Create base Pydantic models and enums from data-model.md in src/models/base.py
- [x] T009 [P] Implement FastAPI application structure with middleware in src/api/app.py
- [x] T010 Create shared utilities for data validation and processing in src/lib/utils.py
- [x] T011 Setup error handling and custom exceptions in src/lib/exceptions.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Define GEX Data Source Priorities (Priority: P1) 🎯 MVP

**Goal**: Enable defining and managing priority levels for GEX data sources with basic API endpoints

**Independent Test**: Can submit priority requests and retrieve data source information independently

### Implementation for User Story 1

- [x] T012 [P] [US1] Create PriorityRequest model in src/models/priority_request.py
- [x] T013 [P] [US1] Create DataSource model in src/models/data_source.py
- [x] T014 [P] [US1] Create PriorityLevel enum in src/models/enums.py
- [x] T015 [US1] Implement basic priority service for request management in src/services/priority_service.py
- [x] T016 [US1] Create POST /ingest/priority endpoint in src/api/routes/priority.py
- [x] T017 [US1] Create GET /sources endpoint in src/api/routes/priority_routes.py
- [x] T018 [US1] Add request validation and basic error handling
- [x] T019 [US1] Add logging for priority operations

**Checkpoint**: At this point, User Story 1 should be fully functional - can submit priority requests and list data sources

---

## Phase 4: User Story 2 - Apply Firm Guidelines for High-Speed GEX Prioritization (Priority: P2)

**Goal**: Implement automatic priority assignment based on firm guidelines and market data characteristics

**Independent Test**: Can verify that priority rules are applied correctly to automatically assign priorities

### Implementation for User Story 2

- [ ] T020 [P] [US2] Create PriorityRule model in src/models/priority_rule.py
- [ ] T021 [US2] Implement priority rule engine for automatic assignment in src/services/rule_engine.py
- [ ] T022 [US2] Create rule evaluation logic with market impact and freshness criteria
- [ ] T023 [US2] Integrate rule engine with priority service from US1
- [ ] T024 [US2] Create GET /rules endpoint in src/api/routes/rules.py
- [ ] T025 [US2] Add rule validation and conflict resolution
- [ ] T026 [US2] Add audit logging for automatic priority decisions

**Checkpoint**: At this point, User Stories 1 AND 2 should work - automatic priority assignment based on guidelines

---

## Phase 5: User Story 3 - Execute High-Speed Methodology for GEX Processing (Priority: P3)

**Goal**: Implement the processing pipeline with priority-based queuing and high-speed execution

**Independent Test**: Can verify that critical priority data is processed within time guarantees

### Implementation for User Story 3

- [ ] T027 [P] [US3] Create ProcessingJob model in src/models/processing_job.py
- [ ] T028 [P] [US3] Create GEXSnapshot and GEXStrike models in src/models/gex_data.py
- [ ] T029 [US3] Implement priority queue management with Redis sorted sets in src/services/queue_service.py
- [ ] T030 [US3] Create high-speed processing pipeline with Polars optimization in src/services/processing_service.py
- [ ] T031 [US3] Create GET /ingest/priority/{request_id} endpoint for status tracking
- [ ] T032 [US3] Create GET /ingest/priority/queue endpoint for queue monitoring
- [ ] T033 [US3] Implement Parquet export for processed GEX data
- [ ] T034 [US3] Add performance monitoring and latency tracking
- [ ] T035 [US3] Integrate with existing GEX import workflows

**Checkpoint**: All user stories should now be independently functional with full priority-based processing

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T036 [P] Add comprehensive API documentation and OpenAPI schema validation
- [ ] T037 Implement health checks and monitoring endpoints
- [ ] T038 [P] Add performance benchmarks and optimization validation
- [ ] T039 Create CLI commands for priority management in src/cli/priority_commands.py
- [ ] T040 [P] Add configuration management for production deployment
- [ ] T041 Run quickstart.md validation and update documentation
- [ ] T042 Add security hardening and input sanitization
- [ ] T043 Implement graceful shutdown and cleanup procedures

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User stories can proceed in parallel (if staffed) or sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Integrates with US1 priority service
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Uses priority system from US1/US2

### Within Each User Story

- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all models for User Story 1 together:
Task: "Create PriorityRequest model in src/models/priority_request.py"
Task: "Create DataSource model in src/models/data_source.py"
Task: "Create PriorityLevel enum in src/models/enums.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently - can define priorities and submit requests
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP: Basic priority system!)
3. Add User Story 2 → Test independently → Deploy/Demo (Automatic guidelines)
4. Add User Story 3 → Test independently → Deploy/Demo (High-speed processing)
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (Priority management)
   - Developer B: User Story 2 (Rule engine)
   - Developer C: User Story 3 (Processing pipeline)
3. Stories complete and integrate independently

---

## Success Criteria Validation

- **SC-001**: User Story 3 ensures critical data processed within 30 seconds
- **SC-002**: User Story 3 maintains priority order with <1% deviation
- **SC-003**: User Story 2 allows guideline modifications without impacting processing
- **SC-004**: User Story 2 achieves 98% automatic priority assignment accuracy
- **SC-005**: User Story 3 maintains <10 seconds average latency during peak hours

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence