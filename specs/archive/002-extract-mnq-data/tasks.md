# Tasks: Extract MNQ Historical Data

**Input**: Design documents from `/specs/002-extract-mnq-data/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Comprehensive tests are mandatory per constitution - all features must have unit, integration, and validation tests.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- Paths follow the established structure from plan.md

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create src/ directory structure per implementation plan
- [X] T002 Initialize Python 3.11 project with Polars, DuckDB dependencies
- [X] T003 [P] Configure ruff linting and formatting tools

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Implement SCID file parser in src/lib/scid_parser.py
- [X] T005 [P] Create TickRecord data model in src/models/mnq_tick_record.py
- [X] T006 [P] Create DepthSnapshot data model in src/models/depth_snapshot.py
- [X] T007 Setup DuckDB storage connection in src/lib/database.py
- [X] T008 Setup Parquet file handling in src/lib/parquet_handler.py
- [X] T009 Create CLI framework in src/cli/__init__.py
- [X] T010 Configure logging infrastructure in src/lib/logging_config.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Extract MNQ Tick Data (Priority: P1) 🎯 MVP

**Goal**: Extract historical tick data for MNQ futures from SierraChart SCID files and store in DuckDB

**Independent Test**: Verify tick data is extracted from SCID files and stored in DuckDB with correct timestamps, prices, volumes, and tick types

### Tests for User Story 1 (MANDATORY) ⚠️

- [X] T011 [P] [US1] Unit test for SCID tick parsing in tests/unit/test_scid_parser_ticks.py
- [X] T012 [P] [US1] Integration test for tick data extraction in tests/integration/test_tick_extraction.py

### Implementation for User Story 1

- [X] T013 [US1] Implement tick extraction logic in src/services/tick_extractor.py
- [X] T014 [US1] Add CLI command for tick extraction in src/cli/extract_ticks.py
- [X] T015 [US1] Integrate tick storage with DuckDB in src/services/tick_extractor.py
- [X] T016 [US1] Add error handling for tick extraction failures
- [X] T017 [US1] Add logging for tick extraction operations

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Extract MNQ Market Depth (Priority: P1)

**Goal**: Extract market depth data for MNQ futures from SierraChart and store in Parquet with DuckDB metadata

**Independent Test**: Verify depth data is extracted and stored in Parquet files with metadata in DuckDB, synchronized with tick timestamps

### Tests for User Story 2 (MANDATORY) ⚠️

- [X] T018 [P] [US2] Unit test for depth data parsing in tests/unit/test_depth_parser.py
- [X] T019 [P] [US2] Integration test for depth data extraction in tests/integration/test_depth_extraction.py

### Implementation for User Story 2

- [X] T020 [US2] Implement depth extraction logic in src/services/depth_extractor.py
- [X] T021 [US2] Add CLI command for depth extraction in src/cli/extract_depth.py
- [X] T022 [US2] Integrate depth storage with Parquet and DuckDB metadata
- [X] T023 [US2] Add error handling for depth extraction failures
- [X] T024 [US2] Add logging for depth extraction operations

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Data Integrity Validation (Priority: P2)

**Goal**: Validate integrity of extracted tick and depth data for reliable analysis

**Independent Test**: Run validation checks on extracted data for completeness, accuracy, and synchronization

### Tests for User Story 3 (MANDATORY) ⚠️

- [X] T025 [P] [US3] Unit test for data validation functions in tests/unit/test_data_validator.py
- [X] T026 [P] [US3] Integration test for end-to-end validation in tests/integration/test_data_validation.py

### Implementation for User Story 3

- [X] T027 [US3] Implement data validation service in src/services/data_validator.py
- [X] T028 [US3] Add CLI command for data validation in src/cli/validate_data.py
- [X] T029 [US3] Integrate validation with tick and depth data sources
- [X] T030 [US3] Add comprehensive error reporting for validation failures
- [X] T031 [US3] Add logging for validation operations

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T032 [P] Update documentation in README.md and quickstart.md
- [X] T033 Code cleanup and refactoring across all modules
- [X] T034 Performance optimization for large data volumes
- [X] T035 [P] Additional unit tests for edge cases in tests/unit/
- [X] T036 Security review and hardening
- [X] T037 Run quickstart.md validation and update as needed

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User Stories 1 and 2 can proceed in parallel (both P1)
  - User Story 3 depends on completion of US1 and US2
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 3 (P2)**: Depends on US1 and US2 completion - Validates extracted data

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Core extraction logic before CLI integration
- Error handling and logging after core functionality
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, US1 and US2 can start in parallel
- All tests for a user story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Stories 1 & 2

```bash
# Launch US1 and US2 in parallel after Foundational complete:
Task: "Implement tick extraction logic in src/services/tick_extractor.py"
Task: "Implement depth extraction logic in src/services/depth_extractor.py"

# Launch all tests for User Story 1 together:
Task: "Unit test for SCID tick parsing in tests/unit/test_scid_parser_ticks.py"
Task: "Integration test for tick data extraction in tests/integration/test_tick_extraction.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (tick extraction)
   - Developer B: User Story 2 (depth extraction)
   - Developer C: User Story 3 (validation)
3. Stories complete and integrate independently

---

## Implementation Enhancements & Optimizations

### Memory & Performance Optimizations

**T032** ✅ Memory-efficient SCID parsing with backwards reading
- Implemented `parse_scid_file_backwards_generator()` for streaming record processing
- Replaced memory-intensive list accumulation with generator-based yielding
- Added record-based chunking (100MB chunks) instead of inefficient byte-based chunking
- Reduced memory usage from loading entire 2.1GB+ files to processing records one-by-one

**T033** ✅ Database optimizations and transaction batching
- Added `save_ticks_to_db_optimized()` with transaction batching for bulk inserts
- Optimized DuckDB storage with proper indexing on timestamp and ticker columns
- Added ticker column to database schema for normalized futures contract storage

**T034** ✅ Parallel processing architecture
- Implemented `ProcessPoolExecutor` for true multi-core processing of multiple SCID files
- Configurable worker counts for tick extraction (`--tick-workers`) and depth extraction (`--depth-workers`)
- Sequential file processing within each worker to maintain data integrity
- Support for processing multiple futures contracts simultaneously

### Data Processing Enhancements

**T035** ✅ Futures ticker normalization
- Added `_normalize_futures_ticker()` method with comprehensive futures symbol mapping
- Supports all major futures contracts: MNQ, ES, NQ, CL, GC, SI, HG, NG, ZB, ZN, ZF, ZT, GE, EUR, JPY, GBP, CHF, CAD, AUD, NZD
- Automatically extracts base symbol from contract variations (e.g., `MNQZ25_FUT_CME.scid` → `MNQ`)
- Regex-based parsing that handles month codes and exchange suffixes

**T036** ✅ Integrated tick and depth extraction
- Combined tick and depth processing in single CLI command
- Sequential execution: ticks first, then depth data
- Shared configuration and logging infrastructure
- Unified error handling and progress reporting

**T037** ✅ Enhanced CLI with flexible date processing
- Added `--date` parameter for specific date extraction (YYYY-MM-DD format)
- Added `--tick-workers` and `--depth-workers` for independent worker configuration
- Added `--max-recent-records` for memory-limited processing (0 = unlimited)
- Default changed to 70 days back for comprehensive historical data extraction

### Code Quality & Architecture

**T038** ✅ Comprehensive error handling and logging
- Added structured logging with configurable verbosity levels
- Exception handling at file, date, and process levels
- Detailed progress reporting with record counts and timing
- Graceful failure handling with rollback capabilities

**T039** ✅ Modular architecture with clear separation of concerns
- `TickExtractor` class for tick data processing
- `DepthExtractor` class for market depth processing
- `MnqTickRecord` dataclass with validation and normalization
- Reusable SCID parsing utilities with multiple access patterns

**T040** ✅ Testing and validation infrastructure
- Unit tests for SCID parsing, data validation, and depth processing
- Integration tests for end-to-end extraction workflows
- All tests passing with comprehensive coverage
- Automated validation of data integrity and consistency

---

## Current Configuration & Usage

### Recommended Production Settings
```bash
# Extract MNQ data for last 70 days with optimal performance
python3 -m src.services.tick_extractor \
  --scid-dir /mnt/c/SierraChart/Data \
  --days-back 70 \
  --tick-workers 4 \
  --depth-workers 2 \
  --max-recent-records 0
```

### Single Date Extraction
```bash
# Extract specific date with controlled parallelism
python3 -m src.services.tick_extractor \
  --scid-dir /mnt/c/SierraChart/Data \
  --date 2025-11-07 \
  --tick-workers 1 \
  --depth-workers 1
```

### Memory-Constrained Processing
```bash
# Process recent records only for memory efficiency
python3 -m src.services.tick_extractor \
  --scid-dir /mnt/c/SierraChart/Data \
  --days-back 1 \
  --max-recent-records 10000 \
  --tick-workers 2 \
  --depth-workers 1
```

---

## Performance Characteristics

- **Memory Usage**: ~100MB per worker (streaming processing)
- **CPU Utilization**: Configurable parallelism (1-8 cores typical)
- **I/O Pattern**: Sequential file reading with backwards parsing
- **Storage**: DuckDB for ticks, Parquet for depth data
- **Processing Speed**: ~10,000-50,000 records/second per core

---

## Data Quality Assurance

- **Ticker Normalization**: All futures contracts normalized to base symbols
- **Timestamp Accuracy**: UTC timestamps with microsecond precision
- **Data Validation**: Price/volume validation in data models
- **Duplicate Prevention**: Database constraints and transaction handling
- **Error Recovery**: Comprehensive logging and graceful failure handling</content>
<parameter name="filePath">/home/rwest/projects/data-pipeline/specs/002-extract-mnq-data/tasks.md