# Tasks: Enable Schwab Streaming

## Phase 1: Setup
 - [X] T001 Create project structure per implementation plan
 - [X] T002 Install required dependencies in requirements.txt
 - [X] T003 Create initial config files in src/config.py and .env.back

## Phase 2: Foundational
 - [X] T004 Implement TokenStore for atomic token persistence in src/services/token_store.py
 - [X] T005 Implement SchwabAuthClient for OpenID/OAuth2 in src/services/schwab_auth_client.py
 - [X] T006 Implement TradingEventPublisher for Redis in src/services/trading_event_publisher.py

## Phase 3: [US1] Start a streaming session (Priority: P1)
- [X] T007 [P] [US1] Implement SchwabStreamClient for streaming in src/services/schwab_streamer.py
- [X] T008 [P] [US1] Add symbol configuration to src/config.py and settings.py
- [X] T009 [P] [US1] Implement startup logic in scripts/start_schwab_streamer.py
- [X] T010 [P] [US1] Write integration test for streaming startup in tests/integration/test_schwab_stream_startup.py

## Phase 4: [US2] Validate token exchange and refresh (Priority: P1)
- [X] T011 [P] [US2] Implement interactive token bootstrap in scripts/schwab_token_manager.py
- [X] T012 [P] [US2] Implement headless token refresh in scripts/verify_token_refresh.py
- [X] T013 [P] [US2] Write integration test for token refresh in tests/integration/test_schwab_token_refresh.py

## Phase 5: [US3] Validate market data flows (Priority: P2)
- [X] T014 [P] [US3] Implement TickEvent and Level2Event models in src/models/stream_event.py
- [X] T015 [P] [US3] Parse and publish tick events in src/services/schwab_streamer.py
- [X] T016 [P] [US3] Parse and publish level2 events in src/services/schwab_streamer.py
- [X] T017 [P] [US3] Write integration test for tick/level2 events in tests/integration/test_schwab_market_data.py

## Final Phase: Polish & Cross-Cutting Concerns
- [ ] T018 Add logging and telemetry for error/backoff in src/services/schwab_streamer.py
- [ ] T019 Validate .env is never overwritten in all scripts and tests
- [ ] T020 Add documentation and quickstart in specs/001-schwab-stream-enable/quickstart.md

## Dependencies
- Phase 1 and 2 must be completed before any user story phases
- US1 and US2 are independent and can be developed/tested in parallel
- US3 depends on US1 (streaming must be working)

## Parallel Execution Examples
- T007, T008, T009, T010 (US1) can run in parallel
- T011, T012, T013 (US2) can run in parallel
- T014, T015, T016, T017 (US3) can run in parallel after US1

## Implementation Strategy
- MVP: Complete all US1 tasks (streaming session)
- Incremental delivery: US2 (token management), then US3 (market data validation)
- Polish phase for logging, safety, and docs

## Discord GEX Feed Notes
- Trading window: 09:35-16:00 America/New_York. The Discord bot posts GEX short-format feed updates to every channel id listed in `DISCORD_GEX_FEED_CHANNEL_IDS`, refreshes/editing every second, and deletes/reposts on wall-clock `GEX_FEED_REFRESH_MINUTES` (default 5) marks. The final 4 PM message is left intact.
- Formatting: uses the short formatter without timestamps plus rolling-window deltas (configurable via `GEX_FEED_WINDOW_SECONDS`, default 60). Spot shows net/percent change, zero gamma + call/put walls show current vs previous values with green/red coloring, net gex/scaled gamma/current max change add scaled `Δ` values.
- Env vars: toggle with `GEX_FEED_ENABLED=true`, choose symbol via `GEX_FEED_SYMBOL`, control cadence with `GEX_FEED_UPDATE_SECONDS`, and emit Redis telemetry hashes at `metrics:gex_feed` when `GEX_FEED_METRICS_ENABLED=true` (override key via `GEX_FEED_METRICS_KEY`).
- Testing override: set `GEX_FEED_FORCE_WINDOW=true` to bypass the 09:35-16:00 ET trading window so you can replay historical snapshots or Redis simulations after-hours without the bot idling.
