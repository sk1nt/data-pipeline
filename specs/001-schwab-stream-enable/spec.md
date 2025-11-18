# Feature Specification: Enable Schwab streaming

**Feature Branch**: `001-schwab-stream-enable`  
**Created**: 2025-11-17  
**Status**: Draft  
**Input**: Enable streaming via Schwab for futures and equities, validate token exchange, test basic streaming, quote lookups, and level2 for MES/MNQ/NQ.

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Start a streaming session (Priority: P1)

Operations or developers want the system to start a stable Schwab streaming session for configured symbols so that real-time market data flows into the pipeline.

**Why this priority**: This enables the core streaming capability the rest of the product depends on.

**Independent Test**: Run `scripts/start_schwab_streamer.py` with `--ensure-env` and live tokens configured (or a test dummy), confirm the client logs in and subscribes to configured symbols.

**Acceptance Scenarios**:

1. **Given** valid `SCHWAB_CLIENT_ID`, `SCHWAB_CLIENT_SECRET`, and a persisted refresh token, **When** the streamer starts, **Then** it should connect to Schwab and subscribe to the configured symbols.
2. **Given** a token rotation in response to refresh logic, **When** tokens rotate, **Then** rotated tokens are persisted to `.tokens/schwab_token.json` and not overwrite `.env` content.

---

### User Story 2 - Validate token exchange and refresh (Priority: P1)

Ops/CI must be able to bootstrap the system headlessly using `schwab_token_manager.py` and validate the refresh process via `verify_token_refresh.py` so that token lifecycle works without interactive login.

**Why this priority**: Without reliable token exchange and rotation, streaming will fail in both CI and production.

**Independent Test**: Run `scripts/schwab_token_manager.py` to perform interactive bootstrap (or seed tokens into `.tokens`), then run `scripts/verify_token_refresh.py` to verify refresh and confirm `TokenStore` snapshots are updated.

**Acceptance Scenarios**:

1. **Given** no token persisted, **When** `schwab_token_manager.py` completes interactive flow, **Then** `.tokens/schwab_token.json` must exist and contain a valid refresh token.
2. **Given** a persisted token, **When** `verify_token_refresh.py` runs, **Then** token rotation occurs and new token is persisted and snapshots updated.

---

### User Story 3 - Validate market data flows for futures and equities (Priority: P2)

Developers and Ops need to validate both level-1 and level-2 data for configured futures and equity symbols, ensuring ticks and book updates are handled.

**Why this priority**: Ensures the consumer endpoints (Redis channels, downstream publishers) are able to route real-time messages for downstream consumers.

**Independent Test**: Use a dummy `schwab-py` streaming client or recordings to validate that tick and level2 payloads are parsed and published to `market_data:ticks` and `market_data:level2` respectively.

**Acceptance Scenarios**:

1. **Given** streamer is running, **When** a level-1 tick arrives for `/MNQ:XCME` or `/MES:XCME`, **Then** the subscriber receives a TickEvent containing symbol, last price, and timestamp.
2. **Given** streamer is running, **When** a level-2 update for `/NQ:XCME` arrives, **Then** a Level2Event with bids and asks is published and persisted.

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right edge cases.
-->

 - Connection failures from the Schwab streaming endpoint; the streamer must back off and reattempt according to a strategy and emit telemetry/logs.
 - Token rotations failing due to rate limits or invalid credentials; the system should pause streaming and signal error conditions for operator action.
 - Legacy token file formats present in `.tokens` directory; the streamer should gracefully rename or upgrade them rather than deleting tokens.

### Assumptions
- The stream client uses exchange-prefixed symbols in the test and configuration (e.g., `/MNQ:XCME`, `/MES:XCME`, `/NQ:XCME`) to match Schwab streaming naming conventions.
- CI will include a restricted set of live integration tests validating streaming behavior (startup, token refresh, level1 and level2) in addition to mocks for unit tests.
- Tokens persist in `.tokens` and are not written to `.env`; refresh persistence to `.env` is disabled by default. The system may write to `.env.back` for optional developer convenience, but `.env` is never overwritten by system tests or token rotation logic.

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

 - **FR-001**: System MUST provide a configurable streamer that supports subscribing to both futures and equity symbols configured in `settings.schwab_symbols` and `settings.tastytrade_symbol_list`.
 - **FR-002**: System MUST bootstrap token storage using `schwab_token_manager.py` for initial interactive token, and support headless refresh using `verify_token_refresh.py` or `TokenStore`'s APIs.
 - **FR-003**: System MUST persist tokens using the `TokenStore` metadata format and atomically write updates to avoid corrupting tokens during concurrent writes.
 - **FR-004**: The streamer MUST start background auto-refresh with configurable refresh intervals, persist rotated tokens, and, when enabled, append refresh token to `.env.back` (not overwrite `.env`).
 - **FR-005**: System MUST publish Level1 (tick) and Level2 (book) events to `market_data:ticks` and `market_data:level2`, respectively, using `TradingEventPublisher`.
 - **FR-006**: Tests and CI MUST NOT delete or overwrite `.env` containing developer credentials; tests should rely on `.env.back` or mocked environments instead.

*Example of marking unclear requirements:*

- **FR-006**: System MUST authenticate users via [NEEDS CLARIFICATION: auth method not specified - email/password, SSO, OAuth?]
- **FR-007**: System MUST retain user data for [NEEDS CLARIFICATION: retention period not specified]

- **TokenStore**: Maintains `.tokens/schwab_token.json` in metadata format. Key attributes: `token`, `creation_timestamp`.
- **SchwabAuthClient**: Wraps `schwab-py` client for OpenID/OAuth2 handling, provides `refresh_tokens()` and `start_auto_refresh()`.
- **SchwabStreamClient**: Runs streaming logic and publishes Tick/Level2 events via `TradingEventPublisher`.
- **TradingEventPublisher**: Publishes tick and level2 events to Redis channels.
- **StreamEvent**: TickEvent and Level2Event schema objects carrying symbol, price/size, and timestamp.

- **[Entity 1]**: [What it represents, key attributes without implementation]
- **[Entity 2]**: [What it represents, relationships to other entities]

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

 - **SC-001**: The streamer starts and subscribes to the configured symbols within 10 seconds of startup (or logs an actionable error), when valid tokens and network conditions are present.
 - **SC-002**: For each configured symbol, the streamer publishes at least one tick event per minute during normal market hours in live mode (or simulated streams in CI mode).
 - **SC-003**: Level2 updates for `/MES:XCME`, `/MNQ:XCME`, and `/NQ:XCME` are received and published to `market_data:level2` when available.
 - **SC-004**: The token refresh flow must persist rotated refresh tokens to `TokenStore` and not overwrite `.env`; if `SCHWAB_PERSIST_REFRESH_TO_ENV` is enabled, `.env.back` is updated safely.
