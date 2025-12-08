# Tasks: Automated Alert Trading

Phase 1 — Setup

- [X] T001 Add required environment flags to `.env.example`: `TASTYTRADE_DRY_RUN`, `TASTYTRADE_USE_SANDBOX`, `ADMIN_API_KEY`; update `.gitignore` to exclude `.env*` and backups. (files: .env.example, .gitignore)
- [X] T002 [P] Create test fixtures for Discord alerts and broker mocks: `tests/fixtures/discord_alerts.py`, `discord-bot/tests/conftest.py`, `backend/tests/conftest.py`. (files: tests/fixtures/discord_alerts.py, discord-bot/tests/conftest.py, backend/tests/conftest.py)

Phase 2 — Foundational

- [X] T003 Implement `ParsedAlert` dataclass and validation in `src/models/alert.py`. (file: src/models/alert.py)
- [X] T004 Implement `AuditRecord` dataclass and persistence helper in `src/models/audit.py` and `src/services/audit.py`. (files: src/models/audit.py, src/services/audit.py)
- [X] T005 Implement allowlist configuration and loader in `src/config.py` with `ALLOWED_USERS` and `ALLOWED_CHANNELS`. (file: src/config.py)

Phase 3 — User Stories

US1 — Alert-driven automated entry & partial exit (P1)

- [X] T006 [US1] Implement `AlertParser` in `discord-bot/bot/alert_parser.py` to parse actionable alerts and validate fields. (file: discord-bot/bot/alert_parser.py)
- [X] T007 [US1] [P] Add unit tests for `AlertParser`: `discord-bot/tests/test_alert_parser.py`. (file: discord-bot/tests/test_alert_parser.py)
- [ ] T008 [US1] Implement `AutomatedOptionsService.create_entry_order()` in `src/services/automated_options_service.py` with dry-run support. (file: src/services/automated_options_service.py)
- [X] T009 [US1] Implement price discovery as `src/services/price_discovery.py` using tick increments and conversion-to-market when ≤ 1 tick. (file: src/services/price_discovery.py)
- [X] T010 [US1] Add unit tests for price discovery and `round_to_tick`: `discord-bot/tests/test_round_to_tick.py` and `tests/unit/test_price_discovery.py`. (files: discord-bot/tests/test_round_to_tick.py, tests/unit/test_price_discovery.py)
 - [X] T011 [US1] Implement exit order creation (50% of filled quantity at 100% profit) in `src/services/automated_options_service.py`. (file: src/services/automated_options_service.py)
- [ ] T012 [US1] Add unit & integration tests for entry->fill->exit flow: `tests/integration/test_alert_to_exit_flow.py`. (file: tests/integration/test_alert_to_exit_flow.py)
 - [X] T013 [US1] Ensure audit events are created for all operations and persisted via `src/services/audit.py`. (files: src/services/audit.py, discord-bot/tests/test_audit.py)

US2 — Authorization, controls & audit (P2)

- [X] T014 [US2] Add allowlist enforcement in `discord-bot/bot/trade_bot.py` and admin API. (file: discord-bot/bot/trade_bot.py)
- [X] T015 [US2] Add tests for allowlist enforcement: `discord-bot/tests/test_allowlist.py`. (file: discord-bot/tests/test_allowlist.py)
- [X] T016 [US2] Implement admin endpoints `POST /admin/alerts/process` and `GET /admin/audit/recent` in `src/api/routes/admin.py` with admin key guard. (file: src/api/routes/admin.py)
- [X] T017 [US2] Add integration tests for admin endpoints: `src/tests/test_admin_api.py`. (file: src/tests/test_admin_api.py)

US3 — Observability & retry/failure handling (P3)

- [ ] T018 [US3] Implement retry/backoff policy in `src/lib/retries.py` and integrate into `src/services/tastytrade_client.py`. (files: src/lib/retries.py, src/services/tastytrade_client.py)
- [X] T018 [US3] Implement retry/backoff policy in `src/lib/retries.py` and integrate into `src/services/tastytrade_client.py`. (files: src/lib/retries.py, src/services/tastytrade_client.py)
- [X] T019 [US3] Implement `TastytradeAuthError` handling and `ensure_authorized()` use in write flows; update `!tt auth` behavior. (files: discord-bot/bot/tastytrade_client.py, discord-bot/bot/trade_bot.py)
- [X] T020 [US3] Implement operator notification flows for critical failures and add tests for notifications. (files: src/services/notifications.py, discord-bot/tests/test_notifications.py)

Phase 4 — Polish & Cross-Cutting Concerns

- [X] T021 [P] Add E2E test for full alert -> entry -> partial fill -> exit behavior using a simulated broker. (file: tests/e2e/test_alert_e2e_flow.py)
- [ ] T022 [P] Update `specs/001-automated-alert-trading/quickstart.md` with setup and run instructions. (file: specs/001-automated-alert-trading/quickstart.md)
- [ ] T023 [P] Add monitoring metrics for price discovery retries, order attempts, and audit writes. (file: src/services/metrics.py)

- [X] T023 [P] Add monitoring metrics for price discovery retries, order attempts, and audit writes. (file: src/services/metrics.py)
- [X] T024 [US2] Add ops task to sanitize repo history for sensitive env backups and rotate exposed secrets; add `.env.back` to `.gitignore`. (files: .gitignore, docs/SECURITY.md)

## Dependencies & Execution Order

- Phase 1 must finish before Phase 2 or 3.
- Phase 3 (US1) depends on Phase 2 tasks.
- US2 requires `AuditRecord` model & config but can parallelize with other foundational tasks.

## Parallel Execution Examples

- T003, T004, T005 (data model/config tasks) can run in parallel.
- T007, T010 (parsing & price discovery unit tests) can be worked on concurrently.

## Implementation Strategy & MVP

- MVP: Deliver US1 parsing + entry order + price discovery + dry-run + audit logging.
- Next: Add admin API (US2), observability & retry (US3), and polish.

## Summary

- Total tasks: 24
- Tasks per story: US1: 8, US2: 4, US3: 3, Setup/Foundational/Polish: 9

End of Tasks
