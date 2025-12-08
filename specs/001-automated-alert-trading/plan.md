# Implementation Plan: Automated Alert Trading (Discord → Tastytrade)

**Branch**: `001-automated-alert-trading` | **Date**: 2025-12-08 | **Spec**: `specs/001-automated-alert-trading/spec.md`
**Input**: Feature specification from `/specs/001-automated-alert-trading/spec.md`

**Note**: Generated via speckit and updated to reflect current work.

## Summary

Automate Discord alert-driven option entries with price discovery, immediate 50% profit-take exits, and full auditability. Parse allowlisted alerts, compute quantity using allocation + buying power, submit entry (or dry-run), retry with 1-tick bumps up to 3 times before converting to market, then submit 50% exit on fills. All actions are audited and protected by allowlist/admin key; retries and notifications handle broker errors.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: discord.py (discord-bot), tastytrade Python SDK, redis, httpx
**Storage**: Redis for hot audit store; DuckDB/Parquet for longer-term persistence
**Testing**: pytest for unit/integration tests; pytest-asyncio for async code
**Target Platform**: Linux server (Docker or local deployment)
**Project Type**: single service + bot  
**Performance Goals**: alert-to-ack ≤10s (NFR-001)  
**Constraints**: non-blocking alert handling; dry-run must avoid live orders; Redis unavailability must not crash flow  
**Scale/Scope**: small team bot + API; single broker integration (Tastytrade)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Code Quality: Repo already uses `ruff`; we add unit+integration tests for the feature.
- Accuracy: `AlertParser`, `AutomatedOptionsService`, and `tastytrade_client` enforce validation.
- Consistency: Uses existing patterns in `src/services` and `discord-bot`; no new frameworks.
- Testing: Unit tests for parser, price discovery, retries, and admin API; e2e harness added.
- Performance: Flow is non-blocking; price discovery bounded to ~90s window.

**Constitution Check - PASS**

## Tests Added (current branch)

- **Unit tests:**
  - discord-bot/tests/test_round_to_tick.py — tick rounding helper
  - discord-bot/tests/test_quantity_allocation.py — allocation sizing
  - discord-bot/tests/test_automated_options_service.py — placement flow & preflight auth
  - tests/unit/test_price_discovery.py — tick math & conversion-to-market
  - tests/unit/test_retries.py, tests/unit/test_tastytrade_client_retries.py — retry/backoff

- **Integration tests:**
  - src/tests/test_admin_api.py — `/admin/alerts/process`, `/admin/audit/recent`

- **E2E:**
  - tests/e2e/test_alert_e2e_flow.py — alert → entry → partial fill → exit (simulated broker)

Current gaps: create_entry_order polish (T008) and live-like buying-power regression coverage.
Default account: using `TASTYTRADE_ACCOUNT=5WT31673`; keep env/whitelist consistent across bot + services.

### Source Code (repository root)

```text
src/
├── services/
│   ├── automated_options_service.py   # entry/exit orchestration, BP checks, dry-run
│   ├── price_discovery.py             # tick math + retry cadence
│   ├── notifications.py               # operator notifications
│   ├── metrics.py                     # counters for price discovery, order attempts, audit writes
│   ├── audit.py                       # audit persistence
│   └── tastytrade_client.py           # broker client with retries/auth guard
├── config/settings.py                 # allowlist + broker env toggles
└── api/routes/admin.py                # admin alert processing + audit fetch

discord-bot/
├── bot/alert_parser.py                # alert parsing/validation
├── bot/trade_bot.py                   # Discord command wiring + allowlist enforcement
└── bot/tastytrade_client.py           # bot-side wrapper using service client

tests/
├── e2e/test_alert_e2e_flow.py
├── unit/ (price discovery, retries, tastytrade client)
└── fixtures/discord_alerts.py

specs/001-automated-alert-trading/    # spec/plan/tasks/quickstart/contracts
```

Project layout reuses existing bot + services patterns and centralizes orchestration in `automated_options_service.py`.

## Complexity Tracking

No violations noted.
