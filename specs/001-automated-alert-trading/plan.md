# Implementation Plan: Automated Alert Trading

**Branch**: `001-automated-alert-trading` | **Date**: 2025-12-12 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-automated-alert-trading/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Automated trading system that parses alert messages from authorized Discord users/channels, places limit entry orders via TastyTrade API with price discovery (start at alert/mid price, retry with 1-tick increments up to 3 times over 90s, convert to market if gap ≤1 tick), and automatically creates limit exit orders for 50% of filled quantity at 100% profit. All actions are audited and support dry-run mode.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: discord.py, redis, FastAPI, Pydantic, TastyTrade SDK, httpx  
**Storage**: Redis (audit queue, state), DuckDB (persistent audit log)  
**Testing**: pytest with pytest-asyncio for Discord bot tests  
**Target Platform**: Linux server (existing data-pipeline deployment)  
**Project Type**: Single project with Discord bot integration  
**Performance Goals**: Process alerts within 10s, place exit orders within 5s of fill  
**Constraints**: <10s alert-to-order latency, reliable audit logging with retry  
**Scale/Scope**: ~10-50 authorized users, ~100 alerts/day peak

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- ✅ **Code Quality**: Existing project uses Ruff linter/formatter; code reviews via PR workflow
- ✅ **Accuracy**: Pydantic models for validation, audit logging for all trades, dry-run mode for testing
- ✅ **Consistency**: Follows existing patterns in discord-bot/ and src/services/
- ✅ **Testing**: pytest infrastructure exists; will add contract/integration/unit tests per TDD
- ✅ **Performance**: Alert processing <10s target aligns with existing GEX feed (<1s) patterns

**Status**: ✅ PASS - No violations detected

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
discord-bot/
├── bot/
│   ├── trade_bot.py              # Existing - add alert listener
│   ├── config.py                 # Existing - add alert config
│   └── tastytrade_client.py      # Existing - use for orders
├── utils/
│   └── alert_parser.py           # NEW - parse alert formats
└── tests/
    ├── test_alert_parser.py      # NEW - parser tests
    ├── test_automated_options_service.py  # Existing - extend
    └── test_notifications.py     # Existing - extend

src/
├── models/
│   └── alert_message.py          # NEW - Pydantic models
├── services/
│   ├── automated_options_service.py  # NEW - core orchestration
│   ├── auth_service.py           # Existing - allowlist management
│   └── audit_service.py          # NEW - audit persistence
└── lib/
    └── redis_client.py           # Existing - use for queue

backend/src/
└── api/
    └── alerts_endpoint.py        # NEW - optional API for allowlist mgmt

tests/
├── contract/
│   └── test_alert_api.py         # NEW - API contract tests
├── integration/
│   └── test_alert_to_order_flow.py  # NEW - end-to-end
└── unit/
    └── test_alert_parsing.py     # NEW - unit tests
```

**Structure Decision**: Extends existing single-project structure with Discord bot integration. Alert parsing and service logic goes in `src/services/`, Discord bot listener in `discord-bot/bot/`, audit persistence uses existing Redis/DuckDB patterns.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
