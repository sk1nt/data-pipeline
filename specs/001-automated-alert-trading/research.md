# Research: Automated Alert Trading

## Decisions
- **Language**: Python 3.11 (consistent with repo, runtime already python3.11 targets).
- **Primary dependencies**: discord.py for bot, tastytrade Python SDK for broker API, redis for audit queue, polars/duckdb for historical data if needed, httpx for any REST fallback.
- **Storage**: Redis for hot audit queue and DuckDB/Parquet for longer-term persistence; Redis used for audit push to avoid blocking the flow.
- **Observability**: Use built-in logger and Redis audit logs; notify operators via a DM/Discord channel when critical errors occur.
- **Auth model**: Allowlist of user IDs and channel IDs; `ensure_authorized()` used before live order actions to avoid invalid refresh token writes.

## Rationale
- Reusing existing infra (discord-bot + Tastytrade client + Redis) reduces implementation cost and leverages the repo's existing authentication patterns.
- Redis provides low-latency audit event persistence and is acceptable given current infra and retry semantics.
- Add preflight `ensure_authorized()` to prevent live orders when refresh token is invalid.

## Alternatives considered
- Use a database-backed queue (e.g., DuckDB/Parquet) for immediate audit persistency — rejected because Redis is already in the stack and is faster for hot writes.
- Add a role-based allowlist vs per-user allowlist — repo currently supports per-user/channel and operations prefer explicit allowlist for safety.

## Implementation details to be resolved
- Add minimal admin endpoints to fetch recent audit events from Redis or DuckDB, with authorization.
- Provide `!tt auth` commands (already exist) for status/refresh; extend to show token expiry.
- Add tests for partial fills and stale network handling under `tests/integration`.

## Summary
This research validates the approach chosen: implement automated alert parsing and order placement paths using existing `AutomatedOptionsService` and TastyTrade client wrappers, use Redis audits, preflight auth checks and dry-run support for safe development and operations.
