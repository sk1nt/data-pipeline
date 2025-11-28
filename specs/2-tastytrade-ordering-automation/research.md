# Research Findings

## Tastytrade API Authentication Flow and Token Management

**Decision**: Use OAuth2 with refresh tokens stored securely in environment variables.

**Rationale**: OAuth2 provides secure access with short-lived session tokens and long-lived refresh tokens. Environment variables prevent credential exposure.

**Alternatives Considered**:
- API keys: Less secure, no token refresh mechanism.
- Username/password: Deprecated, less secure than OAuth2.

**Details**:
- OAuth2 flow with client credentials.
- Refresh tokens don't expire, store securely.
- Session tokens auto-refresh via SDK.
- Sandbox uses different endpoints, controlled by `is_test` parameter.

## Tastytrade API Endpoints for Options and Futures Orders

**Decision**: Use SDK's `place_order` method with instrument-specific legs.

**Rationale**: SDK abstracts REST endpoints, handles authentication and serialization.

**Alternatives Considered**:
- Direct REST API calls: More complex, error-prone.
- Third-party wrappers: Less maintained than official SDK.

**Details**:
- Options: Use `get_option_chain()`, `Option.get()`, `build_leg()`.
- Futures: Use `Future.get()`, `build_leg()`.
- Orders: `NewOrder` with `OrderType.LIMIT/MARKET`, `TimeInForce.DAY`.
- Response includes buying power effects and fees.

## Discord.py Patterns for Channel-Specific Message Handling

**Decision**: Use `on_message` event with channel ID filtering and command extension.

**Rationale**: Efficient filtering, built-in command parsing, prevents recursion.

**Alternatives Considered**:
- Manual string parsing: Error-prone, less maintainable.
- Command extension only: Doesn't handle non-command messages.

**Details**:
- Filter by `message.channel.id` in allowed list.
- Use `@commands.command` for `!tt` subcommands.
- Handle errors with `@command.error`.
- Intents required for message content.

## Implementation of Order Fill Logic with Price Increments

**Decision**: Async retry mechanism with price adjustments and final DAY order fallback.

**Rationale**: Balances fill probability with risk control, handles market conditions.

**Alternatives Considered**:
- Immediate market orders: Higher slippage risk.
- Fixed price limits: May not fill in volatile markets.

**Details**:
- Start at mid-price from market data.
- Retry up to 3 times with 2-3 tick increments.
- Cancel and resubmit on no-fill.
- Leave as DAY order if all retries fail.

## Sandbox to Production Switching Mechanisms

**Decision**: Environment variables with feature flags and gradual rollout.

**Rationale**: Safe transitions, easy rollback, comprehensive testing.

**Alternatives Considered**:
- Code branches: Harder to maintain.
- Manual configuration: Error-prone.

**Details**:
- `TASTYTRADE_USE_SANDBOX` flag.
- Separate credentials for each environment.
- Dry-run mode for testing.
- Pre-deployment checks and monitoring.