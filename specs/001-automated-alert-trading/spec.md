# Feature Specification: Automated Alert Trading

**Feature Branch**: `001-automated-alert-trading`  
**Created**: 2025-12-07  
**Status**: Draft  
**Input**: User description: "Automated trading feature for alerts that parses a number of different messages from select people only and then issues an automated entry trade with TastyTrade and sets a limit exit order for 50% of the position at 100% profit."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Alert-driven automated entry & partial exit (Priority: P1)

As a trader authorized for automated alerts, I want the system to parse an incoming alert message from a permitted channel or user, place a limit entry order with TastyTrade for the relevant instrument (using the price from the alert or a computed mid-market price), and immediately create a limit exit order for 50% of the executed position sized at 100% profit (double the execution price), so that I can capture an intended profit leg automatically.

**Why this priority**: This is the core value of the feature — automated, low-friction execution of curated trade signals.

**Independent Test**: Use a test Discord or Slack environment. Send an alert from a test user and channel that is in the allowlist, e.g., "Alert: BTO UBER 78p 12/05 @ 0.75". Confirm the system:
- Parses the alert correctly,
- Computes quantity based on configured allocation and account balances,
- Places an entry limit order (or simulates in dry-run mode),
- Places an exit limit order for 50% of filled quantity at 100% profit when a fill is confirmed,
#### Price Discovery

- Start the order at the current mid-market price if lower than the alert price, otherwise start at the alert price. If the order does not fill within 20 seconds, the system should increment the price by one instrument tick (determined from instrument metadata) and retry. Repeat up to 3 incremental retries over a 90 second period. If the required increment to reach the market is less than or equal to one tick, the system may convert to a market order. This behavior applies to both equities and futures, using the instrument's tick size for increments.

- Records an audit entry and returns a channel confirmation containing the order IDs and entry/exit prices.

**Acceptance Scenarios**:
1. **Given** the bot is running and user is authorized, **When** the alert is received, **Then** the bot places a limit entry order at the alert price and acknowledges the order in-channel.
2. **Given** that the entry order is filled (fully or partially), **When** the fill is confirmed, **Then** the bot places a limit exit for 50% of the filled quantity at 100% profit and logs the action.

---

### User Story 2 - Authorization, controls & audit (Priority: P2)

As an operator or compliance officer, I want alerts to be processed only for approved users and channels, and for every automated action to be auditable, so that trading is controlled and traceable.

**Why this priority**: Prevents unauthorized execution which can lead to compliance or financial risk.

**Independent Test**: Verify alerts from disallowed users or channels are ignored. Verify that allowed users cause orders and audit entries to be created in the audit store with timestamp, user, channel, parsed alert, computed quantity, order IDs, and entry/exit price.

**Acceptance Scenarios**:
1. **Given** an unauthorized user or channel, **When** a message is sent, **Then** the bot does not attempt to trade and optionally returns a message that the user is unauthorized.
2. **Given** an authorized alert triggers a trade, **When** orders are placed, **Then** an audit record is persisted and includes the required fields and any error/warning details.

---

### User Story 3 - Observability & retry/failure handling (Priority: P3)

As an ops or dev engineer, I need clear logs and safe retry/fallback logic when entry or exit orders fail or when the broker API is unavailable, so that the system behaves predictably and the operators can investigate and resume operations.

**Why this priority**: Automation must be resilient and transparent to reduce manual triage.

**Independent Test**: Simulate TastyTrade API authentication failure, network timeouts, or rejection of the order (invalid symbol or insufficient funds) and verify that the system retries as configured, logs clear errors, and notifies operators when necessary.

**Acceptance Scenarios**:
1. **Given** a transient API failure, **When** we attempt to place an entry order, **Then** the system retries up to configured attempts and logs each attempt; if unresolved, it cancels and logs final failure.
2. **Given** a permanent error (e.g., authentication), **When** an alert triggers an order, **Then** the system returns a clear message to the alert's channel and logs a failure in the audit store.

---

### Edge Cases

- Partial fills: If the entry order is partially filled, the exit size must be computed as 50% of the actual filled quantity (rounded down to an integer contract size). If the filled quantity is 1 contract, the exit does not place a 0.5 contract; behavior should be documented and the system should offer a configurable minimum-exit size.
- Concurrent alerts for the same instrument: The system should detect duplicate or near-duplicate walk-ins and either queue or apply deduplication rules (e.g., a 10-second deduplication window) to avoid double execution.
 - Missing price information: If the alert provides no price and mid-market data is unavailable, the system must not place the order and must log the issue for an operator to review.
 - Alerts must be the first message (not a reply): The system must only trigger on the initial message in a thread or channel; replies to a prior message in a thread should not be treated as alert triggers.
- Order rejection for credit/invalid symbol: The system must record the rejection, notify the alert channel user, and not create an exit for an unfilled entry order.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST parse multiple alert message formats and extract necessary fields: action (BTO/STC), symbol, leg type (put/call), strike, expiry, quantity (optional), and price (optional).
- **FR-002**: Only a configurable allowlist of specific user IDs and channel IDs must be able to trigger automated trading; the system should reject/ignore others.
- **FR-003**: The system MUST compute a trade quantity using allocation rules (configurable allocation percentage), buying power checks, and a per-trade maximum.
 - **FR-003**: The system MUST compute a trade quantity using allocation rules (configurable allocation percentage), buying power checks, and a per-trade maximum. Allocation percentage is a percentage of the account's available buying power at session login (initial BP) rather than dynamically computed on each order.
- **FR-004**: The system MUST place a limit entry order using either alert-provided price or mid-market price if no price is given; when in dry-run mode, simulate order without placing.
 - **FR-004**: The system MUST place a limit entry order using either alert-provided price or mid-market price if no price is given; when in dry-run mode, simulate order without placing. The system MUST implement price discovery using instrument tick-size increments and a conversion to market order when the final required increment is within one tick.
- **FR-005**: When an entry order is filled, the system MUST place a limit exit that sells 50% of the filled quantity at 100% profit based on the actual execution price where possible.
- **FR-006**: The system MUST write an audit record for each automated operation containing the following fields: timestamp, user_id, channel_id, parsed_alert, computed_quantity, entry_order_id, entry_price, fills (if present), exit_order_id (if placed), and error details (if any).
- **FR-007**: The system MUST include configurable retry/backoff policy for transient errors and a cancellation policy for stale or unsuccessful retries.
 - **FR-007**: The system MUST include configurable retry/backoff policy for transient errors and a cancellation policy for stale or unsuccessful retries.
 - **FR-009**: The system MUST implement price discovery rules: start at mid/alert price, wait 20s, then up to 3 increments of 1 tick across a 90s window; convert to market when remaining increment ≤ 1 tick.
- **FR-008**: The system MUST support a dry-run mode that parses alerts, computes quantity, and returns intended orders without placing them.

### Non-Functional Requirements

- **NFR-001**: The system should process a valid alert and confirm placement or rejection within 10 seconds (network latency permitting).
- **NFR-002**: Audit logging must be reliable: audit records must not be lost and should be queued/persisted even with temporary redis unavailability, with reasonable retry policies.
- **NFR-003**: The authorization model must be easy to manage for operations: either user IDs or role-based mapping must be supported.

### Key Entities *(include if feature involves data)*

- **AlertMessage**: raw message content, user_id, channel_id, posted_at, message_id.
- **ParsedAlert**: fields extracted from AlertMessage: action, symbol, strike, option_type, expiry, quantity, price.
- **OrderRecord**: placed order metadata: order_id, account_id, legs, order_type, price, quantity, status, fills.
- **AuditRecord**: persisted operation details for compliance reporting and historical analysis.
- **Whitelist/Allowlist**: admin-managed list of user IDs and/or roles and channel IDs permitted to generate automated trades.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 95% of valid alerts from allowlisted users result in the system placing an entry order (or simulation in dry-run) within 10 seconds when broker APIs and market data are available.
- **SC-002**: For fully or partially filled entry orders, the exit order for 50% of filled quantity is placed within 5 seconds of fill confirmation in 95% of cases.
- **SC-003**: 100% of executed or attempted automated actions are recorded in the audit store with required fields present.
- **SC-004**: Unauthorized alerts are never executed — test coverage should confirm 0% false positives when allowlist is configured.

***Assumptions***

- Authorized users and channels will be configured and updated by operations; this feature will not include user management UI.
- Alerts are intended to be short-lived: normal operations assume market conditions at the time of the alert remain valid for the period needed to place the limit order.
- Existing TastyTrade SDK provides sufficient metadata for fills and order status to compute exit prices based on actual execution price.

***Limitations & Non-Goals***

- Multi-leg bracket orders beyond initial entry + 50% exit are out of scope for this feature; advanced order types are future work.
- Position scaling and conditional risk-limiting rules are out of scope for MVP.

***Key Risks & Mitigations***

- Risk: Unauthorized or misformatted alerts lead to unintended trades. Mitigation: strict allowlist, message normalization, and pre-execution simulation (dry-run) for operator verification.
- Risk: Broker API timing and partial fills. Mitigation: prefer fill price in exit computation and log fills for audit.
- Risk: Audit logs lost due to infrastructure outages. Mitigation: queue audit events and retry until persisted.

***Clarifications (resolved)***

- **Q1**: Allowed senders definition — Resolved to A: we will use a configured allowlist of specific Discord/Slack user IDs and channel IDs. Only messages from those user IDs and channels are processed.

- **Q2**: Exit sizing on partial fills — Resolved to A: exit sizing uses 50% of the actual filled quantity (rounded down to integer contracts), with a configurable minimum-exit size.

- **Q3**: Price source for entry orders if alert price is absent — Resolved to A: use mid-market price (bid+ask)/2 as the primary fallback for entry orders.

- **Q4**: Allocation percentage base — Resolved to A: allocate against the account's available buying power at session login (initial BP). This is a static reference point used for successive allocation calculations.

- **Q5**: Price discovery conversion rule — Resolved to A: Convert to market if the required increment to reach the market price is ≤ 1 tick based on the instrument's tick size (price discovery uses 1 tick increments, up to 3 retries over 90s; initial wait 20s).

All clarifications have been applied to the spec above.

***End of Spec***
