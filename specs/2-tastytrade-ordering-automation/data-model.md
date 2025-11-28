# Data Model

## Order
- **id**: string (unique identifier)
- **symbol**: string (underlying symbol)
- **quantity**: decimal (number of contracts/shares)
- **order_type**: enum (MARKET, LIMIT)
- **price**: decimal (limit price if applicable)
- **status**: enum (PENDING, FILLED, CANCELLED, PARTIAL)
- **environment**: enum (SANDBOX, PRODUCTION)
- **created_at**: datetime
- **updated_at**: datetime
- **channel_id**: string (Discord channel ID)
- **user_id**: string (Discord user ID)

## Trader
- **discord_id**: string (unique Discord user identifier)
- **permissions**: list[string] (allowed actions: FUTURES, OPTIONS, ALERTS)
- **account_id**: string (linked Tastytrade account)
- **allocation_percentage**: decimal (percentage of BP for orders)

## ChatMessage
- **id**: string (message ID)
- **channel_id**: string (Discord channel ID)
- **user_id**: string (sender Discord ID)
- **content**: string (raw message text)
- **timestamp**: datetime
- **command**: string (parsed command, e.g., !tt buy)
- **parsed_args**: dict (extracted arguments)

## TestResult
- **order_id**: string (reference to Order)
- **test_type**: enum (SANDBOX_VALIDATION, PRODUCTION_DRY_RUN)
- **result**: enum (PASS, FAIL)
- **details**: string (error messages or metrics)
- **timestamp**: datetime

## Account
- **tastytrade_account_id**: string (Tastytrade account number)
- **buying_power**: decimal (current BP)
- **allocation_percentage**: decimal (default allocation %)
- **environment**: enum (SANDBOX, PRODUCTION)
- **credentials**: dict (encrypted client_secret, refresh_token)

## Relationships
- Order belongs to Trader (user_id)
- Order belongs to Account (account_id)
- ChatMessage triggers Order (via parsing)
- TestResult validates Order
- Trader has many Orders