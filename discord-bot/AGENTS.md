# Discord Bot Agent Instructions

## Domain Scope
This folder contains the Discord bot for trading alerts and commands. Sub-agents working here should focus on:

### Core Components
- `bot/trade_bot.py` - Main bot class (3868 lines - needs refactoring)
- `bot/tastytrade_client.py` - TastyTrade order execution
- `bot/config.py` - Bot configuration
- `run_discord_bot.py` - Entrypoint

### Primary Responsibilities
1. **Command Handling** - Discord slash commands (!gex, !status, etc.)
2. **Alert Forwarding** - Redis pub/sub → Discord channels
3. **Trade Execution** - TastyTrade order placement via commands
4. **GEX Feed Display** - Real-time GEX updates in channels

## Refactoring Priority
The `trade_bot.py` file is too large. When working here:
- Extract commands to `bot/commands/` using discord.py Cogs
- Extract Redis listeners to `bot/services/`
- Keep `trade_bot.py` as thin orchestration layer

### Target Structure
```
discord-bot/bot/
├── commands/
│   ├── __init__.py
│   ├── gex.py          # !gex, !levels commands
│   ├── trading.py      # !buy, !sell, !position
│   ├── status.py       # !status, !health
│   └── alerts.py       # !subscribe, !unsubscribe
├── services/
│   ├── redis_listener.py
│   └── tastytrade_executor.py
├── trade_bot.py        # Reduced to wiring only
└── config.py
```

## Safety Constraints
- **NEVER** execute trades without confirmation workflow
- **NEVER** log or display full account credentials
- Position limits must be enforced before order submission
- Kill switch (`!killswitch`) must always be available

## Redis Channels Consumed
- `gex:snapshot:stream` - GEX updates
- `uw:option_trade:stream` - UW option alerts
- `uw:market_agg:stream` - Market aggregation alerts

## Testing
- Tests in `discord-bot/tests/`
- Mock Discord client for unit tests
- Mock TastyTrade client for order tests
- Never run trading tests against production accounts

## Environment Variables
Required in `.env`:
- `DISCORD_BOT_TOKEN`
- `DISCORD_STATUS_CHANNEL_ID`
- `TASTYTRADE_USERNAME`, `TASTYTRADE_PASSWORD`
- `REDIS_HOST`, `REDIS_PORT`
