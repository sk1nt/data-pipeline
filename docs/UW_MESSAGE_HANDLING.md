# UW (Unusual Whales) Message Handling

## Overview
The data pipeline now handles 3 types of Unusual Whales websocket messages:
1. **market_agg_socket** - Market aggregation state data
2. **option_trades_super_algo:SPX** - SPX index options
3. **option_trades_super_algo** - Stock options

## Data Flow

```
UW Websocket → /uw endpoint → UWMessageService → Redis → Discord Bot
                                     ↓
                              RedisFlushWorker → DuckDB
```

## Components

### 1. Models (`src/models/uw_message.py`)
Pydantic models for 3 message types:
- `MarketAggState` - Market aggregation data with premium, greeks
- `OptionTradeIndex` - Index options (SPX) trades
- `OptionTradeStock` - Stock options trades

### 2. Service (`src/services/uw_message_service.py`)
- **Processes** raw websocket messages (array format)
- **Stores** in Redis with 24-hour TTL
- **Routes** to Discord channels based on symbol
- **Publishes** to Redis pubsub for real-time subscribers

#### Redis Keys
- `uw:market_agg:latest` - Latest market aggregation state
- `uw:market_agg:history` - Historical market states (list, max 1000)
- `uw:option_trade:latest` - Latest option trade
- `uw:option_trade:history` - Historical option trades (list, max 1000)
- `uw:option_trade:symbol:{SYMBOL}` - Latest trade by symbol

#### Redis Pubsub Channels
- `uw:market_agg:stream` - Market aggregation broadcasts
- `uw:option_trade:stream` - Option trade broadcasts

### 3. API Endpoint (`data-pipeline.py`)
`POST /uw` accepts websocket array format:
```json
[null, null, "channel_name", "event_type", {"data": {...}}]
```

Channels:
- `market_agg_socket` → Market aggregation processing
- `option_trades_super_algo:SPX` → SPX index options
- `option_trades_super_algo` → Stock options

### 4. EOD Persistence (`src/services/redis_flush_worker.py`)
RedisFlushWorker now includes `_flush_uw_messages()`:
- Flushes Redis history lists to DuckDB at end of day
- Creates tables: `market_agg_state`, `option_trades`
- Clears Redis history after successful flush
- Database: `data/uw_messages.db`

#### DuckDB Schema

**market_agg_state:**
```sql
CREATE TABLE market_agg_state (
    received_at TIMESTAMP,
    date VARCHAR,
    call_premium DOUBLE,
    put_premium DOUBLE,
    call_premium_otm_only DOUBLE,
    put_premium_otm_only DOUBLE,
    delta DOUBLE,
    gamma DOUBLE,
    theta DOUBLE,
    vega DOUBLE
)
```

**option_trades:**
```sql
CREATE TABLE option_trades (
    received_at TIMESTAMP,
    topic VARCHAR,
    topic_symbol VARCHAR,
    is_index_option BOOLEAN,
    ticker VARCHAR,
    option_chain_id BIGINT,
    type VARCHAR,
    strike DOUBLE,
    expiry TIMESTAMP,
    dte INTEGER,
    cost_basis DOUBLE,
    volume BIGINT,
    price DOUBLE,
    tags VARCHAR  -- JSON array as string
)
```

### 5. Discord Notifications (`discord-bot/bot/trade_bot.py`)
- Subscribes to `uw:option_trade:stream` Redis pubsub
- Routes messages to specific channels:
  - **SPX trades** → Channel `1429940127899324487`
  - **Other options** → Channel `1425136266676146236`
- Formats alerts with `format_option_trade_alert()`

#### Message Format
```
UW option alert  2025-01-15T14:30:00Z
ticker          AAPL
types           sweep, whale
contract        AAPL250117C175
side/strike     BUY 175 CALL  dte 2
stock spot      170.50  bid-ask 0.50-0.55
option spot     1.25  size 500
premium         62500  volume 1200  oi 5000
chain bid/ask   1.20 / 1.30
legs            n/a
code            SWEEP
flags           unusual_volume
tags            tech, earnings
```

## Configuration

### Environment Variables (discord-bot/.env)
```bash
# Discord channel IDs for UW notifications
DISCORD_UW_CHANNEL_IDS=1425136266676146236,1429940127899324487

# Redis pubsub channel for option trades
UW_OPTION_STREAM_CHANNEL=uw:option_trade:stream
```

### Redis Keys Configuration
- TTL: 24 hours (`CACHE_TTL_SECONDS = 86400`)
- History limit: 1000 messages per type (`HISTORY_LIMIT = 1000`)

## Testing

### Manual Test - Send Mock Message
```python
import requests
import json

# Mock websocket message
mock_message = [
    None,
    None,
    "option_trades_super_algo",
    "trade",
    {
        "data": {
            "ticker": "AAPL",
            "is_index_option": False,
            "option_chain_id": 12345,
            "type": "CALL",
            "strike": 175.0,
            "expiry": "2025-01-17T00:00:00Z",
            "dte": 2,
            "cost_basis": 125.0,
            "volume": 500,
            "price": 1.25,
            "tags": ["sweep", "whale"]
        }
    }
]

response = requests.post(
    "http://localhost:8877/uw",
    json=mock_message
)
print(response.json())
```

### Check Redis
```python
import redis

r = redis.Redis(host='localhost', port=6379, db=0)

# Check latest option trade
latest = r.get("uw:option_trade:latest")
print(json.loads(latest))

# Check history count
count = r.llen("uw:option_trade:history")
print(f"History count: {count}")

# Check by symbol
aapl = r.get("uw:option_trade:symbol:AAPL")
print(json.loads(aapl))
```

### Check DuckDB (after EOD flush)
```python
import duckdb

conn = duckdb.connect("data/uw_messages.db")

# Check option trades
trades = conn.execute("SELECT * FROM option_trades ORDER BY received_at DESC LIMIT 10").fetchall()
for trade in trades:
    print(trade)

# Check market aggregation
agg = conn.execute("SELECT * FROM market_agg_state ORDER BY received_at DESC LIMIT 10").fetchall()
for row in agg:
    print(row)

conn.close()
```

## Monitoring

### RedisFlushWorker Status
Check flush worker status via API:
```bash
curl http://localhost:8877/status
```

Look for:
- `uw_market_agg`: Count of market agg messages flushed
- `uw_option_trades`: Count of option trades flushed

### Discord Bot Logs
Check Discord bot logs for:
- `Subscribed to UW option stream`
- `Broadcasting UW alert to channel {channel_id}`
- `Failed to send UW alert` (errors)

## Troubleshooting

### Messages not appearing in Discord
1. Check Redis pubsub: `redis-cli PUBSUB CHANNELS "uw:*"`
2. Verify Discord bot is subscribed to `uw:option_trade:stream`
3. Check channel IDs in `.env` match Discord server
4. Verify bot has send permissions in channels

### Messages not flushing to DuckDB
1. Check RedisFlushWorker is running: `GET /status`
2. Verify Redis history lists exist: `redis-cli LLEN uw:option_trade:history`
3. Check DuckDB file permissions: `ls -la data/uw_messages.db`
4. Review flush worker logs for exceptions

### Invalid message format
1. Verify websocket array format: `[null, null, channel, event, {data}]`
2. Check channel name matches: `market_agg_socket`, `option_trades_super_algo`, `option_trades_super_algo:SPX`
3. Validate Pydantic models in `src/models/uw_message.py`

## Future Enhancements
- [ ] Add rate limiting to prevent Discord spam
- [ ] Implement filtering by tags (e.g., only "whale" trades)
- [ ] Add market state change alerts (e.g., premium threshold)
- [ ] Create historical analysis endpoints
- [ ] Add DuckDB parquet export for long-term storage
