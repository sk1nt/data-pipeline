# UW Alerting Task Prompts

Use these prompts to guide sub-agents working on UW (Unusual Whales) alerting features.

---

## Task 1: Enhance Userscript Configuration

**Context:** The userscript has a hardcoded endpoint URL that needs to be configurable.

**Prompt:**
```
Working on: userscripts/uw-phoenix-capture.user.js

Enhance the userscript with configurable settings:
1. Add a CONFIG object at the top for:
   - PIPELINE_HOST (default: 192.168.168.151)
   - PIPELINE_PORT (default: 8877)
   - CHANNELS to subscribe (default: market_agg_socket, option_trades_super_algo, option_trades_super_algo:SPX)
   - DEBUG mode toggle
   - OVERLAY_ENABLED toggle

2. Add localStorage persistence for config
3. Add a simple settings UI accessible via Tampermonkey menu
4. Add reconnection logic with exponential backoff

Reference: userscripts/AGENTS.md for patterns
```

---

## Task 2: Add Option Flow Filtering

**Context:** Currently all option trades are forwarded. Add filtering for high-value alerts only.

**Prompt:**
```
Working on: userscripts/uw-phoenix-capture.user.js

Add client-side filtering before forwarding:
1. Add filter config for:
   - MIN_PREMIUM (default: 100000) - minimum trade premium
   - SYMBOLS_WHITELIST (default: empty = all)
   - TRADE_TYPES (default: ["SWEEP", "BLOCK"])
   - SENTIMENT_FILTER (default: null = all)

2. Apply filters in the message handler before forwardRawMsg()
3. Add filter stats to overlay (filtered/total ratio)
4. Add bypass for market_agg_socket (always forward)

Test with: SPX sweeps > $500k premium
```

---

## Task 3: Improve UW Message Service

**Context:** The backend service needs better message routing and storage.

**Prompt:**
```
Working on: src/services/uw_message_service.py

Enhance the UW message processing:
1. Add message deduplication using Redis SET with TTL
   - Key: uw:dedup:{message_hash}
   - TTL: 60 seconds
   
2. Add rate limiting per topic
   - Max 100 messages/minute per topic
   - Log dropped messages
   
3. Add message enrichment:
   - Lookup underlying price from Redis cache
   - Calculate Greeks if strike/expiry available
   
4. Add metrics:
   - Counter: uw_messages_processed_total{topic, status}
   - Histogram: uw_message_processing_seconds

Reference: src/services/AGENTS.md for service patterns
```

---

## Task 4: Build Alert Rule Engine

**Context:** Create a configurable rule engine for UW alerts (Issue #19).

**Prompt:**
```
Working on: src/services/rule_engine.py (create new file)

Build an alert rule engine:

1. Define AlertRule model in src/models/alert.py:
   ```python
   class AlertRule(BaseModel):
       id: str
       name: str
       enabled: bool = True
       conditions: List[RuleCondition]  # AND logic
       actions: List[RuleAction]
       cooldown_seconds: int = 60
   
   class RuleCondition(BaseModel):
       field: str  # e.g., "premium", "symbol", "sentiment"
       operator: str  # eq, gt, lt, in, contains
       value: Any
   
   class RuleAction(BaseModel):
       type: str  # "discord", "webhook", "redis_publish"
       config: Dict[str, Any]
   ```

2. Implement RuleEngine class:
   - load_rules() - from Redis or config file
   - evaluate(message: UWMessage) -> List[AlertRule]
   - execute_actions(rule: AlertRule, message: UWMessage)
   
3. Add default rules:
   - Large SPX sweeps (>$1M premium)
   - Unusual volume on watchlist symbols
   - Bearish sentiment spike

4. Integrate with uw_message_service.py

Reference: Issue #19 for full requirements
```

---

## Task 5: Discord Alert Integration

**Context:** Forward filtered UW alerts to Discord channels.

**Prompt:**
```
Working on: discord-bot/bot/commands/alerts.py (create new file)

Add UW alert commands and listener:

1. Commands:
   - !alerts subscribe <channel_id> - Subscribe channel to UW alerts
   - !alerts unsubscribe - Remove subscription
   - !alerts rules - List active alert rules
   - !alerts test - Send test alert

2. Redis Listener (in discord-bot/bot/services/uw_listener.py):
   - Subscribe to uw:option_trade:stream
   - Subscribe to uw:alerts:triggered
   - Format and send to subscribed Discord channels

3. Alert Embed Format:
   ```python
   embed = discord.Embed(
       title=f"🚨 {alert.symbol} {alert.sentiment}",
       color=0xFF0000 if bearish else 0x00FF00
   )
   embed.add_field(name="Premium", value=f"${alert.premium:,.0f}")
   embed.add_field(name="Strike", value=f"${alert.strike}")
   embed.add_field(name="Expiry", value=alert.expiry)
   ```

4. Rate limiting: Max 10 alerts/minute per channel

Reference: discord-bot/AGENTS.md for bot patterns
```

---

## Task 6: Market Aggregation Alerts

**Context:** Process market_agg_socket messages for regime detection.

**Prompt:**
```
Working on: src/services/market_agg_alert_service.py

Enhance market aggregation alerting:

1. Parse market_agg messages for:
   - Net premium (calls - puts)
   - Volume spikes vs 20-day average
   - Put/call ratio changes
   - Sector rotation signals

2. Implement regime detection:
   ```python
   class MarketRegime(Enum):
       RISK_ON = "risk_on"
       RISK_OFF = "risk_off"
       NEUTRAL = "neutral"
       TRANSITION = "transition"
   
   def detect_regime(agg_data: MarketAggData) -> MarketRegime:
       # Logic based on premium flow, VIX, sector rotation
   ```

3. Store regime history in Redis TimeSeries:
   - Key: ts:market:regime
   - Value: regime enum as int

4. Trigger alerts on regime transitions:
   - Publish to uw:market_regime:stream
   - Include transition direction and confidence

Reference: docs/UW_MESSAGE_HANDLING.md for message formats
```

---

## Execution Order

1. **Task 1** - Userscript config (improves development experience)
2. **Task 3** - Backend service improvements (foundation)
3. **Task 4** - Rule engine (core feature)
4. **Task 2** - Client-side filtering (reduces noise)
5. **Task 5** - Discord integration (user-facing)
6. **Task 6** - Market regime (advanced feature)

---

## Testing Checklist

- [ ] Userscript connects to UW WebSocket
- [ ] Messages forward to /uw endpoint
- [ ] Deduplication prevents duplicates
- [ ] Rule engine evaluates conditions correctly
- [ ] Discord receives formatted alerts
- [ ] Rate limiting prevents spam
- [ ] Metrics are exposed at /metrics
