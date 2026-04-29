# Tasks: Social Sentiment & Market Correlation Alerts

**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

---

## Phase 1: Social Feed Ingest [Foundation]

### 1.1 — Social Event Pydantic Models [P]
**Story**: US-1, US-2
**Files**: `src/models/social_event.py`
- [ ] `SocialEvent` model: timestamp, source (enum: truth_social, twitter, news_rss), author, text, url, relevance_score, keywords_matched, raw_payload
- [ ] `SocialSource` enum: TRUTH_SOCIAL, TWITTER, NEWS_RSS
- [ ] `KeywordCategory` model: name, keywords list, weight (1-3)
- [ ] `ScoredEvent` model: extends SocialEvent with score breakdown
- [ ] Validation: timestamp required, text max 2000 chars, score >= 0

**Tests**: `tests/unit/test_social_event_model.py`
- [ ] Valid event construction
- [ ] Invalid event rejection (missing fields, bad score)
- [ ] Serialization round-trip

### 1.2 — Keyword Scoring Engine [P]
**Story**: US-2
**Files**: `src/services/social_feed_service.py` (scoring portion)
- [ ] `KeywordScorer` class with configurable keyword categories
- [ ] Default keyword sets: tariff/trade, fed/monetary, market-direct, geopolitical, fiscal, crypto/tech
- [ ] Case-insensitive matching, word-boundary aware
- [ ] Returns: total score, list of matched keywords, matched categories
- [ ] Minimum threshold filtering (default: score >= 2)

**Tests**: `tests/unit/test_keyword_scorer.py`
- [ ] Score a tariff tweet → HIGH score
- [ ] Score an unrelated tweet → score 0
- [ ] Score a multi-category headline → combined score
- [ ] Threshold filtering works
- [ ] Custom keyword config overrides defaults

### 1.3 — RSS Feed Poller Service
**Story**: US-1
**Files**: `src/services/social_feed_service.py`
**Depends on**: 1.1, 1.2
- [ ] `SocialFeedService` class: async poller using `httpx` + `feedparser`
- [ ] Feed registry: list of (url, source_type, author) tuples from config
- [ ] Deduplication via Redis set of seen post IDs (TTL 24h)
- [ ] RTH-aware polling interval (30s RTH, 5min off-hours) using `src/lib/market_hours.py`
- [ ] Normalize RSS entries → `SocialEvent` model
- [ ] Score each event via KeywordScorer
- [ ] Publish scored events (score >= threshold) to Redis `social:events:stream`
- [ ] Graceful error handling: log feed errors, continue polling other feeds
- [ ] `start()` / `stop()` lifecycle matching `GEXBotPoller` pattern

**Tests**: `tests/unit/test_social_feed_service.py`
- [ ] Mock RSS response → correct SocialEvent output
- [ ] Dedup: same post ID not published twice
- [ ] Feed error → logged, other feeds still polled
- [ ] RTH vs off-hours interval selection

### 1.4 — Config Settings for Social Feeds [P]
**Story**: US-1
**Files**: `src/config.py`
- [ ] `social_feed_enabled: bool` (default: False)
- [ ] `social_feed_urls: str` (comma-separated RSS URLs)
- [ ] `social_feed_rth_interval_seconds: float` (default: 30.0)
- [ ] `social_feed_off_hours_interval_seconds: float` (default: 300.0)
- [ ] `social_min_score_threshold: int` (default: 2)
- [ ] `social_dedup_ttl_seconds: int` (default: 86400)

---

## Phase 2: Correlation Engine

### 2.1 — Rolling Event Window
**Story**: US-3, US-4, US-5, US-6
**Files**: `src/services/correlation_engine.py`
**Depends on**: Phase 1
- [ ] `EventWindow` class: thread-safe rolling buffer (configurable duration, default 5 min)
- [ ] Stores recent social events and market signal snapshots
- [ ] Auto-evicts expired entries
- [ ] Query methods: `get_recent_social_events(within_seconds)`, `get_latest_market_state()`

**Tests**: `tests/unit/test_event_window.py`
- [ ] Events added and retrieved within window
- [ ] Expired events auto-evicted
- [ ] Thread-safety under concurrent access

### 2.2 — Market Signal Aggregator
**Story**: US-3, US-4, US-5
**Files**: `src/services/correlation_engine.py`
**Depends on**: 2.1
- [ ] Subscribe to Redis channels: `gex:snapshot:stream`, `market_data:ticks`, `uw:market_agg:stream`, `uw:option_trade:stream`
- [ ] Compute rolling metrics:
  - 1-min volume bar from tick data
  - 20-bar volume average
  - Last 2 GEX snapshots for delta calculation
  - Price at 2-min intervals for move detection
  - Latest UW put/call ratio
- [ ] Store computed signals in EventWindow

**Tests**: `tests/unit/test_market_signal_aggregator.py`
- [ ] Volume bar aggregation from mock ticks
- [ ] GEX delta computation
- [ ] Price move detection

### 2.3 — Correlation Rules Implementation
**Story**: US-3, US-4, US-5, US-6
**Files**: `src/services/correlation_engine.py`
**Depends on**: 2.1, 2.2
- [ ] `CorrelationEngine` class with pluggable rules
- [ ] Rule 1: Volume spike + social (volume > 2× avg within 5 min of social event)
- [ ] Rule 2: GEX shift + social (net GEX change > 15% within 5 min)
- [ ] Rule 3: Price move + social (> 0.3% in 2 min within 5 min)
- [ ] Rule 4: UW flow spike + social (premium > $1M or P/C ratio shift > 15%)
- [ ] Rule 5: Multi-signal confluence (social + >= 2 other signals)
- [ ] Cooldown/dedup: no repeat alert for same social event + rule combo within 5 min
- [ ] Each rule check returns `Optional[CorrelationAlert]`

**Tests**: `tests/unit/test_correlation_rules.py`
- [ ] Each rule fires on matching conditions
- [ ] Each rule silent on non-matching conditions
- [ ] Confluence rule requires >= 2 sub-signals
- [ ] Cooldown prevents duplicate alerts
- [ ] Edge cases: exactly at threshold, just below threshold

### 2.4 — Correlation Config Settings [P]
**Story**: US-3, US-4, US-5
**Files**: `src/config.py`
- [ ] `correlation_enabled: bool` (default: False)
- [ ] `correlation_window_seconds: int` (default: 300)
- [ ] `correlation_volume_spike_multiplier: float` (default: 2.0)
- [ ] `correlation_gex_shift_pct: float` (default: 15.0)
- [ ] `correlation_price_move_pct: float` (default: 0.3)
- [ ] `correlation_price_move_window_seconds: int` (default: 120)
- [ ] `correlation_cooldown_seconds: int` (default: 300)
- [ ] `correlation_uw_premium_threshold: int` (default: 1000000)

---

## Phase 3: Alert Delivery + Discord

### 3.1 — Correlation Alert Formatting Service
**Story**: US-7
**Files**: `src/services/correlation_alert_service.py`
**Depends on**: 2.3
- [ ] `CorrelationAlertService` class (pattern: `MarketAggAlertService`)
- [ ] Format methods for each alert type (volume, GEX, price, flow, confluence)
- [ ] Consistent emoji + markdown formatting matching existing alerts
- [ ] Include: event summary (truncated), signal values, symbol, timestamp
- [ ] Sanitize social text (strip URLs, limit length, escape markdown)
- [ ] Publish formatted alerts to Redis `correlation:alerts:stream`

**Tests**: `tests/unit/test_correlation_alert_service.py`
- [ ] Each alert type produces valid Discord-safe markdown
- [ ] Long social text truncated
- [ ] Special characters escaped

### 3.2 — Discord Bot Correlation Listener
**Story**: US-7
**Files**: `discord-bot/bot/trade_bot.py`
**Depends on**: 3.1
- [ ] Add `_listen_correlation_alerts()` method (pattern: `_listen_market_agg_alerts`)
- [ ] Subscribe to Redis `correlation:alerts:stream`
- [ ] Route to configurable Discord channel IDs
- [ ] Add to bot startup alongside existing listeners
- [ ] Add correlation alert channel config to bot settings

**Tests**: `tests/integration/test_correlation_discord.py`
- [ ] Mock Redis message → Discord channel send called
- [ ] Invalid message format → logged, not crashed

### 3.3 — DuckDB Correlation Event Log
**Story**: US-8
**Files**: `src/services/correlation_alert_service.py`
**Depends on**: 2.3
- [ ] Schema: `correlation_events` table (timestamp, social_event_id, social_text, social_source, social_score, signals_triggered JSON, alert_type, alert_fired bool, config_snapshot JSON)
- [ ] Log ALL correlation checks (both alert and no-alert) for later analysis
- [ ] Retention: configurable, default keep all

**Tests**: `tests/integration/test_correlation_persistence.py`
- [ ] Event written to DuckDB on alert
- [ ] Event written on no-alert (score below threshold)
- [ ] Query by date range returns expected rows

---

## Phase 4: Integration + Observability

### 4.1 — Pipeline Startup Registration
**Story**: All
**Files**: `data-pipeline.py`
**Depends on**: Phases 1-3
- [ ] Register `SocialFeedService` startup (if `social_feed_enabled`)
- [ ] Register `CorrelationEngine` startup (if `correlation_enabled`)
- [ ] Graceful shutdown of both services
- [ ] Log startup status for both services

### 4.2 — Integration Testing
**Story**: All
**Files**: `tests/integration/test_correlation_pipeline.py`
**Depends on**: 4.1
- [ ] End-to-end: fake RSS feed → social event → correlation check → alert published
- [ ] End-to-end: social event + mock volume spike → Discord alert sent
- [ ] Service start/stop lifecycle
- [ ] Config toggle: disabled services don't start

### 4.3 — Data Model for Historical Queries
**Story**: US-8
**Files**: `src/models/social_event.py`, `src/services/correlation_alert_service.py`
- [ ] `CorrelationEvent` model for DuckDB reads
- [ ] Query helper: get correlation events by date range, source, signal type
- [ ] Optional: basic CLI command to dump recent correlation events

---

## Task Dependency Graph

```
Phase 1 (Foundation):
  1.1 [P] ──┐
  1.2 [P] ──┤
  1.4 [P] ──┤
             └──→ 1.3 (RSS Poller)

Phase 2 (Correlation):
  1.3 ──→ 2.1 (Event Window) ─┐
           2.4 [P] ───────────┤
                               ├──→ 2.2 (Signal Aggregator) ──→ 2.3 (Rules)
                               │
Phase 3 (Alerts):
  2.3 ──→ 3.1 (Formatting) ──→ 3.2 (Discord Listener)
  2.3 ──→ 3.3 (DuckDB Log)

Phase 4 (Integration):
  3.2 + 3.3 ──→ 4.1 (Startup) ──→ 4.2 (Integration Tests) ──→ 4.3 (Queries)
```

`[P]` = Parallelizable with other `[P]` tasks in same phase.
