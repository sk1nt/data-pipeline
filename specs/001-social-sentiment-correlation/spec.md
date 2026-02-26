# Feature Specification: Social Sentiment & Market Correlation Alerts

## Problem Statement

Market-moving events from influential social media accounts (e.g., Trump on Truth Social/Twitter) and financial news outlets frequently precede or coincide with volume spikes, GEX shifts, and price dislocations. Currently, the pipeline has no way to correlate these external signals with its real-time market microstructure data (GEX, tick volume, UW option flow). Traders must manually monitor social feeds and mentally cross-reference with market data.

## Solution

A real-time correlation engine that:
1. **Ingests** social media posts and financial news headlines via RSS feeds (no paid APIs required for MVP)
2. **Scores** each event by financial relevance using configurable keyword matching
3. **Correlates** scored events against existing pipeline signals (GEX changes, volume spikes, price moves, unusual option flow)
4. **Alerts** when social events coincide with abnormal market behavior, delivered through the existing Discord bot infrastructure

## User Stories

### US-1: Social Feed Ingestion
**As a** trader, **I want** the system to automatically monitor social media posts from key financial figures **so that** I don't have to manually watch multiple feeds.

**Acceptance Criteria:**
- System polls RSS feeds for Truth Social, Twitter/X (via Nitter/RSSHub), and financial news
- Polls every 30s during RTH, every 5min off-hours
- Each post/headline is normalized into a standard event model with timestamp, source, author, text, and relevance score
- Events are published to Redis `social:events:stream` for downstream consumers
- Feed errors are logged without crashing the service

### US-2: Keyword Relevance Scoring
**As a** trader, **I want** social posts and news headlines scored by market relevance **so that** only financially meaningful events trigger correlation checks.

**Acceptance Criteria:**
- Configurable keyword categories with weighted scores (HIGH=3, MEDIUM=2, LOW=1)
- Default keyword sets cover: tariff/trade, fed/monetary, market-direct, geopolitical, fiscal, crypto/tech
- Posts with score < minimum threshold (default 2) are logged but don't trigger correlation
- Keyword lists are configurable via settings without code changes

### US-3: Volume + Social Correlation
**As a** trader, **I want** to be alerted when a volume spike happens within minutes of a social/news event **so that** I can react to news-driven momentum.

**Acceptance Criteria:**
- Alert fires when: social event (score >= 2) in last 5 min AND current 1-min volume > 2× rolling 20-bar average
- Alert includes: event summary, volume ratio, symbol
- No duplicate alerts for same social event within cooldown window

### US-4: GEX + Social Correlation
**As a** trader, **I want** to be alerted when GEX shifts significantly after a social event **so that** I can understand options positioning reactions.

**Acceptance Criteria:**
- Alert fires when: social event in last 5 min AND net GEX changed > 15%
- Alert includes: event summary, GEX before/after, percent change

### US-5: Price Move + Social Correlation
**As a** trader, **I want** to be alerted when price moves sharply after a social event **so that** I can identify directional momentum.

**Acceptance Criteria:**
- Alert fires when: social event in last 5 min AND price moved > 0.3% in 2 min
- Alert includes: event summary, price change %, time elapsed, symbol

### US-6: Multi-Signal Confluence
**As a** trader, **I want** a special alert when multiple signals fire simultaneously after a social event **so that** I can identify the highest-conviction setups.

**Acceptance Criteria:**
- Alert fires when social event + >= 2 of: volume spike, GEX shift, price move, UW flow anomaly
- Alert is visually distinct (higher urgency formatting)
- Alert lists all triggered signals

### US-7: Discord Alert Delivery
**As a** trader, **I want** correlation alerts delivered to my Discord channels **so that** I receive them alongside existing market alerts.

**Acceptance Criteria:**
- Alerts published to configurable Discord channels via existing bot infrastructure
- Alert formatting is consistent with existing market_agg alerts
- Alerts include clickable context (event source, timestamp, signal values)

### US-8: Historical Correlation Log
**As a** trader, **I want** a historical log of all correlation events **so that** I can review patterns and tune thresholds.

**Acceptance Criteria:**
- All correlation events (alert + no-alert) persisted to DuckDB
- Queryable by date range, source, signal type, score
- Schema includes: timestamp, social_event, signals_triggered, alert_fired, thresholds_at_time

## Non-Functional Requirements

- **Latency**: Social event to alert < 10s total (polling interval dominates)
- **Reliability**: Feed failure doesn't crash pipeline; graceful degradation with logging
- **Rate Limiting**: Respect RSS feed rate limits; configurable poll intervals
- **Deduplication**: Same social post doesn't generate multiple alerts
- **Configurability**: All thresholds, feed URLs, keywords, and enable/disable flags via env/config
- **Security**: No storage of API keys for social feeds in MVP (RSS only); sanitize all external text before rendering in alerts
