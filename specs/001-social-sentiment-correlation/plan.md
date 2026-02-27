# Implementation Plan: Social Sentiment & Market Correlation Alerts

**Branch**: `001-social-sentiment-correlation` | **Date**: 2026-02-26 | **Spec**: [spec.md](./spec.md)

## Summary

Real-time correlation engine that ingests social media posts (Truth Social / Twitter from Trump, key financial figures), financial news headlines, and cross-references them with existing pipeline data (GEX shifts, volume spikes, price changes) to generate actionable alerts. Alerts fire when a social/news event coincides with abnormal market microstructure — delivered via the existing Discord bot and Redis pub/sub infrastructure.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: httpx, feedparser, pydantic, redis, polars
**Storage**: Redis (real-time state/cache), DuckDB (historical correlation log)
**Testing**: pytest with pytest-asyncio
**Target Platform**: Linux server (existing data-pipeline deployment)
**Project Type**: New service modules integrated into existing pipeline
**Performance Goals**: Social/news events processed < 5s from publish, correlation checks < 1s
**Constraints**: Rate-limit-aware polling (respect API/RSS limits), no paid API keys required for MVP

## Architecture Overview

```
┌──────────────────────────────────────────────────┐
│              Social / News Ingest Layer           │
│                                                   │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────┐ │
│  │ Truth Social │ │ Twitter/X    │ │ Financial │ │
│  │ RSS Poller   │ │ RSS/Nitter   │ │ News RSS  │ │
│  └──────┬──────┘ └──────┬───────┘ └─────┬─────┘ │
│         └───────────┬───┘               │        │
│                     ▼                   ▼        │
│            ┌────────────────────────┐            │
│            │   Social Feed Service  │            │
│            │  (normalize + score)   │            │
│            └──────────┬─────────────┘            │
└───────────────────────┼──────────────────────────┘
                        │ Redis pub/sub
                        │ social:events:stream
                        ▼
┌──────────────────────────────────────────────────┐
│            Correlation Engine                     │
│                                                   │
│  Inputs (all via Redis):                          │
│  • social:events:stream           (social/news)    │
│  • gex:snapshot:stream            (GEX snapshots)  │
│  • market_data:tastytrade:trades  (MNQ/MES ticks)  │
│                                                   │
│  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Event Window  │  │ Correlation Rules Engine │  │
│  │ (rolling 5m)  │→ │ volume spike + social    │  │
│  │               │  │ gex shift   + social     │  │
│  │               │  │ price move  + social     │  │
│  └──────────────┘  └──────────┬───────────────┘  │
│                               │                   │
└───────────────────────────────┼───────────────────┘
                                │ Redis pub/sub
                                │ correlation:alerts:stream
                                ▼
┌──────────────────────────────────────────────────┐
│  Alert Delivery (existing infrastructure)         │
│  • Discord bot channels                           │
│  • Redis alert history                            │
│  • DuckDB correlation log                         │
└──────────────────────────────────────────────────┘
```

## Existing Infrastructure Leveraged

| Component | What We Reuse | Location |
|-----------|--------------|----------|
| GEX snapshots | Real-time GEX data via Redis pub/sub | `src/services/gexbot_poller.py`, channel `gex:snapshot:stream` |
| Tick/price data | TastyTrade streamer trades (MNQ/MES only) | `src/services/tastytrade_streamer.py`, channel `market_data:tastytrade:trades` |
| Discord bot | Alert delivery to channels | `discord-bot/bot/trade_bot.py` |
| Redis time series | Historical metric storage | `src/services/redis_timeseries.py` |
| Market hours | RTH/off-hours awareness | `src/lib/market_hours.py` |
| Config system | Pydantic settings with env vars | `src/config.py` |
| Alert service pattern | Market agg alert architecture | `src/services/market_agg_alert_service.py` |

## Data Sources (MVP)

### Social Media — RSS-Based (No API Keys Required)

| Source | Method | Targets | Polling Interval |
|--------|--------|---------|------------------|
| Truth Social | RSS via `truthsocial.com/@realDonaldTrump/feed` or Nitter-like proxies | @realDonaldTrump | 30s during RTH |
| Twitter/X | RSS via Nitter instances or `rsshub.app` | @realDonaldTrump, @elikinosian, @DeItaone (Walter Bloomberg), @zaborhedge | 30s during RTH |
| Financial News RSS | feedparser on major feeds | Reuters Business, Bloomberg (public), CNBC, MarketWatch | 60s |

### Keyword Scoring (Configurable)

Posts/headlines are scored by financial relevance using keyword matching:

| Category | Keywords | Score Weight |
|----------|----------|--------------|
| Tariff/Trade | tariff, trade war, trade deal, sanctions, import tax, duties, embargo | HIGH (3) |
| Fed/Monetary | fed, interest rate, rate cut, rate hike, powell, fomc, inflation, cpi | HIGH (3) |
| Market Direct | stock market, dow, nasdaq, s&p, crash, rally, bull, bear | HIGH (3) |
| Geopolitical | china, russia, war, conflict, nato, military, attack | MEDIUM (2) |
| Fiscal | tax, spending, budget, debt ceiling, shutdown, stimulus | MEDIUM (2) |
| Crypto/Tech | bitcoin, crypto, AI, tech, regulation | LOW (1) |

## Correlation Rules (MVP)

### Rule 1: Social Event + Volume Spike
- Social event with score >= 2 occurred in last 5 minutes
- Volume in current 1-min bar > 2× rolling 20-bar average (MNQ/MES via TastyTrade)
- **Alert**: "🚨 VOLUME SPIKE after social event: {summary} | Vol: {current} vs avg {avg} | {symbol}"

### Rule 2: Social Event + GEX Shift
- Social event with score >= 2 occurred in last 5 minutes
- Net GEX changed > 15% from previous snapshot
- **Alert**: "🧲 GEX SHIFT after social event: {summary} | GEX: {prev} → {current} ({pct_change}%) | {symbol}"

### Rule 3: Social Event + Price Move
- Social event with score >= 2 occurred in last 5 minutes
- Price moved > 0.3% in last 2 minutes on MNQ/MES (configurable)
- **Alert**: "📊 PRICE MOVE after social event: {summary} | {symbol}: {price_change}% in {minutes}m"

### Rule 4: Multi-Signal Confluence
- Social event + any 2 of: volume spike, GEX shift, price move
- **Alert**: "⚡ CONFLUENCE ALERT: {n} signals triggered after: {summary} | Signals: {list}"

## Project Structure

### New Files

```text
src/
├── models/
│   └── social_event.py                # Pydantic models for social/news events
├── services/
│   ├── social_feed_service.py         # RSS polling + normalization + scoring
│   ├── correlation_engine.py          # Real-time correlation detection
│   └── correlation_alert_service.py   # Alert formatting + delivery via Redis pub/sub
└── config.py                          # Add social/correlation config fields

discord-bot/
└── bot/
    └── trade_bot.py                   # Add correlation alert listener (pattern: _listen_market_agg_alerts)

tests/
├── unit/
│   ├── test_social_feed_service.py
│   ├── test_correlation_engine.py
│   └── test_social_event_model.py
└── integration/
    └── test_correlation_pipeline.py
```

### Modified Files

| File | Change |
|------|--------|
| `src/config.py` | Add social feed URLs, polling intervals, correlation thresholds, enable flags |
| `discord-bot/bot/trade_bot.py` | Add `_listen_correlation_alerts()` subscriber + `_broadcast_correlation_alert()` |
| `data-pipeline.py` | Register social feed poller + correlation engine startup |

## Phases

### Phase 1: Social Feed Ingest (Foundation)
- Pydantic models for social/news events
- RSS poller service with keyword scoring
- Redis pub/sub publishing on `social:events:stream`
- Unit tests for parsing + scoring

### Phase 2: Correlation Engine
- Rolling event window (5-min buffer of social events + market signals)
- Subscribe to all existing Redis streams (GEX, ticks, UW)
- Implement correlation rules 1-5
- Publish to `correlation:alerts:stream`
- Unit tests for each rule

### Phase 3: Alert Delivery + Discord Integration
- Alert formatting service
- Discord bot subscriber for `correlation:alerts:stream`
- DuckDB persistence for correlation event log
- Integration tests

### Phase 4: Configuration + Observability
- Config settings in `src/config.py` (enable/disable, thresholds, feed URLs)
- Startup registration in `data-pipeline.py`
- Logging + metrics for feed health and alert rates
- Cooldown/dedup logic (no duplicate alerts within 5-min window for same event)
