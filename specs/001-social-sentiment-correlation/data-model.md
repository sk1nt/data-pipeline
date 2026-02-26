# Data Models: Social Sentiment & Market Correlation

## Social Event (Redis + DuckDB)

```python
class SocialSource(str, Enum):
    TRUTH_SOCIAL = "truth_social"
    TWITTER = "twitter"
    NEWS_RSS = "news_rss"

class SocialEvent(BaseModel):
    event_id: str                    # SHA256(source + author + text + timestamp)
    timestamp: datetime              # When post/headline was published
    received_at: datetime            # When our system ingested it
    source: SocialSource
    author: str                      # e.g., "@realDonaldTrump", "Reuters"
    text: str                        # Raw post/headline text (max 2000 chars)
    url: Optional[str]               # Link to original post
    relevance_score: int             # Computed keyword score (0+)
    keywords_matched: List[str]      # Which keywords matched
    categories_matched: List[str]    # Which categories matched
```

**Redis key**: `social:event:{event_id}` (TTL 24h)
**Redis pub/sub**: `social:events:stream`

## Keyword Category

```python
class KeywordCategory(BaseModel):
    name: str                        # e.g., "tariff_trade"
    keywords: List[str]              # e.g., ["tariff", "trade war", "sanctions"]
    weight: int                      # 1=LOW, 2=MEDIUM, 3=HIGH
```

## Market Signal Snapshot

```python
class MarketSignalSnapshot(BaseModel):
    timestamp: datetime
    symbol: str
    # Volume
    volume_1min: Optional[float]
    volume_20bar_avg: Optional[float]
    volume_ratio: Optional[float]       # current / avg
    # GEX
    net_gex: Optional[float]
    prev_net_gex: Optional[float]
    gex_change_pct: Optional[float]
    # Price
    price: Optional[float]
    price_2min_ago: Optional[float]
    price_change_pct: Optional[float]
    # UW Flow
    uw_put_call_ratio: Optional[float]
    uw_prev_ratio: Optional[float]
    uw_max_premium: Optional[float]     # Largest single trade premium in window
```

## Correlation Alert

```python
class CorrelationAlertType(str, Enum):
    VOLUME_SPIKE = "volume_spike"
    GEX_SHIFT = "gex_shift"
    PRICE_MOVE = "price_move"
    UW_FLOW = "uw_flow"
    CONFLUENCE = "confluence"

class CorrelationAlert(BaseModel):
    alert_id: str                    # UUID
    timestamp: datetime
    alert_type: CorrelationAlertType
    social_event: SocialEvent
    market_signals: MarketSignalSnapshot
    signals_triggered: List[str]     # ["volume_spike", "gex_shift"] for confluence
    message: str                     # Pre-formatted alert text
    severity: str                    # "high" for confluence, "medium" for single signal
```

**Redis pub/sub**: `correlation:alerts:stream`

## DuckDB: correlation_events Table

```sql
CREATE TABLE IF NOT EXISTS correlation_events (
    id              INTEGER PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    social_event_id VARCHAR NOT NULL,
    social_source   VARCHAR NOT NULL,
    social_author   VARCHAR NOT NULL,
    social_text     VARCHAR,
    social_score    INTEGER NOT NULL,
    social_url      VARCHAR,
    alert_type      VARCHAR,          -- NULL if no alert fired
    alert_fired     BOOLEAN NOT NULL DEFAULT FALSE,
    signals_triggered VARCHAR,        -- JSON array of signal names
    volume_ratio    DOUBLE,
    gex_change_pct  DOUBLE,
    price_change_pct DOUBLE,
    uw_ratio_change DOUBLE,
    config_snapshot VARCHAR,          -- JSON of thresholds at time of check
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_correlation_events_ts ON correlation_events(timestamp);
CREATE INDEX idx_correlation_events_source ON correlation_events(social_source);
CREATE INDEX idx_correlation_events_alert ON correlation_events(alert_fired);
```
