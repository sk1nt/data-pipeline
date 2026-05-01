"""Pydantic models for social media and financial news events."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class SocialSource(str, Enum):
    """Source of a social/news event."""

    TRUTH_SOCIAL = "truth_social"
    TWITTER = "twitter"
    NEWS_RSS = "news_rss"


class Sentiment(str, Enum):
    """Directional sentiment of a news headline."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class KeywordCategory(BaseModel):
    """A category of keywords with a weight for scoring."""

    name: str
    keywords: List[str]
    weight: int = Field(ge=1, le=3, description="1=LOW, 2=MEDIUM, 3=HIGH")


class SocialEvent(BaseModel):
    """Normalized social media post or news headline."""

    event_id: str = Field(description="SHA256(source + author + text + timestamp)")
    timestamp: datetime = Field(description="When the post/headline was published")
    received_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When our system ingested it",
    )
    source: SocialSource
    author: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    url: Optional[str] = None
    relevance_score: int = Field(default=0, ge=0)
    sentiment: Sentiment = Field(default=Sentiment.NEUTRAL, description="Directional market sentiment")
    keywords_matched: List[str] = Field(default_factory=list)
    categories_matched: List[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def truncate_text(cls, v: str) -> str:
        if len(v) > 2000:
            return v[:2000]
        return v

    @classmethod
    def generate_event_id(
        cls, source: SocialSource, author: str, text: str, timestamp: datetime
    ) -> str:
        raw = f"{source.value}:{author}:{text}:{timestamp.isoformat()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_rss_entry(
        cls,
        source: SocialSource,
        author: str,
        title: str,
        link: Optional[str],
        published: datetime,
        score: int = 0,
        keywords: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> SocialEvent:
        event_id = cls.generate_event_id(source, author, title, published)
        return cls(
            event_id=event_id,
            timestamp=published,
            source=source,
            author=author,
            text=title,
            url=link,
            relevance_score=score,
            keywords_matched=keywords or [],
            categories_matched=categories or [],
        )


# Default keyword categories
DEFAULT_KEYWORD_CATEGORIES: List[KeywordCategory] = [
    KeywordCategory(
        name="tariff_trade",
        keywords=[
            "tariff",
            "trade war",
            "trade deal",
            "sanctions",
            "import tax",
            "duties",
            "embargo",
            "trade agreement",
            "trade deficit",
            "trade surplus",
        ],
        weight=3,
    ),
    KeywordCategory(
        name="fed_monetary",
        keywords=[
            "fed",
            "federal reserve",
            "interest rate",
            "rate cut",
            "rate hike",
            "powell",
            "fomc",
            "inflation",
            "cpi",
            "monetary policy",
            "quantitative",
            "treasury yield",
        ],
        weight=3,
    ),
    KeywordCategory(
        name="market_direct",
        keywords=[
            "stock market",
            "dow",
            "nasdaq",
            "s&p",
            "crash",
            "rally",
            "bull market",
            "bear market",
            "all-time high",
            "correction",
            "sell-off",
            "selloff",
            "market crash",
        ],
        weight=3,
    ),
    KeywordCategory(
        name="energy_oil",
        keywords=[
            "oil",
            "crude",
            "WTI",
            "brent",
            "OPEC",
            "petroleum",
            "natural gas",
            "energy prices",
            "oil price",
            "oil supply",
            "oil embargo",
            "refinery",
            "pipeline",
            "energy crisis",
            "fuel",
            "gasoline",
        ],
        weight=3,
    ),
    KeywordCategory(
        name="geopolitical",
        keywords=[
            "china",
            "russia",
            "ukraine",
            "war",
            "conflict",
            "nato",
            "military",
            "attack",
            "missile",
            "nuclear",
            "middle east",
            "iran",
            "israel",
            "hamas",
            "hezbollah",
            "strait of hormuz",
            "north korea",
            "airstrike",
            "ceasefire",
            "escalation",
            "invasion",
            "troops",
            "sanctions",
            "retaliation",
        ],
        weight=3,
    ),
    KeywordCategory(
        name="fiscal",
        keywords=[
            "tax",
            "spending",
            "budget",
            "debt ceiling",
            "shutdown",
            "stimulus",
            "deficit",
            "infrastructure bill",
            "government funding",
        ],
        weight=2,
    ),
    KeywordCategory(
        name="crypto_tech",
        keywords=[
            "bitcoin",
            "crypto",
            "cryptocurrency",
            "ethereum",
            "regulation",
            "sec",
            "antitrust",
        ],
        weight=1,
    ),
]

# Sentiment keyword lists for directional scoring
BULLISH_KEYWORDS: List[str] = [
    "rally", "surge", "soar", "jump", "gain", "rise", "climb",
    "bull market", "all-time high", "record high", "breakout", "boom",
    "rate cut", "dovish", "easing", "stimulus", "trade deal",
    "beat expectations", "beats", "upgrade", "buy", "outperform",
    "strong earnings", "better than expected", "upbeat", "optimism",
    "recovery", "rebound", "green", "positive",
]

BEARISH_KEYWORDS: List[str] = [
    "crash", "plunge", "tumble", "drop", "fall", "decline", "sink",
    "bear market", "sell-off", "selloff", "correction", "slump",
    "rate hike", "hawkish", "tightening", "tariff", "trade war",
    "miss expectations", "misses", "downgrade", "sell", "underperform",
    "weak earnings", "worse than expected", "warning", "recession",
    "layoffs", "default", "crisis", "shutdown", "sanctions",
    "inflation", "red", "negative", "fear", "panic",
    "oil spike", "crude surge", "energy crisis", "escalation", "airstrike",
    "iran", "war", "conflict", "retaliation", "invasion",
]
