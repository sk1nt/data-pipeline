"""Service for polling social media / news RSS feeds and scoring events by financial relevance."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import feedparser
import httpx

from src.lib.redis_client import RedisClient
from src.models.social_event import (
    BEARISH_KEYWORDS,
    BULLISH_KEYWORDS,
    DEFAULT_KEYWORD_CATEGORIES,
    KeywordCategory,
    Sentiment,
    SocialEvent,
    SocialSource,
)

LOGGER = logging.getLogger(__name__)

SOCIAL_EVENTS_CHANNEL = "social:events:stream"
SOCIAL_ALL_EVENTS_CHANNEL = "social:events:all"
SOCIAL_HISTORY_KEY = "social:events:history"
SOCIAL_HISTORY_MAX = 200
SEEN_SET_KEY = "social:seen_ids"


class KeywordScorer:
    """Score text by matching against weighted keyword categories."""

    def __init__(self, categories: Optional[List[KeywordCategory]] = None) -> None:
        self.categories = categories or DEFAULT_KEYWORD_CATEGORIES
        # Pre-compile patterns for each keyword (word-boundary, case-insensitive)
        self._patterns: List[Tuple[KeywordCategory, str, re.Pattern]] = []
        for cat in self.categories:
            for kw in cat.keywords:
                pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
                self._patterns.append((cat, kw, pattern))
        # Sentiment patterns
        self._bullish = [
            (kw, re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE))
            for kw in BULLISH_KEYWORDS
        ]
        self._bearish = [
            (kw, re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE))
            for kw in BEARISH_KEYWORDS
        ]

    def score(self, text: str) -> Tuple[int, List[str], List[str], Sentiment]:
        """Score text and return (total_score, matched_keywords, matched_categories, sentiment)."""
        total = 0
        matched_keywords: List[str] = []
        matched_categories: Set[str] = set()

        for cat, kw, pattern in self._patterns:
            if pattern.search(text):
                total += cat.weight
                matched_keywords.append(kw)
                matched_categories.add(cat.name)

        # Sentiment direction
        bull = sum(1 for _, p in self._bullish if p.search(text))
        bear = sum(1 for _, p in self._bearish if p.search(text))
        if bull > bear:
            sentiment = Sentiment.BULLISH
        elif bear > bull:
            sentiment = Sentiment.BEARISH
        else:
            sentiment = Sentiment.NEUTRAL

        return total, matched_keywords, sorted(matched_categories), sentiment


class FeedConfig:
    """Configuration for a single RSS feed."""

    def __init__(
        self,
        url: str,
        source: SocialSource,
        author: str,
        *,
        priority: bool = False,
    ) -> None:
        self.url = url
        self.source = source
        self.author = author
        self.priority = priority


class SocialFeedService:
    """Async service that polls RSS feeds, scores entries, and publishes to Redis."""

    def __init__(
        self,
        redis_client: RedisClient,
        feeds: List[FeedConfig],
        *,
        scorer: Optional[KeywordScorer] = None,
        rth_interval_seconds: float = 30.0,
        off_hours_interval_seconds: float = 300.0,
        priority_interval_seconds: float = 10.0,
        min_score_threshold: int = 2,
        dedup_ttl_seconds: int = 86400,
    ) -> None:
        self.redis = redis_client
        self.feeds = feeds
        self.priority_feeds = [f for f in feeds if f.priority]
        self.normal_feeds = [f for f in feeds if not f.priority]
        self.scorer = scorer or KeywordScorer()
        self.rth_interval = rth_interval_seconds
        self.off_hours_interval = off_hours_interval_seconds
        self.priority_interval = priority_interval_seconds
        self.min_score = min_score_threshold
        self.dedup_ttl = dedup_ttl_seconds
        self._task: Optional[asyncio.Task] = None
        self._priority_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    def start(self) -> None:
        if self._task and not self._task.done():
            LOGGER.warning("SocialFeedService already running")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_normal(), name="social-feed-poller")
        if self.priority_feeds:
            self._priority_task = asyncio.create_task(
                self._run_priority(), name="social-feed-priority-poller"
            )
        LOGGER.info(
            "SocialFeedService started with %d feeds (%d priority), min_score=%d",
            len(self.feeds),
            len(self.priority_feeds),
            self.min_score,
        )

    async def stop(self) -> None:
        self._stop_event.set()
        for task in (self._task, self._priority_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        LOGGER.info("SocialFeedService stopped")

    async def _run_priority(self) -> None:
        """Fast loop for priority (breaking-news) feeds."""
        while not self._stop_event.is_set():
            try:
                await self._poll_feeds_concurrent(self.priority_feeds)
            except Exception:
                LOGGER.exception("Error in priority feed poll cycle")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.priority_interval
                )
                break
            except asyncio.TimeoutError:
                pass

    async def _run_normal(self) -> None:
        """Standard loop for non-priority feeds."""
        while not self._stop_event.is_set():
            interval = self._current_interval()
            try:
                await self._poll_feeds_concurrent(self.normal_feeds)
            except Exception:
                LOGGER.exception("Error in social feed poll cycle")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=interval
                )
                break
            except asyncio.TimeoutError:
                pass

    def _current_interval(self) -> float:
        try:
            from src.lib.market_hours import is_rth

            return self.rth_interval if is_rth() else self.off_hours_interval
        except ImportError:
            return self.rth_interval

    async def _poll_feeds_concurrent(self, feeds: List[FeedConfig]) -> None:
        if not feeds:
            return
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DataPipeline/1.0)"},
        ) as client:
            tasks = []
            for feed_cfg in feeds:
                if feed_cfg.source == SocialSource.TRUTH_SOCIAL:
                    tasks.append(self._poll_truth_social(client, feed_cfg))
                else:
                    tasks.append(self._poll_feed(client, feed_cfg))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for feed_cfg, result in zip(feeds, results):
                if isinstance(result, Exception):
                    LOGGER.exception(
                        "Error polling feed %s: %s", feed_cfg.url, result
                    )

    async def _poll_truth_social(
        self, client: httpx.AsyncClient, feed_cfg: FeedConfig
    ) -> None:
        """Poll Truth Social via the public Mastodon-compatible API."""
        username = feed_cfg.url  # stored as just the username
        # Step 1: look up account ID
        lookup = await client.get(
            f"https://truthsocial.com/api/v1/accounts/lookup?acct={username}"
        )
        lookup.raise_for_status()
        acct_id = lookup.json().get("id")
        if not acct_id:
            return

        # Step 2: fetch recent statuses
        statuses_resp = await client.get(
            f"https://truthsocial.com/api/v1/accounts/{acct_id}/statuses",
            params={"limit": 20, "exclude_replies": "true"},
        )
        statuses_resp.raise_for_status()
        statuses = statuses_resp.json()
        if not isinstance(statuses, list):
            return

        for status in statuses:
            # Strip HTML tags from content
            import re as _re
            raw = status.get("content", "")
            text = _re.sub(r"<[^>]+>", "", raw).strip()
            if not text:
                continue

            created = status.get("created_at", "")
            try:
                published = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except Exception:
                published = datetime.now(timezone.utc)

            url = status.get("url") or f"https://truthsocial.com/@{username}/{status.get('id', '')}"
            display_name = feed_cfg.author

            event_id = SocialEvent.generate_event_id(
                SocialSource.TRUTH_SOCIAL, display_name, text, published
            )
            if self._is_seen(event_id):
                continue

            score, keywords, categories, sentiment = self.scorer.score(text)

            event = SocialEvent(
                event_id=event_id,
                timestamp=published,
                source=SocialSource.TRUTH_SOCIAL,
                author=display_name,
                text=text,
                url=url,
                relevance_score=score,
                sentiment=sentiment,
                keywords_matched=keywords,
                categories_matched=categories,
            )

            self._mark_seen(event_id)
            self._publish_event_all(event)

            if score >= self.min_score:
                self._publish_event(event)
                LOGGER.info(
                    "Truth Social event: score=%d, author=%s, text=%s",
                    score, display_name, text[:80],
                )

    async def _poll_feed(
        self, client: httpx.AsyncClient, feed_cfg: FeedConfig
    ) -> None:
        max_retries = 3
        backoff = 5.0
        for attempt in range(max_retries + 1):
            resp = await client.get(feed_cfg.url)
            if resp.status_code == 429:
                if attempt < max_retries:
                    retry_after = float(resp.headers.get("Retry-After", backoff))
                    LOGGER.warning(
                        "Rate limited (429) on %s, retrying in %.0fs (attempt %d/%d)",
                        feed_cfg.url, retry_after, attempt + 1, max_retries,
                    )
                    await asyncio.sleep(retry_after)
                    backoff *= 2
                    continue
                LOGGER.warning(
                    "Rate limited (429) on %s, max retries exhausted — skipping this cycle",
                    feed_cfg.url,
                )
                return
            resp.raise_for_status()
            break
        parsed = feedparser.parse(resp.text)

        for entry in parsed.entries:
            title = entry.get("title", "").strip()
            if not title:
                continue

            link = entry.get("link")
            published = self._parse_published(entry)

            # Build event ID for dedup
            event_id = SocialEvent.generate_event_id(
                feed_cfg.source, feed_cfg.author, title, published
            )

            # Dedup check
            if self._is_seen(event_id):
                continue

            # Score
            score, keywords, categories, sentiment = self.scorer.score(title)

            event = SocialEvent(
                event_id=event_id,
                timestamp=published,
                source=feed_cfg.source,
                author=feed_cfg.author,
                text=title,
                url=link,
                relevance_score=score,
                sentiment=sentiment,
                keywords_matched=keywords,
                categories_matched=categories,
            )

            # Mark as seen
            self._mark_seen(event_id)

            # Always publish to the all-events channel for frontend display
            self._publish_event_all(event)

            # Only publish to the scored stream if score meets threshold
            if score >= self.min_score:
                self._publish_event(event)
                LOGGER.info(
                    "Social event published: score=%d, source=%s, author=%s, text=%s",
                    score,
                    feed_cfg.source.value,
                    feed_cfg.author,
                    title[:80],
                )
            else:
                LOGGER.debug(
                    "Social event below threshold (score=%d): %s",
                    score,
                    title[:80],
                )

    def _is_seen(self, event_id: str) -> bool:
        try:
            return bool(self.redis.client.sismember(SEEN_SET_KEY, event_id))
        except Exception:
            return False

    def _mark_seen(self, event_id: str) -> None:
        try:
            pipe = self.redis.client.pipeline()
            pipe.sadd(SEEN_SET_KEY, event_id)
            pipe.expire(SEEN_SET_KEY, self.dedup_ttl)
            pipe.execute()
        except Exception:
            LOGGER.exception("Failed to mark event %s as seen", event_id)

    def _publish_event(self, event: SocialEvent) -> None:
        try:
            payload = event.model_dump(mode="json")
            self.redis.client.publish(
                SOCIAL_EVENTS_CHANNEL, json.dumps(payload, default=str)
            )
        except Exception:
            LOGGER.exception("Failed to publish social event %s", event.event_id)

    def _publish_event_all(self, event: SocialEvent) -> None:
        """Publish to the all-events channel and persist to history ring buffer."""
        try:
            payload = event.model_dump(mode="json")
            payload_str = json.dumps(payload, default=str)
            pipe = self.redis.client.pipeline()
            pipe.publish(SOCIAL_ALL_EVENTS_CHANNEL, payload_str)
            pipe.lpush(SOCIAL_HISTORY_KEY, payload_str)
            pipe.ltrim(SOCIAL_HISTORY_KEY, 0, SOCIAL_HISTORY_MAX - 1)
            pipe.execute()
        except Exception:
            LOGGER.exception("Failed to publish social event (all) %s", event.event_id)

    @staticmethod
    def _parse_published(entry: Any) -> datetime:
        """Parse published date from RSS entry, fallback to now."""
        published_parsed = entry.get("published_parsed")
        if published_parsed:
            try:
                import calendar

                return datetime.fromtimestamp(calendar.timegm(published_parsed), tz=timezone.utc)
            except Exception:
                pass
        return datetime.now(timezone.utc)
