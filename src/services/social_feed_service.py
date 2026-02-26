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
    DEFAULT_KEYWORD_CATEGORIES,
    KeywordCategory,
    SocialEvent,
    SocialSource,
)

LOGGER = logging.getLogger(__name__)

SOCIAL_EVENTS_CHANNEL = "social:events:stream"
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

    def score(self, text: str) -> Tuple[int, List[str], List[str]]:
        """Score text and return (total_score, matched_keywords, matched_categories)."""
        total = 0
        matched_keywords: List[str] = []
        matched_categories: Set[str] = set()

        for cat, kw, pattern in self._patterns:
            if pattern.search(text):
                total += cat.weight
                matched_keywords.append(kw)
                matched_categories.add(cat.name)

        return total, matched_keywords, sorted(matched_categories)


class FeedConfig:
    """Configuration for a single RSS feed."""

    def __init__(
        self,
        url: str,
        source: SocialSource,
        author: str,
    ) -> None:
        self.url = url
        self.source = source
        self.author = author


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
        min_score_threshold: int = 2,
        dedup_ttl_seconds: int = 86400,
    ) -> None:
        self.redis = redis_client
        self.feeds = feeds
        self.scorer = scorer or KeywordScorer()
        self.rth_interval = rth_interval_seconds
        self.off_hours_interval = off_hours_interval_seconds
        self.min_score = min_score_threshold
        self.dedup_ttl = dedup_ttl_seconds
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    def start(self) -> None:
        if self._task and not self._task.done():
            LOGGER.warning("SocialFeedService already running")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="social-feed-poller")
        LOGGER.info(
            "SocialFeedService started with %d feeds, min_score=%d",
            len(self.feeds),
            self.min_score,
        )

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._stop_event.set()
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        LOGGER.info("SocialFeedService stopped")

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            interval = self._current_interval()
            try:
                await self._poll_all_feeds()
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

    async def _poll_all_feeds(self) -> None:
        async with httpx.AsyncClient(timeout=15.0) as client:
            for feed_cfg in self.feeds:
                try:
                    await self._poll_feed(client, feed_cfg)
                except Exception:
                    LOGGER.exception("Error polling feed %s", feed_cfg.url)

    async def _poll_feed(
        self, client: httpx.AsyncClient, feed_cfg: FeedConfig
    ) -> None:
        resp = await client.get(feed_cfg.url)
        resp.raise_for_status()
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
            score, keywords, categories = self.scorer.score(title)

            event = SocialEvent(
                event_id=event_id,
                timestamp=published,
                source=feed_cfg.source,
                author=feed_cfg.author,
                text=title,
                url=link,
                relevance_score=score,
                keywords_matched=keywords,
                categories_matched=categories,
            )

            # Mark as seen
            self._mark_seen(event_id)

            # Only publish to stream if score meets threshold
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

    @staticmethod
    def _parse_published(entry: Any) -> datetime:
        """Parse published date from RSS entry, fallback to now."""
        published_parsed = entry.get("published_parsed")
        if published_parsed:
            try:
                from time import mktime

                return datetime.fromtimestamp(mktime(published_parsed), tz=timezone.utc)
            except Exception:
                pass
        return datetime.now(timezone.utc)
