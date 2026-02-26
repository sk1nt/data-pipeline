"""Tests for social event Pydantic models."""

from datetime import datetime, timezone

import pytest

from src.models.social_event import (
    DEFAULT_KEYWORD_CATEGORIES,
    KeywordCategory,
    SocialEvent,
    SocialSource,
)


class TestSocialEvent:
    def test_valid_construction(self):
        event = SocialEvent(
            event_id="abc123",
            timestamp=datetime.now(timezone.utc),
            source=SocialSource.TWITTER,
            author="@testuser",
            text="The tariff on China will be 100%",
        )
        assert event.event_id == "abc123"
        assert event.source == SocialSource.TWITTER
        assert event.relevance_score == 0

    def test_missing_text_rejected(self):
        with pytest.raises(Exception):
            SocialEvent(
                event_id="abc123",
                timestamp=datetime.now(timezone.utc),
                source=SocialSource.TWITTER,
                author="@testuser",
                text="",
            )

    def test_missing_author_rejected(self):
        with pytest.raises(Exception):
            SocialEvent(
                event_id="abc123",
                timestamp=datetime.now(timezone.utc),
                source=SocialSource.TWITTER,
                author="",
                text="some text",
            )

    def test_text_truncated_at_2000(self):
        long_text = "x" * 3000
        event = SocialEvent(
            event_id="abc123",
            timestamp=datetime.now(timezone.utc),
            source=SocialSource.NEWS_RSS,
            author="Reuters",
            text=long_text,
        )
        assert len(event.text) == 2000

    def test_generate_event_id_deterministic(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        id1 = SocialEvent.generate_event_id(
            SocialSource.TWITTER, "@user", "hello", ts
        )
        id2 = SocialEvent.generate_event_id(
            SocialSource.TWITTER, "@user", "hello", ts
        )
        assert id1 == id2

    def test_generate_event_id_differs_by_source(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        id1 = SocialEvent.generate_event_id(
            SocialSource.TWITTER, "@user", "hello", ts
        )
        id2 = SocialEvent.generate_event_id(
            SocialSource.TRUTH_SOCIAL, "@user", "hello", ts
        )
        assert id1 != id2

    def test_from_rss_entry(self):
        ts = datetime(2026, 2, 1, tzinfo=timezone.utc)
        event = SocialEvent.from_rss_entry(
            source=SocialSource.NEWS_RSS,
            author="CNBC",
            title="Fed raises rates by 50 bps",
            link="https://cnbc.com/article",
            published=ts,
            score=3,
            keywords=["fed"],
            categories=["fed_monetary"],
        )
        assert event.source == SocialSource.NEWS_RSS
        assert event.relevance_score == 3
        assert "fed" in event.keywords_matched
        assert event.url == "https://cnbc.com/article"

    def test_serialization_roundtrip(self):
        event = SocialEvent(
            event_id="roundtrip1",
            timestamp=datetime.now(timezone.utc),
            source=SocialSource.TWITTER,
            author="@user",
            text="Test message",
            relevance_score=5,
            keywords_matched=["fed", "tariff"],
            categories_matched=["fed_monetary", "tariff_trade"],
        )
        json_str = event.model_dump_json()
        restored = SocialEvent.model_validate_json(json_str)
        assert restored.event_id == event.event_id
        assert restored.keywords_matched == event.keywords_matched


class TestKeywordCategory:
    def test_valid_category(self):
        cat = KeywordCategory(name="test", keywords=["a", "b"], weight=2)
        assert cat.weight == 2

    def test_weight_out_of_range(self):
        with pytest.raises(Exception):
            KeywordCategory(name="test", keywords=["a"], weight=5)

    def test_default_categories_exist(self):
        assert len(DEFAULT_KEYWORD_CATEGORIES) > 0
        names = [c.name for c in DEFAULT_KEYWORD_CATEGORIES]
        assert "tariff_trade" in names
        assert "fed_monetary" in names
        assert "market_direct" in names
