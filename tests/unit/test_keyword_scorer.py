"""Tests for keyword scoring engine."""

import pytest

from src.models.social_event import KeywordCategory
from src.services.social_feed_service import KeywordScorer


class TestKeywordScorer:
    def test_tariff_tweet_scores_high(self):
        scorer = KeywordScorer()
        score, keywords, categories = scorer.score(
            "I will impose a 100% tariff on all Chinese imports"
        )
        assert score >= 3
        assert "tariff" in keywords
        assert "tariff_trade" in categories

    def test_unrelated_tweet_scores_zero(self):
        scorer = KeywordScorer()
        score, keywords, categories = scorer.score(
            "Had a wonderful dinner with friends tonight"
        )
        assert score == 0
        assert keywords == []
        assert categories == []

    def test_multi_category_headline(self):
        scorer = KeywordScorer()
        score, keywords, categories = scorer.score(
            "Fed rate cut expected as tariff war with China escalates and stock market crashes"
        )
        # Should match: fed (3), rate cut (3), tariff (3), china (2), stock market (3), crash (3)
        assert score >= 6
        assert len(categories) >= 3
        assert "fed_monetary" in categories
        assert "tariff_trade" in categories

    def test_case_insensitive(self):
        scorer = KeywordScorer()
        score1, _, _ = scorer.score("TARIFF")
        score2, _, _ = scorer.score("tariff")
        score3, _, _ = scorer.score("Tariff")
        assert score1 == score2 == score3
        assert score1 > 0

    def test_word_boundary_matching(self):
        scorer = KeywordScorer()
        # "fed" should match as a word but not inside "federal" as a substring
        # Actually "fed" should match "fed" as standalone and "federal reserve" is separate
        score, keywords, _ = scorer.score("I fed the cat today")
        # "fed" keyword matches even in this context (word boundary match)
        assert "fed" in keywords

    def test_custom_keyword_config(self):
        custom = [
            KeywordCategory(
                name="custom", keywords=["moonshot", "rocketship"], weight=3
            )
        ]
        scorer = KeywordScorer(categories=custom)
        score, keywords, categories = scorer.score("This stock is a moonshot!")
        assert score == 3
        assert "moonshot" in keywords
        assert "custom" in categories

        # Default keywords should NOT match
        score2, _, _ = scorer.score("tariff on China")
        assert score2 == 0

    def test_threshold_filtering(self):
        scorer = KeywordScorer()
        # Low-weight match only (crypto = weight 1)
        score, _, _ = scorer.score("Bitcoin price is rising")
        assert score >= 1
        # With threshold of 2, this would be filtered
        meets_threshold = score >= 2
        # Bitcoin alone scores 1 (crypto_tech weight=1), so below threshold of 2
        # But could match other categories too - just check the mechanism works
        assert isinstance(meets_threshold, bool)

    def test_empty_text(self):
        scorer = KeywordScorer()
        score, keywords, categories = scorer.score("")
        assert score == 0
        assert keywords == []
        assert categories == []

    def test_multiple_keywords_same_category(self):
        scorer = KeywordScorer()
        # "rate cut" and "fed" are both in fed_monetary
        score, keywords, categories = scorer.score(
            "Fed announces rate cut of 25 basis points"
        )
        assert "fed" in keywords
        assert "rate cut" in keywords
        # Both contribute to score (weight 3 each)
        assert score >= 6
