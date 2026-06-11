"""Integration tests for the social-event → correlation → alert pipeline.

These tests exercise the full flow without hitting external services:
  1. A SocialEvent is scored by KeywordScorer.
  2. The CorrelationEngine detects a market-signal coincidence.
  3. The CorrelationAlertService formats and persists the alert.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest

from src.models.social_event import SocialEvent, SocialSource
from src.services.social_feed_service import KeywordScorer
from src.services.correlation_engine import (
    CorrelationAlert,
    EventWindow,
    MarketSignalSnapshot,
    _check_volume_spike,
    _check_gex_shift,
    _check_price_move,
)
from src.services.correlation_alert_service import CorrelationAlertService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def alert_service(tmp_path):
    return CorrelationAlertService(db_path=str(tmp_path / "integration_test.db"))


@pytest.fixture
def scorer():
    return KeywordScorer()


def _make_social_event(text: str, source: SocialSource = SocialSource.TWITTER, author: str = "@realDonaldTrump") -> SocialEvent:
    scorer = KeywordScorer()
    score, keywords, categories, sentiment = scorer.score(text)
    return SocialEvent(
        event_id=uuid.uuid4().hex[:12],
        timestamp=datetime.now(timezone.utc),
        source=source,
        author=author,
        text=text,
        url="https://example.com/post",
        relevance_score=score,
        keywords_matched=keywords,
        categories_matched=categories,
        sentiment=sentiment,
    )


def _make_signal(
    *,
    volume_ratio: float | None = None,
    volume_1min: float | None = None,
    volume_20bar_avg: float | None = None,
    gex_change_pct: float | None = None,
    net_gex: float | None = None,
    prev_net_gex: float | None = None,
    price_change_pct: float | None = None,
    price: float | None = None,
    price_2min_ago: float | None = None,
) -> MarketSignalSnapshot:
    return MarketSignalSnapshot(
        timestamp=datetime.now(timezone.utc),
        symbol="MNQ",
        volume_ratio=volume_ratio,
        volume_1min=volume_1min,
        volume_20bar_avg=volume_20bar_avg,
        gex_change_pct=gex_change_pct,
        net_gex=net_gex,
        prev_net_gex=prev_net_gex,
        price_change_pct=price_change_pct,
        price=price,
        price_2min_ago=price_2min_ago,
    )


# ---------------------------------------------------------------------------
# Tests: full flow from score → rule → format → persist
# ---------------------------------------------------------------------------

class TestEndToEndVolumePipeline:
    """Tariff tweet + volume spike → alert formatted & persisted."""

    def test_tariff_tweet_volume_spike(self, alert_service):
        # 1. Score social event
        event = _make_social_event("Tariffs on China will be increased to 100% tomorrow!")
        assert event.relevance_score > 0, "Tariff text should trigger keyword scoring"

        # 2. Correlate with a volume spike
        signal = _make_signal(volume_ratio=3.5, volume_1min=7000, volume_20bar_avg=2000)
        msg = _check_volume_spike(event, signal, multiplier=2.0)
        assert msg is not None, "Volume spike should be detected"

        # 3. Build alert
        alert = CorrelationAlert(
            alert_type="volume_spike",
            social_event=event,
            market_signals=signal,
            signals_triggered=["volume_spike"],
            message=msg,
            severity="medium",
        )
        payload = alert.model_dump(mode="json")

        # 4. Format for Discord
        formatted = alert_service.format_alert_message(payload)
        assert "CORRELATION ALERT" in formatted
        assert "volume_spike" in formatted

        # 5. Persist to DuckDB
        alert_service.log_correlation_event(payload, alert_fired=True)
        events = alert_service.query_events(limit=10)
        assert len(events) == 1
        assert events[0]["alert_fired"] is True
        assert events[0]["volume_ratio"] == 3.5


class TestEndToEndConfluencePipeline:
    """Multiple signals fire simultaneously → confluence alert."""

    def test_confluence_volume_and_gex(self, alert_service):
        event = _make_social_event("The Fed just announced emergency rate cuts effective immediately")

        signal = _make_signal(
            volume_ratio=4.0, volume_1min=8000, volume_20bar_avg=2000,
            net_gex=-500000, prev_net_gex=-375000,
        )

        triggered = []
        messages = []

        vol_msg = _check_volume_spike(event, signal, multiplier=2.0)
        if vol_msg:
            triggered.append("volume_spike")
            messages.append(vol_msg)

        gex_msg = _check_gex_shift(event, signal, abs_threshold=1500)
        if gex_msg:
            triggered.append("gex_shift")
            messages.append(gex_msg)

        assert len(triggered) >= 2, "Both volume and GEX should fire"

        alert = CorrelationAlert(
            alert_type="confluence",
            social_event=event,
            market_signals=signal,
            signals_triggered=triggered,
            message="\n\n".join(messages),
            severity="high",
        )
        payload = alert.model_dump(mode="json")

        formatted = alert_service.format_alert_message(payload)
        assert "HIGH PRIORITY" in formatted

        alert_service.log_correlation_event(payload, alert_fired=True)
        events = alert_service.query_events(limit=10)
        assert len(events) == 1
        assert events[0]["alert_fired"] is True


class TestEventWindowIntegration:
    """Verify the EventWindow properly buffers events for correlation checks."""

    def test_social_event_and_signal_coexist(self):
        window = EventWindow(window_seconds=300)

        event = _make_social_event("Interest rates raised unexpectedly")
        window.add_social_event(event)

        signal = _make_signal(volume_ratio=2.5, volume_1min=5000, volume_20bar_avg=2000)
        window.add_signal(signal)

        recent = window.get_recent_social_events()
        latest = window.get_latest_signal()

        assert len(recent) == 1
        assert recent[0].event_id == event.event_id
        assert latest is not None
        assert latest.volume_ratio == 2.5


class TestHistoricalQueries:
    """Exercise query_events with various filters."""

    def test_query_by_source_filter(self, alert_service):
        # Insert events from different sources
        for source in ["twitter", "truth_social", "news_rss"]:
            event = _make_social_event("Tariffs increased", source=SocialSource(source))
            signal = _make_signal(volume_ratio=3.0, volume_1min=6000, volume_20bar_avg=2000)
            alert = CorrelationAlert(
                alert_type="volume_spike",
                social_event=event,
                market_signals=signal,
                signals_triggered=["volume_spike"],
                message="test",
                severity="medium",
            )
            payload = alert.model_dump(mode="json")
            alert_service.log_correlation_event(payload, alert_fired=True)

        all_events = alert_service.query_events(limit=10)
        assert len(all_events) == 3

        twitter_events = alert_service.query_events(source="twitter", limit=10)
        assert len(twitter_events) == 1

    def test_query_alert_fired_only(self, alert_service):
        event = _make_social_event("Crypto regulation announced")
        signal = _make_signal()
        alert = CorrelationAlert(
            alert_type="volume_spike",
            social_event=event,
            market_signals=signal,
            signals_triggered=["volume_spike"],
            message="test",
            severity="medium",
        )
        payload = alert.model_dump(mode="json")

        # One fired, one not
        alert_service.log_correlation_event(payload, alert_fired=True)
        alert_service.log_correlation_event(payload, alert_fired=False)

        fired = alert_service.query_events(alert_fired_only=True, limit=10)
        assert len(fired) == 1

        all_rows = alert_service.query_events(limit=10)
        assert len(all_rows) == 2


class TestSerializationRoundtrip:
    """Ensure alert payloads serialize/deserialize cleanly through JSON (like Redis would)."""

    def test_json_roundtrip(self):
        event = _make_social_event("Rate hike coming")
        signal = _make_signal(price_change_pct=-0.5, price=21500.0, price_2min_ago=21607.5)

        alert = CorrelationAlert(
            alert_type="price_move",
            social_event=event,
            market_signals=signal,
            signals_triggered=["price_move"],
            message="Price moved",
            severity="medium",
        )

        serialized = json.dumps(alert.model_dump(mode="json"), default=str)
        deserialized = json.loads(serialized)

        assert deserialized["alert_type"] == "price_move"
        assert deserialized["social_event"]["text"] == event.text
        assert deserialized["market_signals"]["price_change_pct"] == -0.5


# ---------------------------------------------------------------------------
# Backfill realized impact
# ---------------------------------------------------------------------------

class TestBackfillRealizedImpact:
    def test_backfill_updates_existing_row(self, alert_service):
        event = _make_social_event("Tariffs on Canada announced")
        signal = _make_signal(price_change_pct=0.6, price=21000.0, price_2min_ago=20874.0)
        alert = CorrelationAlert(
            alert_type="price_move",
            social_event=event,
            market_signals=signal,
            signals_triggered=["price_move"],
            message="price moved",
            severity="medium",
        )
        payload = alert.model_dump(mode="json")
        alert_service.log_correlation_event(payload, alert_fired=True)

        alert_service.backfill_realized_impact(
            social_event_id=event.event_id,
            realized_impact_score=72.5,
            price_t0=21000.0,
            price_t15=21315.0,
            price_ticker="MNQ",
            is_noise=False,
        )

        rows = alert_service.query_events(limit=1)
        assert len(rows) == 1
        assert rows[0]["realized_impact_score"] == pytest.approx(72.5)
        assert rows[0]["price_ticker"] == "MNQ"
        assert rows[0]["is_noise"] is False

    def test_backfill_marks_noise(self, alert_service):
        event = _make_social_event("Jimmy Fallon jokes about the Fed")
        signal = _make_signal()
        alert = CorrelationAlert(
            alert_type="volume_spike",
            social_event=event,
            market_signals=signal,
            signals_triggered=["volume_spike"],
            message="vol spike",
            severity="medium",
        )
        payload = alert.model_dump(mode="json")
        alert_service.log_correlation_event(payload, alert_fired=True)

        alert_service.backfill_realized_impact(
            social_event_id=event.event_id,
            realized_impact_score=2.1,
            price_t0=20000.0,
            price_t15=20005.0,
            price_ticker="MES",
            is_noise=True,
        )

        rows = alert_service.query_events(limit=1)
        assert rows[0]["is_noise"] is True
        assert rows[0]["realized_impact_score"] == pytest.approx(2.1)


# ---------------------------------------------------------------------------
# Date-range queries
# ---------------------------------------------------------------------------

class TestQueryDateFilters:
    def test_start_date_excludes_older_rows(self, alert_service):
        from datetime import timedelta

        old_event = _make_social_event("Old rate hike news")
        old_signal = _make_signal(volume_ratio=3.0, volume_1min=6000, volume_20bar_avg=2000)
        old_alert = CorrelationAlert(
            alert_type="volume_spike",
            social_event=old_event,
            market_signals=old_signal,
            signals_triggered=["volume_spike"],
            message="old",
            severity="medium",
        )
        old_payload = old_alert.model_dump(mode="json")
        # Backdate the timestamp
        old_payload["timestamp"] = (
            datetime.now(timezone.utc) - timedelta(hours=2)
        ).isoformat()
        alert_service.log_correlation_event(old_payload, alert_fired=True)

        new_event = _make_social_event("New tariff news just breaking")
        new_signal = _make_signal(volume_ratio=2.5, volume_1min=5000, volume_20bar_avg=2000)
        new_alert = CorrelationAlert(
            alert_type="volume_spike",
            social_event=new_event,
            market_signals=new_signal,
            signals_triggered=["volume_spike"],
            message="new",
            severity="medium",
        )
        alert_service.log_correlation_event(new_alert.model_dump(mode="json"), alert_fired=True)

        cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
        results = alert_service.query_events(start=cutoff, limit=10)
        assert len(results) == 1
        assert "New tariff" in results[0]["social_text"]

    def test_end_date_excludes_newer_rows(self, alert_service):
        from datetime import timedelta

        # Insert a row with a timestamp 2h ago and one now
        old_event = _make_social_event("Rate cut expected")
        old_signal = _make_signal(price_change_pct=0.5, price=21000.0, price_2min_ago=20895.0)
        old_alert = CorrelationAlert(
            alert_type="price_move",
            social_event=old_event,
            market_signals=old_signal,
            signals_triggered=["price_move"],
            message="old price move",
            severity="medium",
        )
        old_payload = old_alert.model_dump(mode="json")
        old_payload["timestamp"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        alert_service.log_correlation_event(old_payload, alert_fired=True)

        new_event = _make_social_event("Breaking: emergency rate cut")
        new_signal = _make_signal(price_change_pct=0.8, price=21200.0, price_2min_ago=21032.0)
        new_alert = CorrelationAlert(
            alert_type="price_move",
            social_event=new_event,
            market_signals=new_signal,
            signals_triggered=["price_move"],
            message="new price move",
            severity="medium",
        )
        alert_service.log_correlation_event(new_alert.model_dump(mode="json"), alert_fired=True)

        end = datetime.now(timezone.utc) - timedelta(hours=1)
        results = alert_service.query_events(end=end, limit=10)
        assert len(results) == 1
        assert "Rate cut expected" in results[0]["social_text"]


# ---------------------------------------------------------------------------
# Security: sanitization in formatted messages
# ---------------------------------------------------------------------------

class TestAlertMessageSecurity:
    def test_sanitizes_at_here(self, alert_service):
        event = _make_social_event("@here everyone check this")
        alert = CorrelationAlert(
            alert_type="volume_spike",
            social_event=event,
            market_signals=_make_signal(),
            signals_triggered=["volume_spike"],
            message="vol",
            severity="medium",
        )
        payload = alert.model_dump(mode="json")
        payload["social_event"]["author"] = "@here dangerous"
        msg = alert_service.format_alert_message(payload)
        assert "@here" not in msg

    def test_sanitizes_at_everyone_in_author(self, alert_service):
        event = _make_social_event("Normal news headline")
        alert = CorrelationAlert(
            alert_type="gex_shift",
            social_event=event,
            market_signals=_make_signal(gex_change_pct=20.0, net_gex=1200, prev_net_gex=1000),
            signals_triggered=["gex_shift"],
            message="gex",
            severity="medium",
        )
        payload = alert.model_dump(mode="json")
        payload["social_event"]["author"] = "@everyone danger"
        msg = alert_service.format_alert_message(payload)
        assert "@everyone" not in msg

    def test_truncates_very_long_author(self, alert_service):
        event = _make_social_event("Normal story")
        alert = CorrelationAlert(
            alert_type="volume_spike",
            social_event=event,
            market_signals=_make_signal(volume_ratio=3.0),
            signals_triggered=["volume_spike"],
            message="vol",
            severity="medium",
        )
        payload = alert.model_dump(mode="json")
        payload["social_event"]["author"] = "A" * 500
        msg = alert_service.format_alert_message(payload)
        # Author field should be truncated to 200 chars by _sanitize_text
        assert "A" * 201 not in msg


# ---------------------------------------------------------------------------
# No market signal → no alert fired
# ---------------------------------------------------------------------------

class TestNoSignalNoAlert:
    def test_volume_spike_rule_none_when_no_volume_data(self):
        event = _make_social_event("Breaking tariff announcement")
        signal = _make_signal()  # all None
        msg = _check_volume_spike(event, signal, multiplier=2.0)
        assert msg is None

    def test_gex_shift_rule_none_when_no_gex_data(self):
        event = _make_social_event("Fed minutes released")
        signal = _make_signal()
        msg = _check_gex_shift(event, signal, abs_threshold=1500)
        assert msg is None

    def test_price_move_rule_none_when_no_price_data(self):
        event = _make_social_event("China trade deal")
        signal = _make_signal()
        msg = _check_price_move(event, signal, pct_threshold=0.3)
        assert msg is None


# ---------------------------------------------------------------------------
# Sentiment propagates into persisted row
# ---------------------------------------------------------------------------

class TestSentimentPersistence:
    def test_bullish_event_text_survives_roundtrip(self, alert_service):
        event = _make_social_event("Major rate cut announced, economy booming, markets rally")
        signal = _make_signal(price_change_pct=0.6, price=21100.0, price_2min_ago=20974.0)
        alert = CorrelationAlert(
            alert_type="price_move",
            social_event=event,
            market_signals=signal,
            signals_triggered=["price_move"],
            message="price up",
            severity="medium",
        )
        payload = alert.model_dump(mode="json")
        alert_service.log_correlation_event(payload, alert_fired=True)

        rows = alert_service.query_events(limit=1)
        assert len(rows) == 1
        # Text is stored (truncated to 500 chars)
        assert rows[0]["social_text"][:50] == event.text[:50]

    def test_multiple_alerts_ordered_newest_first(self, alert_service):
        texts = ["First news", "Second news", "Third news"]
        for i, text in enumerate(texts):
            event = _make_social_event(text)
            signal = _make_signal(volume_ratio=2.5 + i * 0.5,
                                   volume_1min=5000 + i * 1000,
                                   volume_20bar_avg=2000)
            alert = CorrelationAlert(
                alert_type="volume_spike",
                social_event=event,
                market_signals=signal,
                signals_triggered=["volume_spike"],
                message=f"vol {i}",
                severity="medium",
            )
            alert_service.log_correlation_event(alert.model_dump(mode="json"), alert_fired=True)

        rows = alert_service.query_events(limit=10)
        assert len(rows) == 3
        # Most recent inserted comes first (ORDER BY timestamp DESC)
        assert "Third news" in rows[0]["social_text"]
