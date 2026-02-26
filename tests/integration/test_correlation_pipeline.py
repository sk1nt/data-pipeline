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
    _check_uw_flow,
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
    score, keywords, categories = scorer.score(text)
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
    uw_max_premium: float | None = None,
    uw_put_call_ratio: float | None = None,
    uw_prev_ratio: float | None = None,
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
        uw_max_premium=uw_max_premium,
        uw_put_call_ratio=uw_put_call_ratio,
        uw_prev_ratio=uw_prev_ratio,
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
            gex_change_pct=-25.0, net_gex=-500000, prev_net_gex=-375000,
        )

        triggered = []
        messages = []

        vol_msg = _check_volume_spike(event, signal, multiplier=2.0)
        if vol_msg:
            triggered.append("volume_spike")
            messages.append(vol_msg)

        gex_msg = _check_gex_shift(event, signal, pct_threshold=15.0)
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


class TestEndToEndUwFlow:
    """Large option premium after geopolitical tweet → uw_flow alert."""

    def test_large_premium_alert(self, alert_service):
        event = _make_social_event("Military action authorized in South China Sea")

        signal = _make_signal(uw_max_premium=5_000_000)
        msg = _check_uw_flow(event, signal, premium_threshold=1_000_000)
        assert msg is not None

        alert = CorrelationAlert(
            alert_type="uw_flow",
            social_event=event,
            market_signals=signal,
            signals_triggered=["uw_flow"],
            message=msg,
            severity="medium",
        )
        payload = alert.model_dump(mode="json")

        alert_service.log_correlation_event(payload, alert_fired=True)
        events = alert_service.query_events(limit=10)
        assert len(events) == 1


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
