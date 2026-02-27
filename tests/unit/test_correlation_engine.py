"""Tests for the correlation engine rules and event window."""

from datetime import datetime, timezone, timedelta

import pytest

from src.models.social_event import SocialEvent, SocialSource
from src.services.correlation_engine import (
    CorrelationAlert,
    EventWindow,
    MarketSignalSnapshot,
    _check_gex_shift,
    _check_price_move,
    _check_volume_spike,
)


def _make_social_event(text="Tariffs on China raised to 100%", score=3):
    return SocialEvent(
        event_id="test123",
        timestamp=datetime.now(timezone.utc),
        source=SocialSource.TWITTER,
        author="@realDonaldTrump",
        text=text,
        relevance_score=score,
        keywords_matched=["tariff", "china"],
        categories_matched=["tariff_trade", "geopolitical"],
    )


class TestEventWindow:
    def test_add_and_retrieve_social_events(self):
        window = EventWindow(window_seconds=300)
        event = _make_social_event()
        window.add_social_event(event)
        recent = window.get_recent_social_events()
        assert len(recent) == 1
        assert recent[0].event_id == "test123"

    def test_expired_events_evicted(self):
        window = EventWindow(window_seconds=1)
        old_event = SocialEvent(
            event_id="old",
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=5),
            source=SocialSource.TWITTER,
            author="@user",
            text="old post about tariff",
            relevance_score=3,
        )
        window.add_social_event(old_event)
        recent = window.get_recent_social_events()
        assert len(recent) == 0

    def test_signal_added_and_retrieved(self):
        window = EventWindow(window_seconds=300)
        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            volume_ratio=3.5,
        )
        window.add_signal(signal)
        latest = window.get_latest_signal()
        assert latest is not None
        assert latest.volume_ratio == 3.5

    def test_empty_window_returns_none(self):
        window = EventWindow()
        assert window.get_latest_signal() is None
        assert window.get_recent_social_events() == []


class TestVolumeSpike:
    def test_fires_on_match(self):
        event = _make_social_event()
        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol="NQ",
            volume_1min=5000,
            volume_20bar_avg=2000,
            volume_ratio=2.5,
        )
        msg = _check_volume_spike(event, signal, multiplier=2.0)
        assert msg is not None
        assert "VOLUME SPIKE" in msg

    def test_silent_below_threshold(self):
        event = _make_social_event()
        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            volume_ratio=1.5,
        )
        msg = _check_volume_spike(event, signal, multiplier=2.0)
        assert msg is None

    def test_silent_when_no_data(self):
        event = _make_social_event()
        signal = MarketSignalSnapshot(timestamp=datetime.now(timezone.utc))
        msg = _check_volume_spike(event, signal, multiplier=2.0)
        assert msg is None


class TestGexShift:
    def test_fires_on_match(self):
        event = _make_social_event()
        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol="ES_SPX",
            net_gex=1150,
            prev_net_gex=1000,
            gex_change_pct=15.0,
        )
        msg = _check_gex_shift(event, signal, pct_threshold=15.0)
        assert msg is not None
        assert "GEX SHIFT" in msg

    def test_silent_below_threshold(self):
        event = _make_social_event()
        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            gex_change_pct=5.0,
        )
        msg = _check_gex_shift(event, signal, pct_threshold=15.0)
        assert msg is None


class TestPriceMove:
    def test_fires_on_match(self):
        event = _make_social_event()
        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol="NQ",
            price=20100,
            price_2min_ago=20000,
            price_change_pct=0.5,
        )
        msg = _check_price_move(event, signal, pct_threshold=0.3)
        assert msg is not None
        assert "PRICE MOVE" in msg

    def test_silent_below_threshold(self):
        event = _make_social_event()
        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            price_change_pct=0.1,
        )
        msg = _check_price_move(event, signal, pct_threshold=0.3)
        assert msg is None


class TestConfluence:
    def test_multiple_signals_detected(self):
        """When volume+GEX+price all trigger, we expect a confluence scenario."""
        event = _make_social_event()
        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol="NQ",
            volume_ratio=3.0,
            volume_1min=6000,
            volume_20bar_avg=2000,
            gex_change_pct=20.0,
            net_gex=1200,
            prev_net_gex=1000,
            price_change_pct=0.5,
            price=20100,
            price_2min_ago=20000,
        )
        # All rules should fire
        vol_msg = _check_volume_spike(event, signal, 2.0)
        gex_msg = _check_gex_shift(event, signal, 15.0)
        price_msg = _check_price_move(event, signal, 0.3)

        assert vol_msg is not None
        assert gex_msg is not None
        assert price_msg is not None

    def test_single_signal_not_confluence(self):
        """Only one signal → not confluence."""
        event = _make_social_event()
        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            volume_ratio=3.0,
            volume_1min=6000,
            volume_20bar_avg=2000,
            gex_change_pct=5.0,  # below threshold
            price_change_pct=0.1,  # below threshold
        )
        vol_msg = _check_volume_spike(event, signal, 2.0)
        gex_msg = _check_gex_shift(event, signal, 15.0)
        price_msg = _check_price_move(event, signal, 0.3)

        triggered = [m for m in [vol_msg, gex_msg, price_msg] if m is not None]
        assert len(triggered) == 1


class TestExactThreshold:
    def test_volume_exactly_at_threshold(self):
        event = _make_social_event()
        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            volume_ratio=2.0,
            volume_1min=4000,
            volume_20bar_avg=2000,
        )
        msg = _check_volume_spike(event, signal, 2.0)
        assert msg is not None  # >= threshold

    def test_volume_just_below_threshold(self):
        event = _make_social_event()
        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            volume_ratio=1.99,
        )
        msg = _check_volume_spike(event, signal, 2.0)
        assert msg is None

    def test_gex_negative_shift(self):
        event = _make_social_event()
        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            gex_change_pct=-20.0,
            net_gex=800,
            prev_net_gex=1000,
        )
        msg = _check_gex_shift(event, signal, 15.0)
        assert msg is not None  # abs(-20) >= 15
