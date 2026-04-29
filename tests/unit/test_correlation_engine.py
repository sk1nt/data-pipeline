"""Tests for the correlation engine rules and event window."""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call
import json

import pytest

from src.models.social_event import SocialEvent, SocialSource, Sentiment
from src.services.correlation_engine import (
    CORRELATION_ALERT_CHANNEL,
    CorrelationAlert,
    CorrelationEngine,
    EventWindow,
    MarketSignalSnapshot,
    _check_gex_shift,
    _check_price_move,
    _check_volume_spike,
)


def _make_social_event(text="Tariffs on China raised to 100%", score=3, event_id="test123"):
    return SocialEvent(
        event_id=event_id,
        timestamp=datetime.now(timezone.utc),
        source=SocialSource.TWITTER,
        author="@realDonaldTrump",
        text=text,
        relevance_score=score,
        keywords_matched=["tariff", "china"],
        categories_matched=["tariff_trade", "geopolitical"],
    )


def _make_engine(
    *,
    volume_spike_multiplier=2.0,
    gex_shift_pct=15.0,
    price_move_pct=0.3,
    cooldown_seconds=300,
    window_seconds=300,
):
    """Build a CorrelationEngine with a mock Redis client."""
    mock_redis = MagicMock()
    mock_redis.client = MagicMock()
    return CorrelationEngine(
        redis_client=mock_redis,
        volume_spike_multiplier=volume_spike_multiplier,
        gex_shift_pct=gex_shift_pct,
        price_move_pct=price_move_pct,
        cooldown_seconds=cooldown_seconds,
        window_seconds=window_seconds,
    ), mock_redis


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


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_same_event_and_rule_suppressed_within_cooldown(self):
        engine, mock_redis = _make_engine(cooldown_seconds=300)
        event = _make_social_event(event_id="evt_cd")
        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            volume_ratio=3.0, volume_1min=6000, volume_20bar_avg=2000,
        )
        engine.window.add_social_event(event)
        engine.window.add_signal(signal)

        # First call publishes
        engine._check_correlations(event)
        assert mock_redis.client.publish.call_count == 1

        # Second call within cooldown — same event_id + same rule → suppressed
        engine._check_correlations(event)
        assert mock_redis.client.publish.call_count == 1

    def test_different_events_not_affected_by_cooldown(self):
        engine, mock_redis = _make_engine(cooldown_seconds=300)
        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            volume_ratio=3.0, volume_1min=6000, volume_20bar_avg=2000,
        )
        engine.window.add_signal(signal)

        for i in range(3):
            evt = _make_social_event(event_id=f"evt_{i}")
            engine.window.add_social_event(evt)
            engine._check_correlations(evt)

        assert mock_redis.client.publish.call_count == 3

    def test_expired_cooldown_allows_re_fire(self):
        engine, mock_redis = _make_engine(cooldown_seconds=1)
        event = _make_social_event(event_id="evt_exp")
        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            volume_ratio=3.0, volume_1min=6000, volume_20bar_avg=2000,
        )
        engine.window.add_social_event(event)
        engine.window.add_signal(signal)

        engine._check_correlations(event)
        assert mock_redis.client.publish.call_count == 1

        # Backdate cooldown entries so they appear expired
        for key in list(engine._cooldowns.keys()):
            engine._cooldowns[key] = datetime.now(timezone.utc) - timedelta(seconds=10)

        engine._check_correlations(event)
        assert mock_redis.client.publish.call_count == 2


# ---------------------------------------------------------------------------
# Volume bar accumulation
# ---------------------------------------------------------------------------

class TestVolumeBarUpdate:
    def test_accumulates_volume_within_same_minute(self):
        engine, _ = _make_engine()
        now = datetime.now(timezone.utc)
        engine._update_volume_bar("MNQ", now, 1000.0)
        engine._update_volume_bar("MNQ", now + timedelta(seconds=30), 500.0)

        bars = engine._volume_bars["MNQ"]
        assert len(bars) == 1
        assert bars[-1][1] == 1500.0

    def test_creates_new_bar_after_60_seconds(self):
        engine, _ = _make_engine()
        now = datetime.now(timezone.utc)
        engine._update_volume_bar("MNQ", now, 1000.0)
        engine._update_volume_bar("MNQ", now + timedelta(seconds=61), 800.0)

        bars = engine._volume_bars["MNQ"]
        assert len(bars) == 2
        assert bars[0][1] == 1000.0
        assert bars[1][1] == 800.0

    def test_separate_bars_per_symbol(self):
        engine, _ = _make_engine()
        now = datetime.now(timezone.utc)
        engine._update_volume_bar("MNQ", now, 1000.0)
        engine._update_volume_bar("MES", now, 500.0)

        assert "MNQ" in engine._volume_bars
        assert "MES" in engine._volume_bars
        assert engine._volume_bars["MNQ"][-1][1] == 1000.0
        assert engine._volume_bars["MES"][-1][1] == 500.0


# ---------------------------------------------------------------------------
# _build_signal
# ---------------------------------------------------------------------------

class TestBuildSignal:
    def test_volume_ratio_computed_from_bars(self):
        from collections import deque
        engine, _ = _make_engine()
        now = datetime.now(timezone.utc)
        bars = deque(maxlen=20)
        # 3 old bars at 1000, then a spike at 4000
        for i in range(3):
            bars.append((now - timedelta(minutes=3 - i), 1000.0))
        bars.append((now, 4000.0))
        engine._volume_bars["MNQ"] = bars

        signal = engine._build_signal(now, "MNQ")
        assert signal.volume_1min == 4000.0
        assert signal.volume_20bar_avg == pytest.approx(1750.0, rel=1e-3)  # (3000+4000)/4
        assert signal.volume_ratio == pytest.approx(4000.0 / 1750.0, rel=1e-3)

    def test_price_change_pct_computed(self):
        from collections import deque
        engine, _ = _make_engine()
        now = datetime.now(timezone.utc)
        hist = deque(maxlen=120)
        hist.append((now - timedelta(seconds=130), 20000.0))
        hist.append((now, 20100.0))
        engine._price_history["MNQ"] = hist

        signal = engine._build_signal(now, "MNQ")
        assert signal.price == 20100.0
        assert signal.price_2min_ago == 20000.0
        assert signal.price_change_pct == pytest.approx(0.5, rel=1e-3)

    def test_no_data_returns_empty_signal(self):
        engine, _ = _make_engine()
        signal = engine._build_signal(datetime.now(timezone.utc), "MNQ")
        assert signal.volume_ratio is None
        assert signal.price is None
        assert signal.price_change_pct is None


# ---------------------------------------------------------------------------
# Full engine: _check_correlations publishes correct payload
# ---------------------------------------------------------------------------

class TestCheckCorrelationsPublish:
    def test_publishes_alert_to_redis_channel(self):
        engine, mock_redis = _make_engine()
        event = _make_social_event(event_id="pub_test")
        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol="MNQ",
            volume_ratio=3.5, volume_1min=7000, volume_20bar_avg=2000,
        )
        engine.window.add_signal(signal)
        engine._check_correlations(event)

        assert mock_redis.client.publish.call_count == 1
        channel_arg, payload_arg = mock_redis.client.publish.call_args[0]
        assert channel_arg == CORRELATION_ALERT_CHANNEL
        published = json.loads(payload_arg)
        assert published["alert_type"] == "volume_spike"
        assert "volume_spike" in published["signals_triggered"]

    def test_confluence_sets_high_severity(self):
        engine, mock_redis = _make_engine(gex_shift_pct=5.0, price_move_pct=0.1)
        event = _make_social_event(event_id="conf_test")
        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol="MNQ",
            volume_ratio=3.0, volume_1min=6000, volume_20bar_avg=2000,
            gex_change_pct=10.0, net_gex=1100, prev_net_gex=1000,
            price_change_pct=0.5, price=20100, price_2min_ago=20000,
        )
        engine.window.add_signal(signal)
        engine._check_correlations(event)

        assert mock_redis.client.publish.call_count == 1
        published = json.loads(mock_redis.client.publish.call_args[0][1])
        assert published["severity"] == "high"
        assert published["alert_type"] == "confluence"
        assert len(published["signals_triggered"]) >= 2

    def test_no_publish_when_no_signal(self):
        engine, mock_redis = _make_engine()
        event = _make_social_event(event_id="nosig_test")
        engine._check_correlations(event)
        assert mock_redis.client.publish.call_count == 0

    def test_no_publish_when_below_all_thresholds(self):
        engine, mock_redis = _make_engine()
        event = _make_social_event(event_id="below_test")
        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            volume_ratio=1.5,     # below 2.0 multiplier
            gex_change_pct=3.0,   # below 15.0 pct
            price_change_pct=0.1, # below 0.3 pct
        )
        engine.window.add_signal(signal)
        engine._check_correlations(event)
        assert mock_redis.client.publish.call_count == 0

    def test_published_payload_includes_social_event_fields(self):
        engine, mock_redis = _make_engine()
        event = _make_social_event(event_id="field_test")
        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol="MES",
            volume_ratio=2.5, volume_1min=5000, volume_20bar_avg=2000,
        )
        engine.window.add_signal(signal)
        engine._check_correlations(event)

        published = json.loads(mock_redis.client.publish.call_args[0][1])
        assert published["social_event"]["event_id"] == "field_test"
        assert published["social_event"]["author"] == "@realDonaldTrump"
        assert published["market_signals"]["volume_ratio"] == pytest.approx(2.5)
        assert published["market_signals"]["symbol"] == "MES"


# ---------------------------------------------------------------------------
# Market-signal-triggered lookback
# ---------------------------------------------------------------------------

class TestMarketSignalLookback:
    def test_new_signal_checks_recent_social_events(self):
        """A trade tick arriving after a social event should still correlate."""
        engine, mock_redis = _make_engine()
        event = _make_social_event(event_id="lookback_evt")
        engine.window.add_social_event(event)

        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            volume_ratio=3.0, volume_1min=6000, volume_20bar_avg=2000,
        )
        engine.window.add_signal(signal)
        engine._check_correlations_for_market_signal()

        assert mock_redis.client.publish.call_count == 1

    def test_expired_social_event_not_correlated(self):
        """Social events older than window_seconds are ignored on market tick."""
        engine, mock_redis = _make_engine(window_seconds=60)
        old_event = SocialEvent(
            event_id="old_evt",
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=120),
            source=SocialSource.TWITTER,
            author="@user",
            text="Tariffs on China raised to 100%",
            relevance_score=3,
        )
        engine.window.add_social_event(old_event)

        signal = MarketSignalSnapshot(
            timestamp=datetime.now(timezone.utc),
            volume_ratio=5.0, volume_1min=10000, volume_20bar_avg=2000,
        )
        engine.window.add_signal(signal)
        engine._check_correlations_for_market_signal()

        assert mock_redis.client.publish.call_count == 0


# ---------------------------------------------------------------------------
# EventWindow: within_seconds filter
# ---------------------------------------------------------------------------

class TestEventWindowTimeFilter:
    def test_within_seconds_filters_older_events(self):
        window = EventWindow(window_seconds=300)
        old = SocialEvent(
            event_id="old",
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=200),
            source=SocialSource.TWITTER, author="@u", text="old tariff",
            relevance_score=3,
        )
        recent = SocialEvent(
            event_id="recent",
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=20),
            source=SocialSource.TWITTER, author="@u", text="new tariff",
            relevance_score=3,
        )
        window.add_social_event(old)
        window.add_social_event(recent)

        results = window.get_recent_social_events(within_seconds=60)
        ids = [e.event_id for e in results]
        assert "recent" in ids
        assert "old" not in ids

    def test_get_recent_returns_all_within_window(self):
        window = EventWindow(window_seconds=300)
        for i in range(5):
            evt = SocialEvent(
                event_id=f"e{i}",
                timestamp=datetime.now(timezone.utc) - timedelta(seconds=i * 30),
                source=SocialSource.NEWS_RSS, author="@news", text=f"story {i}",
                relevance_score=1,
            )
            window.add_social_event(evt)

        results = window.get_recent_social_events()
        assert len(results) == 5
