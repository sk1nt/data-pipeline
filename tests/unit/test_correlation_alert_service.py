"""Tests for the CorrelationAlertService — formatting, sanitization, and DuckDB persistence."""

import os
import tempfile
from datetime import datetime, timezone

import pytest

from src.services.correlation_alert_service import CorrelationAlertService


@pytest.fixture
def alert_service(tmp_path):
    db_path = str(tmp_path / "test_correlation.db")
    return CorrelationAlertService(db_path=db_path)


def _make_alert(
    alert_type="volume_spike",
    severity="medium",
    signals=None,
    social_text="Tariffs on China raised to 100%",
):
    return {
        "alert_id": "abc123",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "alert_type": alert_type,
        "severity": severity,
        "signals_triggered": signals or ["volume_spike"],
        "message": "🚨 **VOLUME SPIKE** after social event\n> Tariffs...\nVolume ratio: 3.5× avg",
        "social_event": {
            "event_id": "evt001",
            "source": "twitter",
            "author": "@realDonaldTrump",
            "text": social_text,
            "relevance_score": 3,
            "url": "https://example.com/post",
        },
        "market_signals": {
            "volume_ratio": 3.5,
            "volume_1min": 7000,
            "volume_20bar_avg": 2000,
            "gex_change_pct": None,
            "price_change_pct": None,
        },
    }


class TestFormatAlertMessage:
    def test_medium_severity_format(self, alert_service):
        alert = _make_alert()
        msg = alert_service.format_alert_message(alert)
        assert "CORRELATION ALERT" in msg
        assert "🟡" in msg
        assert "volume_spike" in msg

    def test_high_severity_format(self, alert_service):
        alert = _make_alert(
            alert_type="confluence",
            severity="high",
            signals=["volume_spike", "gex_shift"],
        )
        msg = alert_service.format_alert_message(alert)
        assert "🔴" in msg
        assert "HIGH PRIORITY" in msg
        assert "confluence" in msg

    def test_sanitizes_at_everyone(self, alert_service):
        # format_alert_message sanitizes the author field; social_text is in
        # the pre-formatted message field and isn't re-included directly.
        alert = _make_alert()
        alert["social_event"]["author"] = "@everyone look here"
        msg = alert_service.format_alert_message(alert)
        assert "@everyone" not in msg

    def test_sanitizes_backticks(self, alert_service):
        alert = _make_alert(social_text="```python\nimport os\nos.system('rm -rf /')```")
        msg = alert_service.format_alert_message(alert)
        # Backticks should be replaced
        assert "```" not in msg


class TestDuckDBPersistence:
    def test_log_and_query(self, alert_service):
        alert = _make_alert()
        alert_service.log_correlation_event(alert, alert_fired=True)

        events = alert_service.query_events(limit=10)
        assert len(events) == 1
        assert events[0]["social_event_id"] == "evt001"
        assert events[0]["alert_fired"] is True

    def test_log_no_alert_event(self, alert_service):
        alert = _make_alert()
        alert_service.log_correlation_event(alert, alert_fired=False)

        events = alert_service.query_events(alert_fired_only=True, limit=10)
        assert len(events) == 0

        all_events = alert_service.query_events(limit=10)
        assert len(all_events) == 1
        assert all_events[0]["alert_fired"] is False

    def test_query_by_source(self, alert_service):
        alert = _make_alert()
        alert_service.log_correlation_event(alert, alert_fired=True)

        events = alert_service.query_events(source="twitter", limit=10)
        assert len(events) == 1

        events = alert_service.query_events(source="truth_social", limit=10)
        assert len(events) == 0

    def test_multiple_events(self, alert_service):
        for i in range(5):
            alert = _make_alert()
            alert["social_event"]["event_id"] = f"evt{i:03d}"
            alert_service.log_correlation_event(alert, alert_fired=True)

        events = alert_service.query_events(limit=3)
        assert len(events) == 3


class TestSanitizeText:
    def test_truncation(self):
        long_text = "x" * 500
        result = CorrelationAlertService._sanitize_text(long_text, max_length=100)
        assert len(result) <= 101  # 100 + "…"
        assert result.endswith("…")

    def test_at_here_removed(self):
        result = CorrelationAlertService._sanitize_text("@here look at this")
        assert "@here" not in result
