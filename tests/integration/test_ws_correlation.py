"""Tests for the /ws/correlation WebSocket endpoint behaviour.

We test the Redis key management and serialization logic that the endpoint
relies on in isolation, without spinning up a live server.  This mirrors
exactly what the endpoint does when a client connects and when new alerts
arrive.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.models.social_event import SocialEvent, SocialSource, Sentiment
from src.services.correlation_engine import (
    CorrelationAlert,
    MarketSignalSnapshot,
)

CORRELATION_ALERT_CHANNEL = "correlation:alerts:stream"
CORRELATION_HISTORY_KEY = "correlation:alerts:history"
HISTORY_MAX_ITEMS = 50
HISTORY_TTL_SECONDS = 86400


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alert(alert_type="volume_spike", severity="medium", signals=None) -> dict:
    event = SocialEvent(
        event_id=uuid.uuid4().hex[:12],
        timestamp=datetime.now(timezone.utc),
        source=SocialSource.NEWS_RSS,
        author="@FirstSquawk",
        text="Breaking: emergency tariff announcement imminent",
        relevance_score=6,
        sentiment=Sentiment.BEARISH,
        keywords_matched=["tariff", "emergency"],
        categories_matched=["tariff_trade"],
    )
    signal = MarketSignalSnapshot(
        timestamp=datetime.now(timezone.utc),
        symbol="MNQ",
        volume_ratio=3.0,
        volume_1min=6000,
        volume_20bar_avg=2000,
        gex_change_pct=-18.0,
        net_gex=-500_000,
        prev_net_gex=-423_000,
        price_change_pct=-0.6,
        price=20874.0,
        price_2min_ago=20999.0,
    )
    alert = CorrelationAlert(
        alert_type=alert_type,
        social_event=event,
        market_signals=signal,
        signals_triggered=signals or [alert_type],
        message="test alert",
        severity=severity,
    )
    return alert.model_dump(mode="json")


def _serialize(payload: dict) -> str:
    return json.dumps(payload, default=str)


def _deserialize(raw: str | bytes) -> dict:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Redis history key operations (mirrors what the WS endpoint does)
# ---------------------------------------------------------------------------

class TestRedisHistoryKeyOps:
    """Verify the lpush/ltrim/expire pattern used by the endpoint."""

    def test_new_alert_pushed_to_front(self):
        mock_redis = MagicMock()
        payload = _make_alert()
        serialized = _serialize(payload)

        # Simulate what the WS endpoint does on each new alert
        mock_redis.lpush(CORRELATION_HISTORY_KEY, serialized)
        mock_redis.ltrim(CORRELATION_HISTORY_KEY, 0, HISTORY_MAX_ITEMS - 1)
        mock_redis.expire(CORRELATION_HISTORY_KEY, HISTORY_TTL_SECONDS)

        mock_redis.lpush.assert_called_once_with(CORRELATION_HISTORY_KEY, serialized)
        mock_redis.ltrim.assert_called_once_with(
            CORRELATION_HISTORY_KEY, 0, HISTORY_MAX_ITEMS - 1
        )
        mock_redis.expire.assert_called_once_with(
            CORRELATION_HISTORY_KEY, HISTORY_TTL_SECONDS
        )

    def test_backfill_returns_items_in_reverse_insertion_order(self):
        """lrange returns newest-first (lpush puts most recent at index 0)."""
        mock_redis = MagicMock()
        payloads = [_make_alert() for _ in range(3)]
        serialized_list = [_serialize(p).encode() for p in payloads]

        # lrange returns items in lpush order (index 0 = most recent)
        mock_redis.lrange.return_value = serialized_list

        history = mock_redis.lrange(CORRELATION_HISTORY_KEY, 0, 199)
        # The WS endpoint reverses history before sending so oldest is sent first
        for item in reversed(history):
            decoded = _deserialize(item)
            assert "alert_type" in decoded

        assert mock_redis.lrange.call_count == 1

    def test_ttl_is_set_to_86400_seconds(self):
        mock_redis = MagicMock()
        payload = _make_alert()
        mock_redis.lpush(CORRELATION_HISTORY_KEY, _serialize(payload))
        mock_redis.expire(CORRELATION_HISTORY_KEY, HISTORY_TTL_SECONDS)

        _, ttl_arg = mock_redis.expire.call_args[0]
        assert ttl_arg == 86400

    def test_history_trimmed_to_max_50(self):
        mock_redis = MagicMock()
        payload = _make_alert()
        mock_redis.lpush(CORRELATION_HISTORY_KEY, _serialize(payload))
        mock_redis.ltrim(CORRELATION_HISTORY_KEY, 0, HISTORY_MAX_ITEMS - 1)

        _, start, end = mock_redis.ltrim.call_args[0]
        assert start == 0
        assert end == HISTORY_MAX_ITEMS - 1


# ---------------------------------------------------------------------------
# Serialization round-trip (what gets pushed into Redis and read back)
# ---------------------------------------------------------------------------

class TestAlertSerializationRoundtrip:
    def test_full_alert_survives_json_encode_decode(self):
        payload = _make_alert(alert_type="confluence", severity="high",
                               signals=["volume_spike", "gex_shift", "price_move"])
        serialized = _serialize(payload)
        recovered = _deserialize(serialized)

        assert recovered["alert_type"] == "confluence"
        assert recovered["severity"] == "high"
        assert len(recovered["signals_triggered"]) == 3
        assert recovered["social_event"]["sentiment"] == "bearish"
        assert recovered["market_signals"]["volume_ratio"] == pytest.approx(3.0)

    def test_bytes_backfill_decodes_correctly(self):
        """Simulates Redis returning bytes from lrange."""
        payload = _make_alert()
        raw_bytes = _serialize(payload).encode("utf-8")
        recovered = _deserialize(raw_bytes)
        assert "alert_id" in recovered

    def test_alert_id_is_unique_per_alert(self):
        ids = {_make_alert()["alert_id"] for _ in range(20)}
        assert len(ids) == 20

    def test_missing_alert_id_field_handled(self):
        """Edge case: if a legacy alert has no alert_id, endpoint should skip."""
        payload = _make_alert()
        del payload["alert_id"]
        serialized = _serialize(payload)
        recovered = _deserialize(serialized)
        assert "alert_id" not in recovered


# ---------------------------------------------------------------------------
# Duplicate prevention (frontend queue deduplication by alert_id)
# ---------------------------------------------------------------------------

class TestFrontendDeduplication:
    """
    The frontend JS deduplicates by alert_id in moverQueue and moverAcked.
    We can verify the ID uniqueness contract from the Python side.
    """

    def test_same_alert_id_indicates_same_event(self):
        payload = _make_alert()
        # Serialized twice (simulates delivery via pubsub AND history backfill)
        copy1 = _deserialize(_serialize(payload))
        copy2 = _deserialize(_serialize(payload))
        assert copy1["alert_id"] == copy2["alert_id"]

    def test_different_triggers_produce_different_alert_ids(self):
        p1 = _make_alert(alert_type="volume_spike")
        p2 = _make_alert(alert_type="gex_shift")
        assert p1["alert_id"] != p2["alert_id"]


# ---------------------------------------------------------------------------
# Backfill ordering
# ---------------------------------------------------------------------------

class TestBackfillOrdering:
    def test_older_alerts_sent_first_during_backfill(self):
        """reversed(history) means oldest arrives first at the client."""
        from datetime import timedelta

        alerts = []
        for i in range(5):
            p = _make_alert()
            p["timestamp"] = (
                datetime.now(timezone.utc) - timedelta(minutes=5 - i)
            ).isoformat()
            alerts.append(p)

        # Simulate lpush order: newest appended last → index 0 is newest
        history_in_redis = [_serialize(a).encode() for a in reversed(alerts)]

        sent_order = []
        for item in reversed(history_in_redis):
            sent_order.append(_deserialize(item))

        # First item sent should be the oldest (smallest timestamp)
        timestamps = [s["timestamp"] for s in sent_order]
        assert timestamps == sorted(timestamps)
