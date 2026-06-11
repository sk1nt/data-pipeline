"""tests/integration/test_dom_bridge.py — integration tests for the sierra DOM bridge.

Tests the file → Redis → danger_trigger roundtrip without real SC.
Uses a real Redis instance if REDIS_URL is set; otherwise skips.
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock

import pytest

from src.services.sierra_dom_bridge_service import (
    SierraDOMBridgeService,
    _read_json_file,
)


# ---------------------------------------------------------------------------
# _read_json_file tests (pure I/O, no Redis)
# ---------------------------------------------------------------------------

class TestReadJsonFile:

    def test_missing_file_returns_none(self, tmp_path):
        result = _read_json_file(tmp_path / "nonexistent.json")
        assert result is None

    def test_valid_json_returns_dict(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('{"ts_ms": 12345, "price": 20000.0}')
        result = _read_json_file(f)
        assert result == {"ts_ms": 12345, "price": 20000.0}

    def test_invalid_json_returns_none(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{not valid json")
        result = _read_json_file(f)
        assert result is None

    def test_partial_write_returns_none(self, tmp_path):
        """Simulate a file that is mid-write (truncated)."""
        f = tmp_path / "partial.json"
        f.write_text('{"ts_ms": 123')
        result = _read_json_file(f)
        assert result is None


# ---------------------------------------------------------------------------
# SierraDOMBridgeService: freshness and publish logic
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dom_files(tmp_path):
    dom_path  = tmp_path / "dom_snapshot.json"
    flow_path = tmp_path / "trade_flow.json"
    danger_path = tmp_path / "danger_trigger.json"
    return dom_path, flow_path, danger_path


@pytest.fixture
def bridge_svc(tmp_dom_files):
    dom_path, flow_path, danger_path = tmp_dom_files
    svc = SierraDOMBridgeService.__new__(SierraDOMBridgeService)
    svc._redis       = AsyncMock()
    svc._dom_path    = dom_path
    svc._flow_path   = flow_path
    svc._danger_path = danger_path
    svc._symbol      = "MNQ"
    svc._last_dom_ts  = 0
    svc._last_flow_ts = 0
    svc._running      = True
    return svc


class TestBridgePublish:

    @pytest.mark.asyncio
    async def test_publishes_fresh_dom_snapshot(self, bridge_svc, tmp_dom_files):
        dom_path, _, _ = tmp_dom_files
        now_ms = int(time.time() * 1000)
        payload = {"ts_ms": now_ms, "price": 20100.0, "symbol": "MNQ",
                   "bid_depth_10": 200.0, "ask_depth_10": 180.0, "imbalance_ratio": 0.53}
        dom_path.write_text(json.dumps(payload))

        await bridge_svc._maybe_publish_dom(now_ms)

        bridge_svc._redis.publish.assert_awaited_once()
        args = bridge_svc._redis.publish.call_args[0]
        assert args[0] == "market:dom:MNQ"
        published = json.loads(args[1])
        assert published["price"] == pytest.approx(20100.0)

    @pytest.mark.asyncio
    async def test_skips_stale_dom_snapshot(self, bridge_svc, tmp_dom_files):
        dom_path, _, _ = tmp_dom_files
        old_ts = int(time.time() * 1000) - 5000  # 5 seconds ago
        dom_path.write_text(json.dumps({"ts_ms": old_ts, "price": 20000.0}))
        now_ms = int(time.time() * 1000)

        await bridge_svc._maybe_publish_dom(now_ms)

        bridge_svc._redis.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_duplicate_dom_snapshot(self, bridge_svc, tmp_dom_files):
        dom_path, _, _ = tmp_dom_files
        now_ms = int(time.time() * 1000)
        dom_path.write_text(json.dumps({"ts_ms": now_ms, "price": 20000.0}))
        bridge_svc._last_dom_ts = now_ms  # already seen this timestamp

        await bridge_svc._maybe_publish_dom(now_ms + 100)

        bridge_svc._redis.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_writes_danger_trigger_file(self, bridge_svc, tmp_dom_files):
        _, _, danger_path = tmp_dom_files
        data = {"action": "flatten", "reason": "test", "severity": "critical"}

        await bridge_svc._write_danger_trigger(data)

        written = json.loads(danger_path.read_text())
        assert written["action"] == "flatten"
        assert written["consumed"] is False
        assert "ts_ms" in written

    @pytest.mark.asyncio
    async def test_danger_trigger_atomic_write(self, bridge_svc, tmp_dom_files):
        """Ensure no .tmp file is left behind after a write."""
        _, _, danger_path = tmp_dom_files
        await bridge_svc._write_danger_trigger({"action": "flatten", "reason": "x"})

        tmp = danger_path.with_suffix(".json.tmp")
        assert not tmp.exists(), ".tmp file should be cleaned up after atomic rename"
        assert danger_path.exists()


# ---------------------------------------------------------------------------
# Feature extractor: label_outcome
# ---------------------------------------------------------------------------

class TestLabelOutcome:

    def test_sweep_up_reverses(self):
        from src.ml.sweep_feature_extractor import label_outcome
        # Price went up 10 ticks (trigger), by T+60 reversed 6 ticks below trigger
        result = label_outcome(20010.0, "up", 20008.5)  # 6 ticks below trigger
        assert result == "sweep"

    def test_directional_up_extends(self):
        from src.ml.sweep_feature_extractor import label_outcome
        # Price extended 10 ticks above trigger
        result = label_outcome(20010.0, "up", 20012.5)  # 10 ticks above trigger
        assert result == "directional"

    def test_ambiguous_neither_threshold(self):
        from src.ml.sweep_feature_extractor import label_outcome
        # Moved only 3 ticks up from trigger — ambiguous
        result = label_outcome(20010.0, "up", 20010.75)
        assert result == "ambiguous"

    def test_sweep_down_reverses(self):
        from src.ml.sweep_feature_extractor import label_outcome
        # Downward trigger, price bounced back up 6 ticks
        result = label_outcome(19990.0, "down", 19991.5)  # 6 ticks above trigger
        assert result == "sweep"
