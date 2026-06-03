"""tests/unit/test_danger_levels.py — unit tests for PositionMonitorService logic.

Tests the escalation rules without needing Redis or real positions.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.position_monitor_service import (
    PositionMonitorService,
    PositionState,
    label_outcome as _noop,  # just checking import
)


# ---------------------------------------------------------------------------
# PositionState tests
# ---------------------------------------------------------------------------

class TestPositionState:

    def test_long_losing_when_price_falls(self):
        pos = PositionState()
        pos.quantity    = 2
        pos.entry_price = 20100.0
        # Price dropped 20 ticks (= 5 points for MNQ)
        loss = pos.unrealized_loss_ticks(20095.0)   # 5 point drop → 20 ticks
        assert loss == pytest.approx(20.0)

    def test_short_losing_when_price_rises(self):
        pos = PositionState()
        pos.quantity    = -1
        pos.entry_price = 20000.0
        # Price rose 10 ticks (= 2.5 points for MNQ)
        loss = pos.unrealized_loss_ticks(20002.5)
        assert loss == pytest.approx(10.0)

    def test_flat_position_no_loss(self):
        pos = PositionState()
        assert pos.unrealized_loss_ticks(20000.0) == 0.0

    def test_long_in_wrong_direction_when_move_is_down(self):
        pos = PositionState()
        pos.quantity = 1
        assert pos.in_wrong_direction("down") is True
        assert pos.in_wrong_direction("up")   is False

    def test_short_in_wrong_direction_when_move_is_up(self):
        pos = PositionState()
        pos.quantity = -2
        assert pos.in_wrong_direction("up")   is True
        assert pos.in_wrong_direction("down") is False

    def test_flat_never_in_wrong_direction(self):
        pos = PositionState()
        assert pos.in_wrong_direction("up")   is False
        assert pos.in_wrong_direction("down") is False


# ---------------------------------------------------------------------------
# PositionMonitorService escalation tests
# ---------------------------------------------------------------------------

def _make_directional_alert(confidence=0.80, direction="down", trigger_price=20000.0):
    return {
        "classification": "directional",
        "confidence":     confidence,
        "direction":      direction,
        "trigger_price":  trigger_price,
        "gex_regime":     "negative",
        "at_wall":        False,
        "through_wall":   True,
    }


def _make_long_position(quantity=2, entry_price=20010.0) -> PositionState:
    pos = PositionState()
    pos.quantity    = quantity
    pos.entry_price = entry_price
    pos.source      = "test"
    return pos


@pytest.fixture
def monitor():
    svc = PositionMonitorService.__new__(PositionMonitorService)
    svc._redis           = AsyncMock()
    svc._symbol          = "MNQ"
    svc._running         = True
    svc._level2_fired_at = None
    svc._suppress_until  = 0.0
    svc._ack_received    = False
    svc._current_level   = 0
    return svc


class TestMonitorEscalation:

    @pytest.mark.asyncio
    async def test_sweep_alert_resets_level(self, monitor):
        monitor._current_level = 2
        alert = {"classification": "sweep", "confidence": 0.80,
                 "direction": "up", "trigger_price": 20000.0,
                 "gex_regime": "positive", "at_wall": True, "through_wall": False}
        with patch("src.services.position_monitor_service.get_current_position",
                   return_value=_make_long_position()):
            await monitor._handle_alert(alert)
        assert monitor._current_level == 0

    @pytest.mark.asyncio
    async def test_no_position_no_escalation(self, monitor):
        alert = _make_directional_alert(confidence=0.85, direction="down",
                                        trigger_price=19990.0)
        pos = PositionState()  # flat
        with patch("src.services.position_monitor_service.get_current_position",
                   return_value=pos):
            await monitor._handle_alert(alert)
        assert monitor._current_level == 0

    @pytest.mark.asyncio
    async def test_position_in_right_direction_no_escalation(self, monitor):
        """Short position + downward directional = we're WINNING, no escalation."""
        alert = _make_directional_alert(confidence=0.85, direction="down",
                                        trigger_price=19990.0)
        pos = PositionState()
        pos.quantity    = -1   # short
        pos.entry_price = 20050.0
        with patch("src.services.position_monitor_service.get_current_position",
                   return_value=pos):
            await monitor._handle_alert(alert)
        assert monitor._current_level == 0

    @pytest.mark.asyncio
    async def test_escalates_to_level_2_on_high_conf_and_loss(self, monitor):
        # Long position, price dropping — losing 30 ticks (= 7.5 points)
        alert = _make_directional_alert(confidence=0.80, direction="down",
                                        trigger_price=19992.5)
        pos = _make_long_position(entry_price=20000.0)  # 30 ticks down
        with patch("src.services.position_monitor_service.get_current_position",
                   return_value=pos):
            await monitor._handle_alert(alert)
        assert monitor._current_level == 2

    @pytest.mark.asyncio
    async def test_level3_requires_l2_timeout(self, monitor):
        """Level 3 must NOT fire immediately; requires L2 to have been unacked for ACK_TIMEOUT_SECONDS."""
        alert = _make_directional_alert(confidence=0.90, direction="down",
                                        trigger_price=19991.25)   # 35 ticks down from 20000
        pos = _make_long_position(entry_price=20000.0)
        with patch("src.services.position_monitor_service.get_current_position",
                   return_value=pos):
            # First call: should raise to L2, set timer
            await monitor._handle_alert(alert)
        assert monitor._current_level == 2
        assert monitor._level2_fired_at is not None
        # Second call immediately — L2 timer not expired, should NOT go to L3
        with patch("src.services.position_monitor_service.get_current_position",
                   return_value=pos):
            await monitor._handle_alert(alert)
        assert monitor._current_level == 2  # still 2, not yet 3

    @pytest.mark.asyncio
    async def test_level3_fires_after_timeout_in_dry_run(self, monitor):
        """After ACK_TIMEOUT_SECONDS with no ack, level 3 fires (dry run — no Redis publish)."""
        alert = _make_directional_alert(confidence=0.90, direction="down",
                                        trigger_price=19991.25)   # 35 ticks
        pos = _make_long_position(entry_price=20000.0)
        monitor._level2_fired_at = time.time() - 15  # 15s ago (> ACK_TIMEOUT_SECONDS=10)
        monitor._current_level   = 2
        # Patch dynamic thresholds to return static base values regardless of TOD
        from src.services import position_monitor_service as pms
        static = (pms.WARNING_CONFIDENCE, pms.DANGER_CONFIDENCE, pms.CRITICAL_CONFIDENCE,
                  pms.WARNING_TICKS, pms.DANGER_TICKS, pms.CRITICAL_TICKS)
        with patch("src.services.position_monitor_service.SWEEP_LIVE_MODE", False), \
             patch("src.services.position_monitor_service._dynamic_thresholds",
                   return_value=static), \
             patch("src.services.position_monitor_service.get_current_position",
                   return_value=pos):
            await monitor._handle_alert(alert)
        assert monitor._current_level == 3
        # In dry run mode Redis publish should NOT be called for the danger channel
        for call in monitor._redis.publish.call_args_list:
            assert "sweep:danger" not in str(call), \
                "Danger flatten must NOT publish in dry-run mode"

    @pytest.mark.asyncio
    async def test_suppress_prevents_escalation(self, monitor):
        monitor._suppress_until = time.time() + 300  # suppressed for 5 min
        alert = _make_directional_alert(confidence=0.90, direction="down",
                                        trigger_price=19991.25)
        pos = _make_long_position(entry_price=20000.0)
        with patch("src.services.position_monitor_service.get_current_position",
                   return_value=pos):
            await monitor._handle_alert(alert)
        assert monitor._current_level == 0
