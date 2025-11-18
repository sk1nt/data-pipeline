from datetime import datetime, timedelta, timezone

from bot.trade_bot import RollingWindowTracker


def test_tracker_retains_window_and_last_delta():
    tracker = RollingWindowTracker(window_seconds=60)
    base = datetime(2024, 1, 1, 14, 30, tzinfo=timezone.utc)

    snap_initial = tracker.update('spot', 100.0, base)
    assert snap_initial.delta == 0

    snap_next = tracker.update('spot', 101.0, base + timedelta(seconds=30))
    assert snap_next.delta == 1.0

    snap_window = tracker.update('spot', 102.0, base + timedelta(seconds=90))
    assert snap_window.delta == 1.0

    snap_missing = tracker.update('spot', None, base + timedelta(seconds=95))
    assert snap_missing.delta == 1.0
