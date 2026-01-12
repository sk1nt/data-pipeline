"""Tests for market hours utilities."""

from datetime import datetime, time
import pytest
from zoneinfo import ZoneInfo

from src.lib.market_hours import (
    is_rth,
    is_weekend,
    is_market_holiday,
    is_market_open,
    get_next_market_open,
    RTH_START,
    RTH_END,
)


def test_is_weekend():
    """Test weekend detection."""
    # Saturday
    saturday = datetime(2026, 1, 10, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    assert is_weekend(saturday) is True
    
    # Sunday
    sunday = datetime(2026, 1, 11, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    assert is_weekend(sunday) is True
    
    # Monday (not weekend)
    monday = datetime(2026, 1, 12, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    assert is_weekend(monday) is False
    
    # Friday (not weekend)
    friday = datetime(2026, 1, 9, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    assert is_weekend(friday) is False


def test_is_market_holiday():
    """Test holiday detection."""
    # New Year's Day 2026
    new_years = datetime(2026, 1, 1, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    assert is_market_holiday(new_years) is True
    
    # Christmas 2026
    christmas = datetime(2026, 12, 25, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    assert is_market_holiday(christmas) is True
    
    # Regular trading day
    regular_day = datetime(2026, 1, 9, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    assert is_market_holiday(regular_day) is False


def test_is_rth():
    """Test RTH (Regular Trading Hours) detection."""
    et_tz = ZoneInfo("America/New_York")
    
    # During RTH (10:30 AM on a Friday)
    during_rth = datetime(2026, 1, 9, 10, 30, tzinfo=et_tz)
    assert is_rth(during_rth) is True
    
    # Before market open (8:00 AM)
    before_open = datetime(2026, 1, 9, 8, 0, tzinfo=et_tz)
    assert is_rth(before_open) is False
    
    # After market close (5:00 PM)
    after_close = datetime(2026, 1, 9, 17, 0, tzinfo=et_tz)
    assert is_rth(after_close) is False
    
    # Weekend (even during normal hours)
    weekend_rth_time = datetime(2026, 1, 10, 12, 0, tzinfo=et_tz)
    assert is_rth(weekend_rth_time) is False
    
    # Holiday (even during normal hours)
    holiday_rth_time = datetime(2026, 1, 1, 12, 0, tzinfo=et_tz)
    assert is_rth(holiday_rth_time) is False
    
    # Right at market open (9:30 AM)
    market_open = datetime(2026, 1, 9, 9, 30, tzinfo=et_tz)
    assert is_rth(market_open) is True
    
    # Right before market close (3:59 PM)
    before_close = datetime(2026, 1, 9, 15, 59, tzinfo=et_tz)
    assert is_rth(before_close) is True
    
    # Right at market close (4:00 PM) - should be False
    at_close = datetime(2026, 1, 9, 16, 0, tzinfo=et_tz)
    assert is_rth(at_close) is False


def test_is_market_open():
    """Test market open detection."""
    et_tz = ZoneInfo("America/New_York")
    
    # During market hours
    during_hours = datetime(2026, 1, 9, 12, 0, tzinfo=et_tz)
    assert is_market_open(during_hours) is True
    
    # Weekend
    weekend = datetime(2026, 1, 10, 12, 0, tzinfo=et_tz)
    assert is_market_open(weekend) is False
    
    # Holiday
    holiday = datetime(2026, 1, 1, 12, 0, tzinfo=et_tz)
    assert is_market_open(holiday) is False
    
    # Before market open
    before_open = datetime(2026, 1, 9, 8, 0, tzinfo=et_tz)
    assert is_market_open(before_open) is False
    
    # Extended hours (with flag)
    premarket = datetime(2026, 1, 9, 7, 0, tzinfo=et_tz)
    assert is_market_open(premarket, include_extended=False) is False
    assert is_market_open(premarket, include_extended=True) is True


def test_get_next_market_open():
    """Test next market open calculation."""
    et_tz = ZoneInfo("America/New_York")
    
    # Friday afternoon -> Monday morning
    friday_afternoon = datetime(2026, 1, 9, 17, 0, tzinfo=et_tz)
    next_open = get_next_market_open(friday_afternoon)
    assert next_open.weekday() == 0  # Monday
    assert next_open.hour == 9
    assert next_open.minute == 30
    
    # Friday before market open -> same day
    friday_morning = datetime(2026, 1, 9, 8, 0, tzinfo=et_tz)
    next_open = get_next_market_open(friday_morning)
    assert next_open.date() == friday_morning.date()
    assert next_open.hour == 9
    assert next_open.minute == 30
    
    # Thursday before New Year's -> skip holiday
    # Dec 31, 2025 (Wednesday) after close -> Jan 2, 2026 (Friday) open
    # (Jan 1 is New Year's Day holiday)
    dec_31 = datetime(2025, 12, 31, 17, 0, tzinfo=et_tz)
    next_open = get_next_market_open(dec_31)
    assert next_open.date().day == 2  # Jan 2
    assert next_open.date().month == 1
    assert next_open.date().year == 2026


def test_rth_boundaries():
    """Test exact RTH boundary times."""
    et_tz = ZoneInfo("America/New_York")
    date = datetime(2026, 1, 9, tzinfo=et_tz)  # Friday
    
    # Just before market open
    before_open = date.replace(hour=9, minute=29, second=59)
    assert is_rth(before_open) is False
    
    # Exactly at market open
    at_open = date.replace(hour=9, minute=30, second=0)
    assert is_rth(at_open) is True
    
    # Just before market close
    before_close = date.replace(hour=15, minute=59, second=59)
    assert is_rth(before_close) is True
    
    # Exactly at market close
    at_close = date.replace(hour=16, minute=0, second=0)
    assert is_rth(at_close) is False
