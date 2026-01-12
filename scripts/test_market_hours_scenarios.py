#!/usr/bin/env python3
"""Test market hours checking in bot operations."""

import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lib.market_hours import (
    is_rth,
    is_weekend,
    is_market_holiday,
    is_market_open,
    get_next_market_open,
)


def test_scenarios():
    """Test various market hours scenarios."""
    et_tz = ZoneInfo("America/New_York")
    
    print("=" * 70)
    print("Market Hours Testing Scenarios")
    print("=" * 70)
    print()
    
    # Test different times
    test_cases = [
        # (description, datetime)
        ("Friday 10:30 AM (during RTH)", datetime(2026, 1, 9, 10, 30, tzinfo=et_tz)),
        ("Friday 8:00 AM (before open)", datetime(2026, 1, 9, 8, 0, tzinfo=et_tz)),
        ("Friday 5:00 PM (after close)", datetime(2026, 1, 9, 17, 0, tzinfo=et_tz)),
        ("Saturday 12:00 PM (weekend)", datetime(2026, 1, 10, 12, 0, tzinfo=et_tz)),
        ("Sunday 12:00 PM (weekend)", datetime(2026, 1, 11, 12, 0, tzinfo=et_tz)),
        ("Thursday Jan 1 12:00 PM (New Year's)", datetime(2026, 1, 1, 12, 0, tzinfo=et_tz)),
        ("Friday Dec 25 12:00 PM (Christmas)", datetime(2026, 12, 25, 12, 0, tzinfo=et_tz)),
    ]
    
    for desc, dt in test_cases:
        print(f"Scenario: {desc}")
        print(f"  Date/Time: {dt.strftime('%Y-%m-%d %I:%M %p %Z')}")
        print(f"  Is Weekend: {is_weekend(dt)}")
        print(f"  Is Holiday: {is_market_holiday(dt)}")
        print(f"  Is RTH: {is_rth(dt)}")
        print(f"  Is Market Open: {is_market_open(dt)}")
        
        if not is_market_open(dt):
            next_open = get_next_market_open(dt)
            print(f"  Next Market Open: {next_open.strftime('%Y-%m-%d %I:%M %p %Z')}")
        
        # Determine if GEX feed should send
        should_send_gex = is_rth(dt)
        print(f"  ✓ GEX Feed Should Send: {'YES' if should_send_gex else 'NO'}")
        
        # Determine if market agg scheduled update should send
        # (Only during RTH and at specific times: 10:00, 10:30, 11:00, etc.)
        should_send_market_agg = is_rth(dt) and dt.hour >= 10 and dt.minute in (0, 30)
        print(f"  ✓ Market Agg Should Send: {'YES' if should_send_market_agg else 'NO'}")
        print()
    
    print("=" * 70)
    print("Current Time Check")
    print("=" * 70)
    now = datetime.now(et_tz)
    print(f"Current Time (ET): {now.strftime('%Y-%m-%d %I:%M:%S %p %Z')}")
    print(f"Is Weekend: {is_weekend(now)}")
    print(f"Is Holiday: {is_market_holiday(now)}")
    print(f"Is RTH: {is_rth(now)}")
    print(f"Is Market Open: {is_market_open(now)}")
    
    if not is_market_open(now):
        next_open = get_next_market_open(now)
        print(f"Next Market Open: {next_open.strftime('%Y-%m-%d %I:%M %p %Z')}")
    
    print()
    print("=" * 70)
    print("Holiday Calendar for 2026")
    print("=" * 70)
    
    from lib.market_hours import US_MARKET_HOLIDAYS_2026
    print("US Market Holidays in 2026:")
    for holiday in sorted(US_MARKET_HOLIDAYS_2026):
        dt = datetime.strptime(holiday, "%Y-%m-%d")
        print(f"  • {dt.strftime('%A, %B %d, %Y')}")
    
    print()
    print("✓ All checks completed successfully!")


if __name__ == "__main__":
    test_scenarios()
