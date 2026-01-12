#!/usr/bin/env python3
"""Test script to trigger market aggregation messages through Redis."""

import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lib.redis_client import RedisClient
from services.market_agg_alert_service import MarketAggAlertService


def test_scheduled_update():
    """Send a test scheduled update message."""
    print("Sending test scheduled update...")
    
    redis_client = RedisClient()
    alert_service = MarketAggAlertService(redis_client)
    
    # Sample market data
    test_data = {
        "date": "2026-01-09",
        "call_premium": "12345678.90",
        "put_premium": "9876543.21",
        "call_volume": 500000,
        "put_volume": 650000,
        "put_call_ratio": "1.30",
    }
    
    update = alert_service.create_scheduled_update(test_data)
    if update:
        message = alert_service.format_alert_message(update)
        print("\n--- SCHEDULED UPDATE MESSAGE ---")
        print(message)
        print("--- END MESSAGE ---\n")
        
        # Publish to Redis for Discord bot to pick up
        serialized = json.dumps(update, default=str)
        redis_client.client.publish("market_agg:alerts", serialized)
        print("✓ Published scheduled update to Redis")
        return True
    else:
        print("✗ Failed to create scheduled update")
        return False


def test_sentiment_change_alert():
    """Send a test sentiment change alert."""
    print("Sending test sentiment change alert...")
    
    redis_client = RedisClient()
    alert_service = MarketAggAlertService(redis_client)
    
    # Create alert payload
    alert = {
        "timestamp": "2026-01-09T14:30:00Z",
        "alert_type": "SENTIMENT_CHANGE",
        "current_ratio": 1.30,
        "previous_ratio": 0.75,
        "from_regime": "long",
        "to_regime": "short",
        "date": "2026-01-09",
        "call_premium": "12345678.90",
        "put_premium": "9876543.21",
        "call_volume": 500000,
        "put_volume": 650000,
        "discord_channels": [1425136266676146236, 1429940127899324487, 1440464526695731391],
    }
    
    message = alert_service.format_alert_message(alert)
    print("\n--- SENTIMENT CHANGE ALERT MESSAGE ---")
    print(message)
    print("--- END MESSAGE ---\n")
    
    # Publish to Redis for Discord bot to pick up
    serialized = json.dumps(alert, default=str)
    redis_client.client.publish("market_agg:alerts", serialized)
    print("✓ Published sentiment change alert to Redis")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Market Aggregation Message Tests")
    print("=" * 60)
    print()
    
    try:
        # Test scheduled update
        success1 = test_scheduled_update()
        print()
        
        # Test sentiment change alert
        success2 = test_sentiment_change_alert()
        print()
        
        if success1 and success2:
            print("=" * 60)
            print("✓ All tests completed successfully!")
            print("Check Discord channels for messages.")
            print("=" * 60)
            return 0
        else:
            print("=" * 60)
            print("✗ Some tests failed")
            print("=" * 60)
            return 1
    except Exception as exc:
        print(f"\n✗ Error: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
