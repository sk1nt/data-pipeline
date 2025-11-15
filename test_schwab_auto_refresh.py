#!/usr/bin/env python3
"""Test the Schwab streamer auto-refresh functionality."""

import time
import threading
from src.services.schwab_streamer import SchwabAuthClient

# Use settings or mock for testing
from src.config import settings

def test_auto_refresh():
    if not settings.schwab_client_id or not settings.schwab_client_secret or not settings.schwab_refresh_token:
        print("Schwab credentials not configured in settings")
        return

    auth_client = SchwabAuthClient(
        client_id=settings.schwab_client_id,
        client_secret=settings.schwab_client_secret,
        refresh_token=settings.schwab_refresh_token,
        rest_url=settings.schwab_rest_url,
    )

    print("Starting auto-refresh...")
    auth_client.start_auto_refresh()

    # Let it run for a short time
    time.sleep(10)  # Run for 10 seconds

    print("Stopping auto-refresh...")
    auth_client.stop_auto_refresh()

    print("Test completed. Check logs for refresh activity.")

if __name__ == "__main__":
    test_auto_refresh()