import time
import sys
from pathlib import Path
from threading import Lock
import pytest

# Ensure project root appears on sys.path so we can import top-level `src` modules
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.services.schwab_streamer import SchwabAuthClient


class CounterDummy:
    def __init__(self):
        self.calls = 0

    def refresh_token(self):
        self.calls += 1


class QuickSchwabDummy:
    """Simple wrapper to expose `schwab` interface for tests"""
    def __init__(self, dummy):
        self._dummy = dummy
        self.tokens = {"access_token": "x", "refresh_token": "r", "expires_in": 3600}
        self.calls = 0

    def refresh_token(self):
        # delegate to the dummy refresh to count calls if it implements it
        try:
            self._dummy.refresh_token()
        except Exception:
            pass
        # emulate token rotation
        self.calls += 1
        # If the wrapped dummy exposed tokens, prefer that to emulate real client
        if hasattr(self._dummy, "tokens"):
            self.tokens = self._dummy.tokens
        else:
            self.tokens["access_token"] = f"x{self.calls}"
        if self.calls % 3 == 0:
            # change refresh token occasionally to emulate rotation
            self.tokens["refresh_token"] = f"r{self.calls}"

    @property
    def access_token(self):
        return self.tokens["access_token"]



def test_auth_auto_refresh_background_loop():
    dummy = CounterDummy()
    quick = QuickSchwabDummy(dummy)
    client = SchwabAuthClient(
        client_id="test",
        client_secret="secret",
        refresh_token="r",
        rest_url="https://api.test",
        schwab_client=quick,
        access_refresh_interval_seconds=0.05,
        refresh_token_rotate_interval_seconds=0.15,
    )
    # Ensure not running
    assert dummy.calls == 0
    # Start auto-refresh; this creates a thread that will run our quick loop
    client.start_auto_refresh()
    # Wait for some activity
    # loop will run up to 5 times at 0.05s each
    time.sleep(0.3)
    client.stop_auto_refresh()
    assert dummy.calls >= 1


def test_refresh_token_rotates_scheduled():
    dummy = CounterDummy()
    quick = QuickSchwabDummy(dummy)
    client = SchwabAuthClient(
        client_id="test",
        client_secret="secret",
        refresh_token="r",
        rest_url="https://api.test",
        schwab_client=quick,
        access_refresh_interval_seconds=0.05,
        refresh_token_rotate_interval_seconds=0.12,
    )
    initial_rt = quick.tokens["refresh_token"]
    client.start_auto_refresh()
    time.sleep(0.5)
    client.stop_auto_refresh()
    assert quick.tokens["refresh_token"] != initial_rt


def test_auth_manual_refresh_updates_tokens(monkeypatch):
    # Create a dummy with tokens store and methods as expected
    class ManualDummy:
        def __init__(self):
            self.tokens = {'access_token': 'a', 'refresh_token': 'r', 'expires_in': 3600}
            self.refresh_called = False

        def refresh_token(self):
            self.tokens['access_token'] = 'b'
            self.refresh_called = True

        @property
        def access_token(self):
            return self.tokens['access_token']

    dummy = ManualDummy()
    quick = QuickSchwabDummy(dummy)
    client = SchwabAuthClient(
        client_id="test",
        client_secret="secret",
        refresh_token="r",
        rest_url="https://api.test",
        schwab_client=quick,
    )
    # Simulate manual refresh
    tokens = client.refresh_tokens()
    # We expect the tokens to reflect the manual refresh
    assert dummy.refresh_called is True
    # The SchwabAuthClient.refresh_tokens() returns tokens dict
    assert tokens is not None
    assert tokens.get('access_token') == 'b'
