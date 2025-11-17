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


class QuickAuthClient(SchwabAuthClient):
    def __init__(self, dummy):
        # Avoid calling parent __init__ which expects real schwab client and file IO
        # Initialize minimal fields required by our tests.
        self.client_id = 'test'
        self.client_secret = 'secret'
        self.refresh_token = 'r'
        self.rest_url = 'https://api.test'
        import threading
        self._stop_refresh = threading.Event()
        self._refresh_thread = None
        self._lock = Lock()
        self.schwab = dummy

    def _auto_refresh_loop(self):
        # Run a short-lived quick loop that will call refresh_token a few times.
        self._stop_refresh = self._stop_refresh or __import__('threading').Event()
        count = 0
        while not self._stop_refresh.is_set() and count < 5:
            with self._lock:
                self.schwab.refresh_token()
            count += 1
            time.sleep(0.05)


def test_auth_auto_refresh_background_loop():
    dummy = CounterDummy()
    client = QuickAuthClient(dummy)
    # Ensure not running
    assert dummy.calls == 0
    # Start auto-refresh; this creates a thread that will run our quick loop
    client.start_auto_refresh()
    # Wait for some activity
    # loop will run up to 5 times at 0.05s each
    time.sleep(0.3)
    client.stop_auto_refresh()
    assert dummy.calls >= 1


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
    client = QuickAuthClient(dummy)
    # Simulate manual refresh
    tokens = client.refresh_tokens()
    # We expect the tokens to reflect the manual refresh
    assert dummy.refresh_called is True
    # The SchwabAuthClient.refresh_tokens() returns tokens dict
    assert tokens is not None
    assert tokens.get('access_token') == 'b'
