from datetime import datetime, timedelta
import sys
from pathlib import Path

# Ensure project root appears on sys.path so we can import top-level `src` modules
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import threading
import time
from typing import List

import pytest

from src.services.schwab_streamer import (
    SchwabMessageParser,
    SchwabStreamClient,
)


class DummyStream:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self, symbols=None, on_tick=None, on_level2=None):
        self.started = True

    def stop(self):
        self.stopped = True


class DummyAuthClient:
    def __init__(self):
        self.started_refresh = False
        self.stopped_refresh = False
        self.tokens = {'access_token': 'x', 'refresh_token': 'r', 'expires_in': 3600}
        self._stream = DummyStream()
        # Expose `schwab` attribute like actual auth client
        self.schwab = self

    def start_auto_refresh(self):
        self.started_refresh = True

    def stop_auto_refresh(self):
        self.stopped_refresh = True

    def refresh_tokens(self):
        # pretend we refreshed tokens
        self.tokens['access_token'] = 'y'
        return self.tokens

    @property
    def access_token(self):
        return self.tokens['access_token']

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, v):
        self._tokens = v

    def stream(self):
        return self._stream


class DummyPublisher:
    def __init__(self):
        self.ticks = []
        self.level2 = []

    def publish_tick(self, tick):
        self.ticks.append(tick)

    def publish_level2(self, event):
        self.level2.append(event)


def test_parser_parses_tick_and_level2():
    parser = SchwabMessageParser()
    sample_tick = {"type": "TICK", "symbol": "MNQ", "timestamp": 1762837200000, "price": 100}
    events = parser.parse(sample_tick)
    assert any(kind == 'tick' for kind, _ in events)

    sample_level2 = {"type": "LEVEL2", "symbol": "MNQ", "timestamp": 1762837200000, "bids": [{"price": 100, "size": 1}], "asks": [{"price": 101, "size": 2}]}
    events = parser.parse(sample_level2)
    assert any(kind == 'level2' for kind, _ in events)


def test_streamer_start_stop_and_publish(monkeypatch):
    auth_client = DummyAuthClient()
    publisher = DummyPublisher()
    stream = auth_client.stream()

    # Build a SchwabStreamClient using DummyAuthClient
    client = SchwabStreamClient(
        auth_client=auth_client,
        publisher=publisher,
        stream_url='wss://example.com',
        symbols=['MNQ'],
        heartbeat_seconds=1,
    )

    # Replace the stream object with our DummyStream
    client.stream = stream

    client.start()
    assert auth_client.started_refresh is True
    assert stream.started is True

    # Simulate receiving a tick
    tick_payload = {"type": "TICK", "symbol": "MNQ", "timestamp": 1762837200000, "price": 100}
    client._on_tick(tick_payload)
    assert len(publisher.ticks) == 1

    # Simulate receiving a level2
    level2_payload = {"type": "LEVEL2", "symbol": "MNQ", "timestamp": 1762837200000, "bids": [{"price": 100, "size": 1}], "asks": [{"price": 101, "size": 2}]}
    client._on_level2(level2_payload)
    assert len(publisher.level2) == 1

    client.stop()
    assert auth_client.stopped_refresh is True
    assert stream.stopped is True
