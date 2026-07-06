from __future__ import annotations

import pytest

from src.services.schwab_streamer import SchwabAuthClient, SchwabStreamClient


class _DummyStream:
    def __init__(self, *, fail_on_start: Exception | None = None) -> None:
        self.fail_on_start = fail_on_start
        self.started = False
        self.stopped = False

    def start(self, symbols=None, on_tick=None, on_level2=None):
        self.started = True
        if self.fail_on_start is not None:
            raise self.fail_on_start

    def stop(self):
        self.stopped = True


class _DummyAuthClient:
    def __init__(self, stream: _DummyStream) -> None:
        self._stream = stream
        self.start_calls = 0
        self.stop_calls = 0

    def stream(self):
        return self._stream

    def start_auto_refresh(self):
        self.start_calls += 1

    def stop_auto_refresh(self):
        self.stop_calls += 1


class _DummySchwabClient:
    def __init__(self, *, fail_with: Exception | None = None) -> None:
        self.fail_with = fail_with
        self.tokens = {}

    def refresh_token(self):
        if self.fail_with is not None:
            raise self.fail_with
        self.tokens = {"refresh_token": "new-token", "access_token": "access"}


class _DummyPublisher:
    def publish_tick(self, event):
        pass

    def publish_level2(self, event):
        pass


def test_refresh_tokens_marks_reauth_on_invalid_grant():
    auth_client = SchwabAuthClient(
        client_id="client",
        client_secret="secret",
        refresh_token="refresh",
        rest_url="https://example.invalid",
        interactive=False,
        schwab_client=_DummySchwabClient(fail_with=RuntimeError("invalid_grant")),
    )

    assert auth_client.refresh_tokens() is None
    status = auth_client.status()
    assert status["needs_reauth"] is True
    assert "invalid_grant" in (status["last_error"] or "")


def test_stream_start_stops_auto_refresh_on_login_failure():
    stream = _DummyStream(fail_on_start=RuntimeError("invalid_grant"))
    auth_client = _DummyAuthClient(stream)
    client = SchwabStreamClient(
        auth_client=auth_client,
        publisher=_DummyPublisher(),
        stream_url="wss://example.invalid",
        symbols=["SPY"],
    )

    with pytest.raises(RuntimeError):
        client.start()

    assert auth_client.start_calls == 1
    assert auth_client.stop_calls == 1
    assert stream.started is True
