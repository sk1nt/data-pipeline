import time
import pytest

from src.lib.retries import retry_with_backoff, TransientError


def test_retry_with_backoff_succeeds_after_retry(monkeypatch):
    calls = {"count": 0}

    class FakeTransient(Exception):
        pass

    # Simulate function that fails once then succeeds
    def flaky():
        calls["count"] += 1
        if calls["count"] == 1:
            raise TransientError("timed out")
        return "ok"

    wrapped = retry_with_backoff(max_retries=3, initial_backoff=0.01)(flaky)
    assert wrapped() == "ok"
    assert calls["count"] == 2


def test_retry_with_backoff_raises_on_non_transient(monkeypatch):
    def fail_permanent():
        raise ValueError("bad request")

    wrapped = retry_with_backoff(max_retries=2, initial_backoff=0.01)(fail_permanent)
    with pytest.raises(ValueError):
        wrapped()
