import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from src.services.tastytrade_client import post_with_retry, get_with_retry


def test_post_with_retry_succeeds_after_transient(monkeypatch):
    class FakeSession:
        def __init__(self):
            self.calls = 0

        def _post(self, url, data=None):
            self.calls += 1
            if self.calls == 1:
                raise Exception("502 Bad Gateway")
            return {"result": "ok"}

    sess = FakeSession()
    res = post_with_retry(sess, "/some/url", data="{}", max_retries=2, initial_backoff=0.01)
    assert res["result"] == "ok"


def test_get_with_retry_non_transient_raises(monkeypatch):
    class FakeSession:
        def _get(self, url):
            raise ValueError("400 Bad Request")

    sess = FakeSession()
    with pytest.raises(ValueError):
        get_with_retry(sess, "/bad/url", max_retries=2, initial_backoff=0.01)
