import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from services.tastytrade_auth_service import TastytradeTransientAuthError  # noqa: E402
from services.tastytrade_client import TastyTradeClient, TastytradeAuthError  # noqa: E402


class FakeAuthService:
    def __init__(self):
        self.session = object()
        self.calls = 0
        self.session_expiration = None

    def get_session(self):
        self.calls += 1
        return self.session

    def refresh_session(self, force=False):
        self.calls += 1
        return self.session


def test_get_session_uses_shared_auth_service_session():
    client = TastyTradeClient()
    fake_auth = FakeAuthService()
    client._auth_service = fake_auth

    assert client.get_session() is fake_auth.session
    assert client.get_session() is fake_auth.session
    assert fake_auth.calls == 2


def test_transient_auth_is_normalized():
    class FailingAuth(FakeAuthService):
        def get_session(self):
            raise TastytradeTransientAuthError("timeout")

    client = TastyTradeClient()
    client._auth_service = FailingAuth()

    with pytest.raises(TastytradeAuthError):
        client.get_session()
