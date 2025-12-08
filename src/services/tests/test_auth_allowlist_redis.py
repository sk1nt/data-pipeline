import os
import sys
from types import SimpleNamespace
import pytest
import os

sys.path.insert(0, os.path.join(os.getcwd()))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from services.auth_service import AuthService


class FakeRedis:
    def __init__(self):
        self.sets = {}

    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value)

    def srem(self, key, value):
        self.sets.setdefault(key, set()).discard(value)

    def smembers(self, key):
        return self.sets.get(key, set())

    def sismember(self, key, value):
        return value in self.sets.get(key, set())


class FakeWrapper:
    def __init__(self, client):
        self.client = client


def test_auth_service_redis_allowlist(monkeypatch):
    fake_redis = FakeRedis()
    wrapper = FakeWrapper(fake_redis)
    monkeypatch.setattr("lib.redis_client.get_redis_client", lambda: wrapper)

    # Ensure user not present initially
    uid = "999"
    assert not AuthService.verify_user_for_alerts(uid)

    assert AuthService.add_user_to_allowlist(uid)
    assert AuthService.verify_user_for_alerts(uid)

    # Remove and verify
    assert AuthService.remove_user_from_allowlist(uid)
    assert not AuthService.verify_user_for_alerts(uid)

    # Channels
    cid = "321"
    assert not AuthService.verify_channel_for_automated_trades(cid)
    assert AuthService.add_channel_to_allowlist(cid)
    assert AuthService.verify_channel_for_automated_trades(cid)
    assert AuthService.remove_channel_from_allowlist(cid)
    assert not AuthService.verify_channel_for_automated_trades(cid)
    # Ensure the new fallback works for allowed channels/users
    monkeypatch.setenv("DISCORD_AUTOMATED_TRADE_IDS", "1255265167113978008,970439141512871956")
    monkeypatch.setenv("ALERT_USERS", "700068629626224700,704125082750156840")
