import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from services.automated_options_service import (  # noqa: E402
    AutomatedOptionsService,
    PENDING_AUTH_QUEUE,
)
from services.tastytrade_client import TastytradeAuthError  # noqa: E402


class FakeRedis:
    def __init__(self):
        self.lists = {}
        self.sets = {}

    def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)

    def rpop(self, key):
        values = self.lists.get(key) or []
        return values.pop() if values else None

    def sismember(self, key, value):
        return value in self.sets.get(key, set())

    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value)

    def expire(self, key, ttl):
        return True


class TransientAuthClient:
    def ensure_authorized(self):
        raise TastytradeAuthError("TastyTrade authorization temporarily unavailable")


class HardAuthClient:
    def ensure_authorized(self):
        raise TastytradeAuthError("invalid_grant")


@pytest.mark.asyncio
async def test_transient_auth_queues_valid_alert(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(
        "services.automated_options_service.get_redis_client",
        lambda: SimpleNamespace(client=fake_redis),
    )

    svc = AutomatedOptionsService(tastytrade_client=TransientAuthClient())
    result = await svc.process_alert(
        "Alert: BTO UBER 78p 12/05 @ 0.75",
        "1255265167113978008",
        "704125082750156840",
    )

    assert result["status"] == "queued_auth_refresh"
    assert fake_redis.lists[PENDING_AUTH_QUEUE]
    queued = json.loads(fake_redis.lists[PENDING_AUTH_QUEUE][0])
    assert queued["message"].startswith("Alert: BTO")
    assert queued["idempotency_key"] == result["idempotency_key"]


@pytest.mark.asyncio
async def test_invalid_grant_does_not_queue(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(
        "services.automated_options_service.get_redis_client",
        lambda: SimpleNamespace(client=fake_redis),
    )

    svc = AutomatedOptionsService(tastytrade_client=HardAuthClient())

    with pytest.raises(TastytradeAuthError):
        await svc.process_alert(
            "Alert: BTO UBER 78p 12/05 @ 0.75",
            "1255265167113978008",
            "704125082750156840",
        )

    assert fake_redis.lists.get(PENDING_AUTH_QUEUE) is None
