import json
import os
import sys
import pytest
from types import SimpleNamespace

# Ensure 'src' is in sys.path when running tests in discord-bot directory
# Ensure project root and src dir are on sys.path so both `services` and `src.services` imports work
sys.path.insert(0, os.path.join(os.getcwd()))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

# Test the automated options service audit logging and returned structure
from src.services.automated_options_service import AutomatedOptionsService


class FakeRedis:
    def __init__(self):
        self.calls = []

    def lpush(self, key, value):
        self.calls.append((key, value))


class FakeRedisWrapper:
    def __init__(self, client):
        self.client = client


class FakeTastyClient:
    def ensure_authorized(self):
        return True


@pytest.mark.asyncio
async def test_process_alert_logs_to_redis_and_returns_struct(monkeypatch):
    fake_redis = FakeRedis()
    wrapper = FakeRedisWrapper(fake_redis)

    # Make get_redis_client return our fake wrapper
    monkeypatch.setattr(
        "src.services.automated_options_service.get_redis_client",
        lambda: wrapper,
    )

    # Fake fill service to return a known order result
    fake_fill_result = {"order_id": "abc123", "entry_price": "0.75"}

    async def fake_fill_options_order(*args, **kwargs):
        return fake_fill_result

    fake_fill_service = SimpleNamespace(fill_options_order=fake_fill_options_order)

    # Construct service with fake tastytrade client and inject fake fill service
    svc = AutomatedOptionsService(tastytrade_client=FakeTastyClient())
    svc.fill_service = fake_fill_service

    message = "Alert: BTO UBER 78p 12/05 @ 0.75"
    # Use a known allowed user ID from AuthService
    user_id = "704125082750156840"
    channel_id = "1255265167113978008"

    result = await svc.process_alert(message, channel_id, user_id)
    assert isinstance(result, dict)
    assert result.get("order_id") == "abc123"
    assert result.get("quantity") >= 1
    assert result.get("entry_price") == "0.75"

    # Check Redis audit entry was created
    assert len(fake_redis.calls) == 1
    key, payload = fake_redis.calls[0]
    assert key == "audit:automated_alerts"
    payload_obj = json.loads(payload)
    assert payload_obj.get("order_id") == "abc123"
    assert payload_obj.get("computed_quantity") == result.get("quantity")

