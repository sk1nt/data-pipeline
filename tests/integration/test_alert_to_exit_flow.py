import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.getcwd()))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

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


class FakeFillService:
    def __init__(self):
        self.exit_placed = False

    async def fill_options_order(self, *_, **__):
        # Simulate an entry fill and exit placement
        self.exit_placed = True
        return {"order_id": "entry123", "entry_price": "1.00"}


@pytest.mark.asyncio
async def test_entry_fill_exit_flow(monkeypatch):
    fake_redis = FakeRedis()
    wrapper = FakeRedisWrapper(fake_redis)

    # Allow all users/channels for this flow
    monkeypatch.setattr(
        "services.auth_service.AuthService.verify_user_and_channel_for_automated_trades",
        lambda u, c: True,
    )

    # Ensure notifications are no-op for the test
    notify_calls = []
    monkeypatch.setattr(
        "services.notifications.notify_operator",
        lambda msg: notify_calls.append(msg) or True,
    )

    # Use fake redis for auditing
    monkeypatch.setattr(
        "src.services.automated_options_service.get_redis_client",
        lambda: wrapper,
    )

    svc = AutomatedOptionsService(tastytrade_client=FakeTastyClient())
    svc.fill_service = FakeFillService()

    # Keep sizing deterministic to avoid external balance calls
    monkeypatch.setattr(
        svc,
        "_compute_quantity",
        lambda alert, channel: (2, 10000.0, 200.0),
    )

    result = await svc.process_alert(
        "Alert: BTO UBER 78p 12/05 @ 1.00", "channel123", "user456"
    )

    assert result["order_id"] == "entry123"
    assert result["quantity"] == 2
    assert svc.fill_service.exit_placed is True

    # Audit log captured
    assert fake_redis.calls, "Expected audit lpush call"
    key, payload = fake_redis.calls[0]
    assert key == "audit:automated_alerts"
    payload_obj = json.loads(payload)
    assert payload_obj.get("order_id") == "entry123"
    assert payload_obj.get("computed_quantity") == 2
