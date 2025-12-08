import os
import sys
import json
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from src.api.app import app


def test_admin_process_alert_e2e(monkeypatch):
    client = TestClient(app)
    # Ensure allowlist check passes and user allowed for alerts
    monkeypatch.setattr("services.auth_service.AuthService.verify_user_and_channel_for_automated_trades", lambda u, c: True)
    monkeypatch.setattr("services.auth_service.AuthService.verify_user_for_alerts", lambda u: True)
    # Ensure tastytrade_client does not raise auth error by replacing with a stub
    class StubClient:
        def ensure_authorized(self):
            return True

        def get_session(self):
            return object()

    monkeypatch.setattr("services.tastytrade_client.tastytrade_client", StubClient())

    # Fake fill service to simulate entry filled and exit placed
    async def fake_fill_options_order(self, symbol, strike, option_type, expiry, quantity, action, user_id, channel_id, initial_price=None):
        return {"order_id": "entry123", "entry_price": 0.75}

    monkeypatch.setattr("services.automated_options_service.OptionsFillService.fill_options_order", fake_fill_options_order)

    # Fake redis lpush to capture audit logs
    captured = {}
    class FakeRedis:
        def lpush(self, key, value):
            captured["last"] = {"key": key, "value": value}

    def fake_get_redis():
        return type("W", (), {"client": FakeRedis()})()

    # Patch both admin route and services to use fake Redis wrapper
    monkeypatch.setattr("src.api.routes.admin.get_redis_client", lambda: fake_get_redis())
    monkeypatch.setattr("lib.redis_client.get_redis_client", lambda: fake_get_redis())
    monkeypatch.setattr("services.automated_options_service.get_redis_client", lambda: fake_get_redis())

    payload = {
        "message": "Alert: BTO UBER 78p 12/05 @ 0.75",
        "channel_id": "123",
        "user_id": "456",
        "dry_run": True,
    }
    headers = {"X-ADMIN-KEY": os.getenv("ADMIN_API_KEY", "")}
    resp = client.post("/admin/alerts/process", json=payload, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"
    assert data.get("result")["order_id"] == "entry123"
    # Validate audit log present
    assert captured.get("last") is not None
    audit = json.loads(captured["last"]["value"])
    assert audit.get("order_id") == "entry123"
