import os
import sys
import json
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from src.api.app import app
from src.services.automated_options_service import AutomatedOptionsService


def test_admin_process_alert(monkeypatch):
    client = TestClient(app)
    # Monkeypatch AutomatedOptionsService.process_alert to return a known value
    async def fake_process_alert(self, message, channel_id, user_id):
        return {"order_id": "abc123", "quantity": 1, "entry_price": 0.75}

    # Monkeypatch the same module path used by the admin router (services.*)
    monkeypatch.setattr("services.automated_options_service.AutomatedOptionsService.process_alert", fake_process_alert)
    # Ensure AuthService allows this user/channel (path used by services modules)
    monkeypatch.setattr("services.auth_service.AuthService.verify_user_and_channel_for_automated_trades", lambda u, c: True)

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
    assert data.get("result")["order_id"] == "abc123"


def test_admin_get_recent_audit(monkeypatch):
    client = TestClient(app)
    # Monkeypatch Redis client to return fake entries
    class FakeRedis:
        def lrange(self, key, start, end):
            return [json.dumps({"order_id": "abc123"})]

    def get_fake_redis():
        return type("W", (), {"client": FakeRedis()})()

    monkeypatch.setattr("src.api.routes.admin.get_redis_client", lambda: get_fake_redis())
    monkeypatch.setattr("services.auth_service.AuthService.verify_user_and_channel_for_automated_trades", lambda u, c: True)
    headers = {"X-ADMIN-KEY": os.getenv("ADMIN_API_KEY", "")}
    resp = client.get("/admin/audit/recent", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data.get("events"), list)
    assert data.get("events")[0]["order_id"] == "abc123"
