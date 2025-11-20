import importlib.util
import json
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "data-pipeline.py"

spec = importlib.util.spec_from_file_location("pipeline_app", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_client(monkeypatch):
    monkeypatch.setattr(module.service_manager, "start", lambda: None)

    async def _noop():
        return None

    monkeypatch.setattr(module.service_manager, "stop", _noop)
    return TestClient(module.app)


def test_status_endpoint(monkeypatch):
    expected = {"tastytrade_streamer": {"running": True}}
    monkeypatch.setattr(module.service_manager, "status", lambda: expected)
    with make_client(monkeypatch) as client:
        response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == expected


def test_status_page(monkeypatch):
    monkeypatch.setattr(module.service_manager, "status", lambda: {"ok": True})
    with make_client(monkeypatch) as client:
        response = client.get("/status.html")
    assert response.status_code == 200
    assert "Data Pipeline Status" in response.text


def test_control_requires_token(monkeypatch):
    monkeypatch.setattr(module.settings, "service_control_token", "secret")
    with make_client(monkeypatch) as client:
        response = client.post("/control/tastytrade/restart")
    assert response.status_code == 403


def test_control_restart(monkeypatch):
    monkeypatch.setattr(module.settings, "service_control_token", "secret")

    async def fake_restart(name):
        fake_restart.called = name

    fake_restart.called = None
    monkeypatch.setattr(module.service_manager, "restart_service", fake_restart)

    with make_client(monkeypatch) as client:
        response = client.post(
            "/control/tastytrade/restart",
            headers={"X-Service-Token": "secret"},
        )
    assert response.status_code == 200
    assert fake_restart.called == "tastytrade"


def test_ml_trade_endpoint_persists_and_publishes(monkeypatch):
    fake_redis = _FakeRedis()
    monkeypatch.setattr(module, "_get_redis_client", lambda: fake_redis)
    payload = {
        "symbol": "mnq",
        "action": "entry",
        "direction": "long",
        "price": 165.25,
        "confidence": 0.785,
        "position_before": 0,
        "position_after": 1,
        "pnl": 0.0,
        "total_pnl": 0.0,
        "total_trades": 1,
        "timestamp": "2025-11-19T14:30:22.123456+00:00",
        "simulated": True,
    }
    with make_client(monkeypatch) as client:
        response = client.post("/ml-trade", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "MNQ"
    lpush_calls = [cmd for cmd in fake_redis.pipeline_cmds if cmd[0] == "lpush"]
    assert lpush_calls, "expected lpush to be invoked"
    stored_payload = json.loads(lpush_calls[0][2])
    assert stored_payload["symbol"] == "MNQ"
    assert fake_redis.published
    channel, message = fake_redis.published[0]
    assert channel == module.ML_TRADE_STREAM_CHANNEL
    decoded = json.loads(message)
    assert decoded["action"] == "entry"


class _FakePipeline:
    def __init__(self, recorder):
        self.recorder = recorder

    def set(self, *args):
        self.recorder.pipeline_cmds.append(("set", *args))
        return self

    def lpush(self, *args):
        self.recorder.pipeline_cmds.append(("lpush", *args))
        return self

    def ltrim(self, *args):
        self.recorder.pipeline_cmds.append(("ltrim", *args))
        return self

    def execute(self):
        self.recorder.pipeline_executed = True


class _FakeRedis:
    def __init__(self):
        self.pipeline_cmds = []
        self.pipeline_executed = False
        self.published = []

    def pipeline(self):
        return _FakePipeline(self)

    def publish(self, channel, message):
        self.published.append((channel, message))
