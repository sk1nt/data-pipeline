import importlib.util
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
