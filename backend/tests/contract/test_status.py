from fastapi.testclient import TestClient
from backend.src.api.main import app

client = TestClient(app)


def test_status_contract():
    """Contract test for /status endpoint."""
    response = client.get("/api/v1/status")
    assert response.status_code == 200

    data = response.json()
    assert "services" in data
    assert "timestamp" in data
    assert isinstance(data["services"], list)

    if data["services"]:
        service = data["services"][0]
        assert "service_name" in service
        assert "current_status" in service
        assert "last_update_time" in service
