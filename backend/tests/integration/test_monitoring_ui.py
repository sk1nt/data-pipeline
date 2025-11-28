from fastapi.testclient import TestClient
from backend.src.api.main import app

client = TestClient(app)


def test_monitoring_ui_integration():
    """Integration test for monitoring UI user journey."""
    # Test API endpoint that UI would use
    response = client.get("/api/v1/status")
    assert response.status_code == 200

    data = response.json()
    assert len(data["services"]) > 0

    # Test that services have expected fields
    for service in data["services"]:
        assert service["current_status"] in ["healthy", "degraded", "down"]
        assert "uptime_percentage" in service
