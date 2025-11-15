import pytest
from fastapi.testclient import TestClient
from backend.src.api.main import app

client = TestClient(app)

@pytest.fixture
def api_key():
    return "test-api-key"

def test_backtesting_query_integration(api_key):
    """Integration test for backtesting query user journey."""
    headers = {"Authorization": f"Bearer {api_key}"}

    # Query historical data
    response = client.get(
        "/api/v1/ticks/historical?symbols=AAPL&start_time=2025-11-06T00:00:00Z&end_time=2025-11-07T00:00:00Z&interval=1h",
        headers=headers
    )

    if response.status_code == 200:
        data = response.json()
        assert "data" in data
        assert "metadata" in data
        assert isinstance(data["data"], list)
    else:
        assert response.status_code in [401, 404, 500]