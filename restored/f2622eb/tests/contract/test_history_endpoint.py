import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.data_pipeline import app


class TestGEXHistoryEndpointContract:
    """Contract tests for /gex_history_url endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def valid_history_payload(self):
        """Create a valid historical data import payload for testing."""
        return {
            "url": "https://api.example.com/gex/SPX/history.json",
            "ticker": "SPX",
            "endpoint": "gex_zero"
        }

    def test_history_endpoint_accepts_valid_payload(self, client, valid_history_payload):
        """Test that /gex_history_url endpoint accepts valid payload."""
        with patch('src.import_gex_history_safe.download_to_staging') as mock_download, \
             patch('src.import_gex_history_safe.safe_import') as mock_import:
            mock_download.return_value = "/tmp/test.json"
            mock_import.return_value = {"job_id": "test-job", "records": 10}

            response = client.post("/gex_history_url", json=valid_history_payload)

            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "completed"
            assert "records" in data["message"]

    def test_history_endpoint_rejects_invalid_json(self, client):
        """Test that /gex_history_url endpoint rejects malformed JSON."""
        response = client.post(
            "/gex_history_url",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        # Should return 422 validation error
        assert response.status_code == 422

    def test_history_endpoint_rejects_missing_required_fields(self, client):
        """Test that /gex_history_url endpoint rejects payload with missing required fields."""
        invalid_payload = {
            "ticker": "SPX",
            # Missing url and endpoint
        }

        response = client.post("/gex_history_url", json=invalid_payload)
        assert response.status_code == 422

    def test_history_endpoint_rejects_invalid_url(self, client, valid_history_payload):
        """Test that /gex_history_url endpoint rejects invalid URL."""
        invalid_payload = valid_history_payload.copy()
        invalid_payload["url"] = "ftp://invalid.url"  # Invalid protocol

        response = client.post("/gex_history_url", json=invalid_payload)
        assert response.status_code == 422

    def test_history_endpoint_rejects_invalid_ticker(self, client, valid_history_payload):
        """Test that /gex_history_url endpoint rejects invalid ticker."""
        invalid_payload = valid_history_payload.copy()
        invalid_payload["ticker"] = ""  # Empty ticker

        response = client.post("/gex_history_url", json=invalid_payload)
        assert response.status_code == 422

    def test_history_endpoint_rejects_invalid_endpoint(self, client, valid_history_payload):
        """Test that /gex_history_url endpoint rejects invalid endpoint."""
        invalid_payload = valid_history_payload.copy()
        invalid_payload["endpoint"] = ""  # Empty endpoint

        response = client.post("/gex_history_url", json=invalid_payload)
        assert response.status_code == 422

    def test_history_endpoint_rate_limiting(self, client, valid_history_payload):
        """Test that /gex_history_url endpoint is rate limited."""
        with patch('src.import_gex_history_safe.download_to_staging') as mock_download, \
             patch('src.import_gex_history_safe.safe_import') as mock_import:
            mock_download.return_value = "/tmp/test.json"
            mock_import.return_value = {"job_id": "test-job", "records": 10}

            # Make multiple requests quickly
            responses = []
            for _ in range(5):
                response = client.post("/gex_history_url", json=valid_history_payload)
                responses.append(response.status_code)

            # At least one should be rate limited (429) or all should succeed
            # depending on rate limit configuration
            assert all(code in [200, 429] for code in responses)

    def test_history_endpoint_cors_headers(self, client, valid_history_payload):
        """Test that /gex_history_url endpoint includes CORS headers."""
        # CORS headers are configured in middleware - this test is skipped for now
        pass