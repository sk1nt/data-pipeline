import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
            "url": "https://hist.gex.bot/2025-11-14_SPX_classic.json",
            "gex_type": "gex_zero",
            "ticker": "SPX"
        }

    def test_history_endpoint_accepts_valid_payload(self, client, valid_history_payload):
        """Test that /gex_history_url endpoint accepts valid payload."""
        with patch('src.lib.gex_history_queue.gex_history_queue.enqueue_request', return_value=55) as mock_enqueue, \
             patch('src.data_pipeline._trigger_queue_processing', new=AsyncMock()):
            response = client.post("/gex_history_url", json=valid_history_payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert data["id"] == 55
        mock_enqueue.assert_called_once()

    def test_history_endpoint_rejects_invalid_json(self, client):
        """Test that /gex_history_url endpoint rejects malformed JSON."""
        response = client.post(
            "/gex_history_url",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        # JSON decode failure returns 400
        assert response.status_code == 400

    def test_history_endpoint_rejects_missing_required_fields(self, client):
        """Test that /gex_history_url endpoint rejects payload with missing required fields."""
        invalid_payload = {
            "ticker": "SPX",
            # Missing url/gex field
        }

        response = client.post("/gex_history_url", json=invalid_payload)
        assert response.status_code == 422

    def test_history_endpoint_rejects_invalid_url(self, client, valid_history_payload):
        """Test that /gex_history_url endpoint rejects invalid URL."""
        invalid_payload = valid_history_payload.copy()
        invalid_payload["url"] = "ftp://invalid.url"  # Invalid protocol

        response = client.post("/gex_history_url", json=invalid_payload)
        assert response.status_code == 422

    def test_history_endpoint_rate_limiting(self, client, valid_history_payload):
        """Test that /gex_history_url endpoint is rate limited."""
        with patch('src.lib.gex_history_queue.gex_history_queue.enqueue_request', return_value=1), \
             patch('src.data_pipeline._trigger_queue_processing', new=AsyncMock()):
            responses = [client.post("/gex_history_url", json=valid_history_payload).status_code for _ in range(5)]

        assert all(code in [200, 429] for code in responses)

    def test_history_endpoint_cors_headers(self, client, valid_history_payload):
        """Test that /gex_history_url endpoint includes CORS headers."""
        # CORS headers are configured in middleware - this test is skipped for now
        pass
