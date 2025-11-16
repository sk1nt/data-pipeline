"""Integration-style tests for /gex_history_url workflow."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_history_endpoint_enqueues_request(client):
    payload = {
        "url": "https://hist.gex.bot/2025-11-14_SPY_classic.json",
        "type": "gex_zero",
        "ticker": "SPY",
    }
    with patch("src.lib.gex_history_queue.gex_history_queue.enqueue_request", return_value=101) as mock_enqueue, \
         patch("src.data_pipeline._trigger_queue_processing", new=AsyncMock()) as mock_trigger:
        response = client.post("/gex_history_url", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"
    assert data["id"] == 101
    mock_enqueue.assert_called_once_with(
        url=payload["url"],
        ticker="SPY",
        endpoint="gex_zero",
        payload={},
    )


def test_history_endpoint_rejects_bad_url(client):
    payload = {
        "url": "https://example.com/file.json",
        "type": "gex_zero",
        "ticker": "SPY",
    }
    with patch("src.lib.gex_history_queue.gex_history_queue.enqueue_request") as mock_enqueue:
        response = client.post("/gex_history_url", json=payload)

    assert response.status_code == 422
    assert mock_enqueue.call_count == 0


def test_history_endpoint_infers_ticker_from_url(client):
    payload = {
        "url": "https://hist.gex.bot/2025-11-14_NQ_NDX_classic.json",
        "gex_type": "gex_zero",
        "metadata": {"any": "value"},
    }
    with patch("src.lib.gex_history_queue.gex_history_queue.enqueue_request", return_value=202) as mock_enqueue, \
         patch("src.data_pipeline._trigger_queue_processing", new=AsyncMock()):
        response = client.post("/gex_history_url", json=payload)

    assert response.status_code == 200
    mock_enqueue.assert_called_once()
    _, kwargs = mock_enqueue.call_args
    assert kwargs["ticker"] == "NQ_NDX"
    assert kwargs["endpoint"] == "gex_zero"
