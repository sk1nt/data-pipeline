import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.services.gexbot_poller import GEXBotPoller, GEXBotPollerSettings


class _FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status = status
        self._payload = payload or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload


class _FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, headers=None):
        self.calls.append((url, headers))
        if url.endswith("/tickers"):
            return _FakeResponse(payload={"stocks": ["SPY"], "indexes": ["SPX"]})
        return _FakeResponse(
            payload={
                "timestamp": "2026-06-29T13:00:00Z",
                "spot": 100.0,
                "zero_gamma": 1.0,
                "net_gex": 2.0,
                "sum_gex_oi": 3.0,
                "major_pos_vol": 4.0,
                "major_neg_vol": -5.0,
                "major_pos_oi": 6.0,
                "major_neg_oi": -7.0,
            }
        )


@pytest.mark.asyncio
async def test_gexbot_poller_uses_authorization_header():
    poller = GEXBotPoller(GEXBotPollerSettings(api_key="secret", symbols=["SPX"]))
    session = _FakeSession()

    symbols = await poller._fetch_supported_symbol_list(session)
    assert symbols == ["SPY", "SPX"]
    assert session.calls[0][1] == {
        "Authorization": "Bearer secret",
        "User-Agent": "DataPipeline/2.0",
        "Accept": "application/json",
    }

    snapshot = await poller._fetch_symbol(session, "SPX")
    assert snapshot is not None
    expected_headers = {
        "Authorization": "Bearer secret",
        "User-Agent": "DataPipeline/2.0",
        "Accept": "application/json",
    }
    assert len(session.calls) == 2
    assert session.calls[-1][0] == "https://api.gexbot.com/SPX/classic/zero"
    assert session.calls[-1][1] == expected_headers
