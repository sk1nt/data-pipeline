from datetime import datetime, timedelta, timezone

import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.services.gexbot_poller import GEXBotPoller, GEXBotPollerSettings


class FakeRedisClient:
    def __init__(self):
        self._store = {}
        self.client = self

    # raw client methods
    def set(self, key, value):
        self._store[key] = value

    def get(self, key):
        return self._store.get(key)

    # wrapper API
    def get_cached(self, key):
        return self._store.get(key)

    def set_cached(self, key, value, ttl_seconds=300):
        self._store[key] = value
        return True

    def delete_cached(self, key):
        if key in self._store:
            del self._store[key]
            return True
        return False


@pytest.mark.asyncio
async def test_nq_poller_ignores_dynamic_enrollment():
    settings = GEXBotPollerSettings(api_key="apikey", symbols=["NQ_NDX"])
    fake_redis = FakeRedisClient()
    poller = GEXBotPoller(settings, redis_client=fake_redis, ts_client=None)

    # add_symbol_for_day no longer exists
    assert not hasattr(poller, "add_symbol_for_day")

    # Add dynamic key externally to simulate other process writing it
    expires_at = (
        datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(hours=24)
    ).isoformat()
    dynamic_payload = [{"symbol": "META", "expires_at": expires_at}]
    fake_redis.set_cached("gexbot:symbols:dynamic", dynamic_payload, ttl_seconds=86400)

    # Monkeypatch _fetch_symbol to return a snapshot if called
    async def fake_fetch_symbol(session, symbol):
        now = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        return {
            "symbol": symbol.upper(),
            "timestamp": now,
            "spot": 100.0,
            "zero_gamma": 0.1,
            "net_gex": 50,
            "major_pos_vol": 10,
            "major_neg_vol": -5,
            "major_pos_oi": 1,
            "major_neg_oi": -1,
            "sum_gex_oi": 100,
            "max_priors": [],
            "strikes": [],
        }

    poller._fetch_symbol = fake_fetch_symbol

    # Run sync - poller should not fetch dynamic symbols
    # _sync_dynamic_symbols is now a no-op and should not fetch external dynamic symbols
    await poller._sync_dynamic_symbols(None)

    # The poller should not have fetched META
    assert "META" not in poller.latest
    assert "gex:snapshot:META" not in fake_redis._store
    assert "META" not in poller._dynamic_symbols
