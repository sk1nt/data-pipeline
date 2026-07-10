import os
import sys
from datetime import datetime, timezone
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.services.gexbot_poller import GEXBotPoller, GEXBotPollerSettings


class FakeRedisClient:
    def __init__(self):
        self._store = {}
        self.client = self

    def set(self, key, value):
        self._store[key] = value

    def get(self, key):
        return self._store.get(key)

    def get_cached(self, key):
        return self._store.get(key)

    def set_cached(self, key, value, ttl_seconds=300):
        self._store[key] = value
        return True


@pytest.mark.asyncio
async def test_nq_fast_poller_symbol_selection():
    settings = GEXBotPollerSettings(
        api_key="apikey", symbols=["NQ_NDX", "SPX", "VIX", "ES_SPX"]
    )
    fake_redis = FakeRedisClient()
    poller = GEXBotPoller(settings, redis_client=fake_redis, ts_client=None)

    fetched = []

    async def fake_fetch_symbol(session, symbol):
        fetched.append(symbol)
        # Stop after we fetch the RTH fast-path symbols (SPX, NQ_NDX, VIX)
        if len(fetched) >= 3:
            poller._stop_event.set()
        return {
            "symbol": symbol.upper(),
            "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
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
    # Force the poller to use the RTH branch by monkeypatching _is_rth_now
    poller._is_rth_now = lambda: True
    # Run the poller (will stop after three fetches)
    await poller._run()
    assert fetched == ["SPX", "NQ_NDX", "VIX"]
    status = poller.status()
    assert status["last_cycle_mode"] == "rth_fast_path"
    assert status["last_cycle_symbols"] == ["SPX", "NQ_NDX", "VIX"]
    assert status["last_cycle_success_count"] == 3
    assert status["is_rth_now"] is True
    assert status["effective_interval_seconds"] == 1.0
