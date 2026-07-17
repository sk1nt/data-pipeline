import asyncio
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


@pytest.mark.asyncio
async def test_nq_fast_poller_accepts_distinct_timestamps_within_same_second():
    settings = GEXBotPollerSettings(
        api_key="apikey",
        symbols=["NQ_NDX"],
        rth_overlap_enabled=True,
        same_timestamp_retry_seconds=0.25,
    )
    fake_redis = FakeRedisClient()
    poller = GEXBotPoller(settings, redis_client=fake_redis, ts_client=None)

    calls = {"count": 0}
    ts_values = [
        "2026-07-16T13:00:00+00:00",
        "2026-07-16T13:00:00.500000+00:00",
    ]

    async def fake_fetch_symbol(session, symbol):
        calls["count"] += 1
        idx = min(calls["count"] - 1, len(ts_values) - 1)
        if calls["count"] >= 2:
            poller._stop_event.set()
        return {
            "symbol": symbol.upper(),
            "timestamp": ts_values[idx],
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
    poller._is_rth_now = lambda: True

    result = await poller._fetch_and_store_symbol(object(), "NQ_NDX")

    assert result["ok"] is True
    assert calls["count"] == 1

    second_result = await poller._fetch_and_store_symbol(object(), "NQ_NDX")
    assert second_result["ok"] is True
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_nq_fast_poller_can_overlap_rth_cycles():
    settings = GEXBotPollerSettings(
        api_key="apikey",
        symbols=["NQ_NDX"],
        rth_overlap_enabled=True,
    )
    fake_redis = FakeRedisClient()
    poller = GEXBotPoller(settings, redis_client=fake_redis, ts_client=None)

    start_times = []
    release_first = asyncio.Event()

    async def fake_poll_cycle(session):
        start_times.append(asyncio.get_event_loop().time())
        if len(start_times) == 1:
            await release_first.wait()
            return
        poller._stop_event.set()
        release_first.set()

    poller._poll_cycle = fake_poll_cycle
    poller._is_rth_now = lambda: True
    poller._current_interval_seconds = lambda: 1.0

    await asyncio.wait_for(poller._run(), timeout=5)

    assert len(start_times) >= 2
    assert start_times[1] - start_times[0] < 1.5
