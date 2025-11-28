import os
import sys
from datetime import datetime, timedelta, timezone
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.services.gexbot_poller import GEXBotPoller, GEXBotPollerSettings


class FakeTSClient:
    def __init__(self):
        self.samples = []

    def multi_add(self, records):
        # records is a list of tuples: (key, ts, value, labels)
        for r in records:
            self.samples.append(r)


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
async def test_snapshot_timestamps_increment():
    # Create fake TS client to capture samples
    ts = FakeTSClient()
    settings = GEXBotPollerSettings(
        api_key="apikey", symbols=["NQ_NDX", "SPX", "ES_SPX"]
    )
    fake_redis = FakeRedisClient()
    poller = GEXBotPoller(settings, redis_client=fake_redis, ts_client=ts)

    calls = {"count": 0}

    start_dt = datetime.utcnow().replace(tzinfo=timezone.utc)

    async def fake_fetch_symbol(session, symbol):
        calls["count"] += 1
        # advance timestamp by calls count seconds for visibility
        ts_val = (start_dt + timedelta(seconds=calls["count"])).isoformat()
        if calls["count"] >= 4:
            poller._stop_event.set()
        return {
            "symbol": symbol.upper(),
            "timestamp": ts_val,
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
    # Force RTH branch to use SPX and NQ_NDX
    poller._is_rth_now = lambda: True
    await poller._run()

    # Collect snapshot keys and timestamps
    pairs = [(rec[0], rec[1]) for rec in ts.samples if rec[0].startswith("ts:gex:")]
    # Filter to 'net_gex' metric as a sample per symbol
    net_gex_pairs = [p for p in pairs if ":net_gex:" in p[0]]
    # There should be multiple entries and timestamps should strictly increase
    assert len(net_gex_pairs) >= 2
    timestamps = [p[1] for p in net_gex_pairs]
    assert all(timestamps[i] < timestamps[i + 1] for i in range(len(timestamps) - 1))


@pytest.mark.asyncio
async def test_static_snapshot_timestamps_bumped():
    # Configure TS client and poller
    ts = FakeTSClient()
    settings = GEXBotPollerSettings(api_key="apikey", symbols=["SPX", "NQ_NDX"])
    fake_redis = FakeRedisClient()
    poller = GEXBotPoller(settings, redis_client=fake_redis, ts_client=ts)

    # Return a constant timestamp each fetch to simulate static server timestamp
    static_ts = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    calls = {"count": 0}

    async def fake_fetch_symbol_static(session, symbol):
        calls["count"] += 1
        if calls["count"] >= 4:
            poller._stop_event.set()
        return {
            "symbol": symbol.upper(),
            "timestamp": static_ts,
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

    poller._fetch_symbol = fake_fetch_symbol_static
    poller._is_rth_now = lambda: True
    await poller._run()

    # Ensure multiple net_gex samples for same symbol have increasing timestamps
    pairs = [(rec[0], rec[1]) for rec in ts.samples if rec[0].startswith("ts:gex:")]
    net_gex_pairs = [p for p in pairs if ":net_gex:" in p[0]]
    assert len(net_gex_pairs) >= 2
    # Group by symbol and ensure timestamps increase per symbol
    by_symbol = {}
    for key, ts_ms in net_gex_pairs:
        # key format: ts:gex:net_gex:SYMBOL
        parts = key.split(":")
        if len(parts) >= 4:
            sym = parts[3]
        else:
            sym = "UNKNOWN"
        by_symbol.setdefault(sym, []).append(ts_ms)
    # Check each symbol's timestamps are strictly increasing
    for sym, tss in by_symbol.items():
        assert all(tss[i] < tss[i + 1] for i in range(len(tss) - 1)), (
            f"timestamps not increasing for {sym}"
        )
