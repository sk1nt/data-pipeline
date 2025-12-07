import json
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

    # Ensure duplicate timestamps are bumped so they still publish in order
    pairs = [(rec[0], rec[1]) for rec in ts.samples if rec[0].startswith("ts:gex:")]
    net_gex_pairs = [p for p in pairs if ":net_gex:" in p[0]]
    assert len(net_gex_pairs) >= 2
    by_symbol = {}
    for key, ts_ms in net_gex_pairs:
        parts = key.split(":")
        sym = parts[3] if len(parts) >= 4 else "UNKNOWN"
        by_symbol.setdefault(sym, []).append(ts_ms)
    for sym, tss in by_symbol.items():
        assert all(tss[i] < tss[i + 1] for i in range(len(tss) - 1)), (
            f"timestamps not increasing for {sym}"
        )


@pytest.mark.asyncio
async def test_snapshot_older_than_cutoff_is_skipped():
    ts = FakeTSClient()
    settings = GEXBotPollerSettings(api_key="apikey", symbols=["SPX"])
    fake_redis = FakeRedisClient()
    poller = GEXBotPoller(settings, redis_client=fake_redis, ts_client=ts)

    stale_ts = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(minutes=10)
    stale_snapshot = {
        "symbol": "SPX",
        "timestamp": stale_ts.isoformat(),
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

    await poller._record_timeseries(stale_snapshot)

    assert ts.samples == []
    assert poller.snapshot_count == 0
    assert fake_redis.get("gex:snapshot:SPX") is None


@pytest.mark.asyncio
async def test_stale_snapshot_is_skipped():
    ts = FakeTSClient()
    settings = GEXBotPollerSettings(api_key="apikey", symbols=["SPX"])
    fake_redis = FakeRedisClient()
    poller = GEXBotPoller(settings, redis_client=fake_redis, ts_client=ts)

    latest_ts = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    stale_ts = latest_ts - timedelta(minutes=5)

    fresh_snapshot = {
        "symbol": "SPX",
        "timestamp": latest_ts.isoformat(),
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
    stale_snapshot = dict(fresh_snapshot)
    stale_snapshot["timestamp"] = stale_ts.isoformat()

    await poller._record_timeseries(fresh_snapshot)
    await poller._record_timeseries(stale_snapshot)

    # Only the fresh snapshot should be stored and published
    net_gex_samples = [rec for rec in ts.samples if ":net_gex:" in rec[0]]
    assert len(net_gex_samples) == 1
    assert poller.snapshot_count == 1

    cached = json.loads(fake_redis.get("gex:snapshot:SPX"))
    assert cached["timestamp"] == latest_ts.isoformat()
