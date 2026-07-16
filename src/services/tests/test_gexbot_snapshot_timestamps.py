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


class FailingTSClient:
    def multi_add(self, records):
        raise RuntimeError("TS unavailable")


class FakeDuckDBWriter:
    def __init__(self):
        self.snapshots = []

    def enqueue_snapshot(self, snapshot):
        self.snapshots.append(snapshot)


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
            "sum_gex_vol": 50,
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
    await poller.wait_for_pending_timeseries_writes()

    # Collect snapshot keys and timestamps
    pairs = [(rec[0], rec[1]) for rec in ts.samples if rec[0].startswith("ts:gex:")]
    # Filter to 'sum_gex_vol' metric as a sample per symbol
    sum_gex_vol_pairs = [p for p in pairs if ":sum_gex_vol:" in p[0]]
    assert len(sum_gex_vol_pairs) >= 2
    # Each symbol's timestamps must be strictly increasing (the per-symbol guarantee).
    # Global ordering across symbols is not guaranteed because symbols are fetched concurrently.
    by_symbol: dict = {}
    for key, ts_ms in sum_gex_vol_pairs:
        sym = key.split(":")[3]  # "ts:gex:sum_gex_vol:SPX" → "SPX"
        by_symbol.setdefault(sym, []).append(ts_ms)
    for sym, tss in by_symbol.items():
        assert all(tss[i] < tss[i + 1] for i in range(len(tss) - 1)), (
            f"timestamps not strictly increasing for {sym}: {tss}"
        )


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
            "sum_gex_vol": 50,
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
    await poller.wait_for_pending_timeseries_writes()

    # Ensure duplicate timestamps are bumped so they still publish in order
    pairs = [(rec[0], rec[1]) for rec in ts.samples if rec[0].startswith("ts:gex:")]
    sum_gex_vol_pairs = [p for p in pairs if ":sum_gex_vol:" in p[0]]
    assert len(sum_gex_vol_pairs) >= 2
    by_symbol = {}
    for key, ts_ms in sum_gex_vol_pairs:
        parts = key.split(":")
        sym = parts[3] if len(parts) >= 4 else "UNKNOWN"
        by_symbol.setdefault(sym, []).append(ts_ms)
    for sym, tss in by_symbol.items():
        assert all(tss[i] < tss[i + 1] for i in range(len(tss) - 1)), (
            f"timestamps not increasing for {sym}"
        )


@pytest.mark.asyncio
async def test_snapshot_older_than_cutoff_is_persisted():
    ts = FakeTSClient()
    settings = GEXBotPollerSettings(api_key="apikey", symbols=["SPX"])
    fake_redis = FakeRedisClient()
    fake_writer = FakeDuckDBWriter()
    poller = GEXBotPoller(
        settings, redis_client=fake_redis, ts_client=ts, duckdb_writer=fake_writer
    )

    stale_ts = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(minutes=10)
    stale_snapshot = {
        "symbol": "SPX",
        "timestamp": stale_ts.isoformat(),
        "spot": 100.0,
        "zero_gamma": 0.1,
        "sum_gex_vol": 50,
        "major_pos_vol": 10,
        "major_neg_vol": -5,
        "major_pos_oi": 1,
        "major_neg_oi": -1,
        "sum_gex_oi": 100,
        "max_priors": [],
        "strikes": [],
    }

    await poller._record_timeseries(stale_snapshot)
    await poller.wait_for_pending_timeseries_writes()

    assert ts.samples
    assert poller.snapshot_count == 1
    assert json.loads(fake_redis.get("gex:snapshot:SPX"))["timestamp"] == stale_ts.isoformat()
    assert len(fake_writer.snapshots) == 1
    assert fake_writer.snapshots[0]["symbol"] == "SPX"


@pytest.mark.asyncio
async def test_stale_snapshot_still_persists():
    ts = FakeTSClient()
    settings = GEXBotPollerSettings(api_key="apikey", symbols=["SPX"])
    fake_redis = FakeRedisClient()
    fake_writer = FakeDuckDBWriter()
    poller = GEXBotPoller(
        settings, redis_client=fake_redis, ts_client=ts, duckdb_writer=fake_writer
    )

    latest_ts = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    stale_ts = latest_ts - timedelta(minutes=5)

    fresh_snapshot = {
        "symbol": "SPX",
        "timestamp": latest_ts.isoformat(),
        "spot": 100.0,
        "zero_gamma": 0.1,
        "sum_gex_vol": 50,
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
    await poller.wait_for_pending_timeseries_writes()

    sum_gex_vol_samples = [rec for rec in ts.samples if ":sum_gex_vol:" in rec[0]]
    assert len(sum_gex_vol_samples) == 2
    assert poller.snapshot_count == 2
    assert len(fake_writer.snapshots) == 2

    cached = json.loads(fake_redis.get("gex:snapshot:SPX"))
    assert cached["timestamp"] == stale_ts.isoformat()
    assert fake_writer.snapshots[-1]["timestamp"] == stale_ts.isoformat()


def test_combine_payloads_normalizes_numeric_strings():
    poller = GEXBotPoller(GEXBotPollerSettings(api_key="apikey", symbols=["SPX"]))

    snapshot = poller._combine_payloads(
        "SPX",
        {
            "timestamp": 1700000000,
            "spot": "100.5",
            "zero_gamma": "101.5",
            "sum_gex_vol": "2500000",
            "sum_gex_oi": "1250000",
            "major_pos_vol": "1200000",
            "major_neg_vol": "-900000",
            "major_pos_oi": "800000",
            "major_neg_oi": "-700000",
            "major_pos_strike": "5025",
            "major_neg_strike": "4980",
            "delta_risk_reversal": "-0.75",
        },
    )

    assert snapshot["spot"] == 100.5
    assert snapshot["zero_gamma"] == 101.5
    assert snapshot["sum_gex_vol"] == 2500000.0
    assert snapshot["delta_risk_reversal"] == -0.75
    assert snapshot["major_pos_vol"] == 1200000.0
    assert snapshot["major_neg_vol"] == -900000.0


@pytest.mark.asyncio
async def test_snapshot_cache_writes_even_if_timeseries_fails():
    ts = FailingTSClient()
    settings = GEXBotPollerSettings(api_key="apikey", symbols=["NQ_NDX"])
    fake_redis = FakeRedisClient()
    fake_writer = FakeDuckDBWriter()
    poller = GEXBotPoller(
        settings, redis_client=fake_redis, ts_client=ts, duckdb_writer=fake_writer
    )

    snapshot = {
        "symbol": "NQ_NDX",
        "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "spot": 100.0,
        "zero_gamma": 0.1,
        "sum_gex_vol": 50,
        "major_pos_vol": 10,
        "major_neg_vol": -5,
        "major_pos_oi": 1,
        "major_neg_oi": -1,
        "sum_gex_oi": 100,
        "max_priors": [],
        "strikes": [],
    }

    await poller._record_timeseries(snapshot)
    await poller.wait_for_pending_timeseries_writes()

    cached = json.loads(fake_redis.get("gex:snapshot:NQ_NDX"))
    assert cached["symbol"] == "NQ_NDX"
    assert cached["sum_gex_vol"] == 50
    assert poller.snapshot_count == 1
    assert len(fake_writer.snapshots) == 1
    assert fake_writer.snapshots[0]["symbol"] == "NQ_NDX"
