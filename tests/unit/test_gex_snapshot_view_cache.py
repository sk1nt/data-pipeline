import json

from src import data_pipeline


class FakeRedis:
    def __init__(self):
        self.calls = []
        self.store = {}

    def setex(self, key, ttl_seconds, value):
        self.calls.append((key, ttl_seconds, value))
        self.store[key] = value


def test_gex_snapshot_view_cache_uses_separate_key():
    redis_conn = FakeRedis()
    payload = {"symbol": "NQ_NDX", "call_wall_major_strike": 29544.92}

    data_pipeline._module._cache_gex_snapshot_view(
        redis_conn, "NQ_NDX", payload, ttl_seconds=300
    )

    assert "gex:snapshot:NQ_NDX" not in redis_conn.store
    assert "gex:snapshot:view:NQ_NDX" in redis_conn.store
    assert redis_conn.calls[0][0] == "gex:snapshot:view:NQ_NDX"
    assert redis_conn.calls[0][1] == 300
    assert json.loads(redis_conn.calls[0][2]) == payload
