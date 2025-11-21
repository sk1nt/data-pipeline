import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

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
async def test_dynamic_symbol_sync_fetches_snapshot_and_persists():
    # Setup a poller with a fake redis; base symbols exclude 'META'
    settings = GEXBotPollerSettings(api_key='apikey', symbols=['NQ_NDX', 'ES_SPX'])
    fake_redis = FakeRedisClient()
    poller = GEXBotPoller(settings, redis_client=fake_redis, ts_client=None)

    # Add dynamic key externally to simulate other process writing it
    expires_at = (datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(hours=24)).isoformat()
    dynamic_payload = [{'symbol': 'META', 'expires_at': expires_at}]
    fake_redis.set_cached('gexbot:symbols:dynamic', dynamic_payload, ttl_seconds=86400)

    # Monkeypatch _fetch_symbol to return a synthetic snapshot
    async def fake_fetch_symbol(session, symbol):
        now = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        return {
            'symbol': symbol.upper(),
            'timestamp': now,
            'spot': 100.0,
            'zero_gamma': 0.1,
            'net_gex': 50,
            'major_pos_vol': 10,
            'major_neg_vol': -5,
            'major_pos_oi': 1,
            'major_neg_oi': -1,
            'sum_gex_oi': 100,
            'max_priors': [],
            'strikes': [],
        }

    poller._fetch_symbol = fake_fetch_symbol

    # Run sync - equivalent to poller detecting new addition
    await poller._sync_dynamic_symbols(None)

    # The poller's latest should include META and redis should have snapshot key
    assert 'META' in poller.latest
    snapshot_key = 'gex:snapshot:META'
    assert snapshot_key in fake_redis._store
    # Dynamic symbols map should include META with expiry
    assert 'META' in poller._dynamic_symbols
    # Ensure the persisted dynamic key still contains META
    stored = fake_redis.get_cached('gexbot:symbols:dynamic')
    assert isinstance(stored, list)
    assert any((isinstance(e, dict) and e.get('symbol') == 'META') or (isinstance(e, str) and e.upper() == 'META') for e in stored)
