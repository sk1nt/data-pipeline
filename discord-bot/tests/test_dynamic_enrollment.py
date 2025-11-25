import sys
import os
import pytest
from types import SimpleNamespace
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from bot.trade_bot import TradeBot


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        v = self.store.get(key)
        if isinstance(v, str):
            return v
        if v is None:
            return None
        return v

    def set(self, key, value):
        self.store[key] = value

    def setex(self, key, ttl, value):
        self.store[key] = value


class FakePoller:
    def __init__(self, base_symbols=None):
        self.base_symbols = set(base_symbols or [])
        self.added = set()
        self.fetched = []

    @property
    def _base_symbols(self):
        return self.base_symbols

    def add_symbol_for_day(self, s):
        self.added.add(s.upper())

    async def fetch_symbol_now(self, symbol):
        self.fetched.append(symbol.upper())
        # Return a snapshot that has minimal required fields for get_gex_data
        now = datetime.now(timezone.utc).isoformat()
        tick = symbol.upper()
        return {
            'timestamp': now,
            'ticker': tick,
            'spot_price': 123.45,
            'zero_gamma': 0.1,
            'net_gex': 1000,
            'major_pos_vol': 111,
            'major_neg_vol': -222,
            'major_pos_oi': 10,
            'major_neg_oi': -5,
            'sum_gex_oi': 20,
            'max_priors': [],
            'strikes': [],
        }


@pytest.mark.asyncio
async def test_dynamic_enrollment_and_cache_write(monkeypatch):
    cfg = SimpleNamespace(allowed_channel_ids=())
    bot = TradeBot(cfg)
    fake_redis = FakeRedis()
    bot.redis_client = fake_redis

    # Setup fake poller with base symbols not containing 'TST'
    fake_poller = FakePoller(base_symbols={'NQ_NDX', 'ES_SPX'})
    bot.gex_poller = fake_poller

    # Ensure supported tickers map includes our test symbol 'TST'
    async def fake_get_supported():
        return {'test': ['TST'], 'futures': ['NQ_NDX']}

    monkeypatch.setattr(bot, '_get_supported_tickers', fake_get_supported)

    # Create a fake ctx with async send
    sent = []

    class Ctx:
        async def send(self, msg):
            sent.append(msg)

    ctx = Ctx()

    # Invoke the command callback via registered command object
    cmd = None
    for c in bot.commands:
        if getattr(c, 'name', None) == 'gex':
            cmd = c
            break
    assert cmd, 'gex command not registered'

    # Call the command with 'TST'
    await cmd.callback(ctx, 'TST')

    # Poller should have fetched TST (no dynamic enrollment required)
    assert 'TST' in fake_poller.fetched

    # Redis should contain snapshot key (canonical source)
    snapshot_key = 'gex:snapshot:TST'
    assert snapshot_key in fake_redis.store

    # Subsequent call to get_gex_data should hit cache and have _source redis-cache or redis-snapshot
    data = await bot.get_gex_data('TST')
    assert data is not None
    assert data.get('_source') in ('redis-cache', 'redis-snapshot', 'DB', 'local', 'current', 'stale')

    # Ensure we replied at least once
    assert sent, 'expected bot to send a response'


@pytest.mark.asyncio
async def test_dynamic_enrollment_writes_dynamic_key_without_local_poller(monkeypatch):
    cfg = SimpleNamespace(allowed_channel_ids=())
    bot = TradeBot(cfg)
    fake_redis = FakeRedis()
    bot.redis_client = fake_redis

    # No local poller instance present (None)
    bot.gex_poller = None

    # Ensure supported tickers map includes 'META'
    async def fake_get_supported():
        return {'equity': ['META', 'AAPL', 'QQQ']}

    monkeypatch.setattr(bot, '_get_supported_tickers', fake_get_supported)

    sent = []
    class Ctx:
        async def send(self, msg):
            sent.append(msg)

    ctx = Ctx()

    # Call the gex command with 'META'
    cmd = None
    for c in bot.commands:
        if getattr(c, 'name', None) == 'gex':
            cmd = c
            break
    assert cmd
    await cmd.callback(ctx, 'META')

    # Dynamic enrollment removed — no dynamic key should be written to Redis
    key = 'gexbot:symbols:dynamic'
    assert key not in fake_redis.store
    assert sent
    # No local poller was present, so snapshots may or may not be created by DB/API logic
