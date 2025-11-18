import pytest
from types import SimpleNamespace
from datetime import datetime, timezone
import os
import sys
import json

# Ensure bot package is importable from discord-bot directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from bot.trade_bot import TradeBot


@pytest.fixture
def bot():
    cfg = SimpleNamespace()
    # Provide minimal config to instantiate
    return TradeBot(cfg)


def mk_data(source=None, ladder_source=None, freshness=None):
    now = datetime.now(timezone.utc)
    data = {
        'timestamp': now.isoformat(),
        'ticker': 'QQQ',
        'spot_price': 400.0,
        'zero_gamma': 0.01,
        'net_gex': 0.1,
        'major_pos_vol': 1000,
        'major_neg_vol': -500,
    }
    if source:
        data['_source'] = source
    if ladder_source:
        data['_wall_ladders'] = {'source': ladder_source, 'call': None, 'put': None}
    if freshness:
        data['_freshness'] = freshness
    return data


@pytest.mark.parametrize(
    "data, expected",
    [
        (mk_data(source='cache'), 'cache'),
        (mk_data(source='DB'), 'DB'),
        (mk_data(source='API'), 'API'),
        (mk_data(ladder_source='redis-cache'), 'redis'),
        (mk_data(ladder_source='redis-snapshot'), 'redis'),
        (mk_data(ladder_source='duckdb'), 'DB'),
        (mk_data(ladder_source='payload'), 'API'),
        (mk_data(freshness='current'), 'current'),
    ],
)
def test_resolve_gex_source_label(bot, data, expected):
    label = bot._resolve_gex_source_label(data)
    assert label == expected
