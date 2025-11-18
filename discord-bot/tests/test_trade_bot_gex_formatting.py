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


def mk_data(zero_gamma=12345, net_gex=879.22222):
    now = datetime.now(timezone.utc)
    return {
        'timestamp': now.isoformat(),
        'ticker': 'QQQ',
        'spot_price': 400.0,
        'zero_gamma': zero_gamma,
        'net_gex': net_gex,
        'major_pos_vol': 1000,
        'major_neg_vol': -500,
        'major_pos_oi': 10,
        'major_neg_oi': -5,
        'sum_gex_oi': 15,
    }


def test_format_gex_gamma_and_net_positive(bot):
    data = mk_data(zero_gamma=12345, net_gex=879.22222)
    out_full = bot.format_gex(data)
    out_short = bot.format_gex_short(data)

    # Zero gamma should be displayed as-is (12345 -> 12345.00)
    assert "zero gamma" in out_full
    assert "12345.00" in out_full
    assert "12345.00" in out_short

    # Net gex should include 'Bn' suffix for positive numbers and be scaled by 1000
    assert "Bn" in out_full
    assert "Bn" in out_short
    assert "0.8792Bn" in out_full or "0.8792Bn" in out_short


def test_format_gex_net_negative_shows_whole_number(bot):
    data = mk_data(zero_gamma=12345, net_gex=-879.22222)
    out_full = bot.format_gex(data)
    out_short = bot.format_gex_short(data)

    # Negative net_gex should show magnitude with 'Bn' and be colored red
    assert "0.8792Bn" in out_full
    assert "0.8792Bn" in out_short
    # Ensure color codes for red/green are present
    green = "\u001b[2;32m"
    red = "\u001b[2;31m"
    assert green in out_full or green in out_short
    assert red in bot.format_gex(data) or red in bot.format_gex_short(data)
