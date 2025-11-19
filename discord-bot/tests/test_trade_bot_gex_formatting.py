import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

# Ensure bot package is importable from discord-bot directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from bot.trade_bot import TradeBot, MetricSnapshot


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


def test_format_gex_short_feed_variant_hides_timestamp(bot):
    data = mk_data(zero_gamma=15000, net_gex=500.0)
    data['timestamp'] = '2024-09-01T13:00:00+00:00'
    delta_map = {
        'spot': MetricSnapshot(400.0, 1.5, 0.25, 398.5, 398.5),
        'zero_gamma': MetricSnapshot(15000.0, 25.0, None, 14975.0, 14950.0),
        'call_wall': MetricSnapshot(16000.0, 10.0, None, 15990.0, 15900.0),
        'put_wall': MetricSnapshot(14000.0, -15.0, None, 14020.0, 14010.0),
        'net_gex': MetricSnapshot(500.0, 20.0, None, 480.0, 480.0),
        'scaled_gamma': MetricSnapshot(200.0, -5.0, None, 205.0, 205.0),
        'maxchange': MetricSnapshot(30.0, 2.0, None, 28.0, 28.0),
    }
    rendered = bot.format_gex_short(data, include_time=False, delta_block=delta_map)
    assert '09/01/2024' not in rendered
    assert 'Δ1.50 +0.25%' in rendered
    assert '(14975.00)' in rendered
    assert 'scaled gamma' in rendered
    assert 'Δ' in rendered
