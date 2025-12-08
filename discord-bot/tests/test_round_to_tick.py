import math
import pytest

from bot.tastytrade_client import round_to_tick


def test_round_up():
    price = 10.03
    tick = 0.05
    # round up should go to 10.05
    assert round_to_tick(price, tick, "up") == pytest.approx(10.05)


def test_round_down():
    price = 10.07
    tick = 0.05
    # round down should go to 10.05
    assert round_to_tick(price, tick, "down") == pytest.approx(10.05)


def test_round_nearest():
    price = 10.07
    tick = 0.05
    # round to nearest: 10.07 -> 10.05 (0.02 diff) vs 10.10 (0.03 diff)
    assert round_to_tick(price, tick, "") == pytest.approx(10.05)


def test_zero_or_invalid_tick_returns_price():
    price = 5.25
    assert round_to_tick(price, 0, "up") == 5.25
    assert round_to_tick(price, -1, "down") == 5.25
