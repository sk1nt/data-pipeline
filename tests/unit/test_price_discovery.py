from decimal import Decimal
from src.services.price_discovery import discovery_attempts


def test_discovery_attempts_toward_market_up():
    start = Decimal("0.75")
    market = Decimal("1.00")
    tick = Decimal("0.01")
    attempts = list(discovery_attempts(start, market, tick, max_increments=3))
    # First attempt is start
    assert attempts[0] == (start, False)
    # Next increments move toward market
    assert attempts[1][0] == Decimal("0.76")
    assert attempts[-1][1] in (False, True)


def test_discovery_attempts_convert_to_market_when_close():
    start = Decimal("0.99")
    market = Decimal("1.00")
    tick = Decimal("0.01")
    attempts = list(discovery_attempts(start, market, tick, max_increments=3, convert_to_market_if_remaining_ticks_leq=1))
    # Expect conversion to market in last yielded tuple
    assert any(convert for _, convert in attempts)
