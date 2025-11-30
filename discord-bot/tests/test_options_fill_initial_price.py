import os
import sys
from types import SimpleNamespace
from decimal import Decimal
import pytest

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from services.options_fill_service import OptionsFillService


class FakeTastyClient:
    def ensure_authorized(self):
        return True

    def get_session(self):
        return object()


@pytest.mark.asyncio
async def test_initial_price_is_used(monkeypatch):
    svc = OptionsFillService(tastytrade_client=FakeTastyClient())

    # fake option chain with a matching option
    fake_option = SimpleNamespace(
        symbol="UBER_121220P78", strike_price=Decimal("78"), option_type=SimpleNamespace(value="p"),
    )

    # get_option_chain replacement: return a dict keyed by a date-like key
    chain = {"2025-12-05": [fake_option]}
    monkeypatch.setattr("services.options_fill_service.get_option_chain", lambda session, symbol: chain)

    # Ensure we return a mid_price if needed (but we'll provide initial_price so it's not used)
    async def fake_get_mid_price(symbol):
        return Decimal("0.5")
    monkeypatch.setattr(svc, "get_mid_price", fake_get_mid_price)

    captured = {}

    async def fake_place_limit_order(option, quantity, action, price):
        # Record only the first attempt's price
        if 'price' not in captured:
            captured['price'] = price
        return "ok"

    async def fake_check_order_filled(order_id):
        return True

    monkeypatch.setattr(svc, "place_limit_order", fake_place_limit_order)
    monkeypatch.setattr(svc, "check_order_filled", fake_check_order_filled)

    # Run fill with initial price
    svc.max_retries = 0
    svc.timeout_seconds = 0
    result = await svc.fill_options_order(
        symbol="UBER",
        strike=78,
        option_type="p",
        expiry="12/05",
        quantity=1,
        action="BTO",
        user_id="u",
        channel_id="c",
        initial_price=Decimal("0.75"),
    )

    assert result is not None
    assert captured.get('price') == Decimal("0.75")
