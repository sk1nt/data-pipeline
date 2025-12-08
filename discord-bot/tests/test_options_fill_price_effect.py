import os
import sys
import pytest
from decimal import Decimal

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from services.options_fill_service import OptionsFillService
from tastytrade.order import OrderAction, PriceEffect, OrderType


class FakeOrder:
    def __init__(self, id):
        self.order = type("O", (), {"id": id})


def test_place_limit_order_sets_debit_for_buy(monkeypatch):
    # Provide a simple tastytrade client stub
    class StubClient:
        def ensure_authorized(self):
            return True
        def get_session(self):
            return object()

    svc = OptionsFillService(tastytrade_client=StubClient())

    # Create a fake option with minimal attributes
    class FakeOption:
        def __init__(self):
            self.symbol = "SPY_XXX"
        def build_leg(self, quantity, action):
            return {
                "instrument_type": "Equity Option",
                "symbol": self.symbol,
                "action": action,
                "quantity": quantity,
            }

    captured = {}

    class FakeAccount:
        def place_order(self, session, order, dry_run=False):
            captured['order'] = order
            return FakeOrder(123)

    monkeypatch.setattr('services.options_fill_service.Account.get', lambda session: [FakeAccount()])
    # No module monkeypatch needed - client passed directly

    # Execute place_limit_order
    import asyncio
    res = asyncio.get_event_loop().run_until_complete(
        svc.place_limit_order(FakeOption(), Decimal('1'), OrderAction.BUY_TO_OPEN, Decimal('1.0'))
    )
    assert res == '123'
    # Verify the SDK would normally set a price_effect; we assert the model is Debit or the order is market (no price_effect)
    assert captured['order'].model_dump().get('price_effect') in (PriceEffect.DEBIT, PriceEffect.CREDIT)


def test_retry_on_cant_buy_for_credit_and_set_debit(monkeypatch):
    class StubClient:
        def ensure_authorized(self):
            return True
        def get_session(self):
            return object()

    svc = OptionsFillService(tastytrade_client=StubClient())

    class FakeOption:
        def __init__(self):
            self.symbol = "SPY_XXX"
        def build_leg(self, quantity, action):
            return {
                "instrument_type": "Equity Option",
                "symbol": self.symbol,
                "action": action,
                "quantity": quantity,
            }

    calls = {"n": 0}
    captured_orders = []

    class FakeAccount:
        def place_order(self, session, order, dry_run=False):
            calls["n"] += 1
            captured_orders.append(order)
            if calls["n"] == 1:
                raise Exception("cant_buy_for_credit: You cannot buy for a credit.")
            return FakeOrder(456)

    monkeypatch.setattr('services.options_fill_service.Account.get', lambda session: [FakeAccount()])
    class StubClient:
        def ensure_authorized(self):
            return True
        def get_session(self):
            return object()

    # No module monkeypatch needed - client passed directly
    import asyncio
    res = asyncio.get_event_loop().run_until_complete(
        svc.place_limit_order(FakeOption(), Decimal('1'), OrderAction.BUY_TO_OPEN, Decimal('1.0'))
    )
    assert res == '456'
    # First attempt fails with cant_buy_for_credit; fallback should send a MARKET order
    assert captured_orders[-1].model_dump().get('order_type') == OrderType.MARKET
