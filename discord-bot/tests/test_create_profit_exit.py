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


class FakeOrder:
    def __init__(self, id, price, legs=None):
        self.id = id
        self.price = price
        self.legs = legs or []


@pytest.mark.asyncio
async def test_create_profit_exit_uses_entry_price(monkeypatch):
    svc = OptionsFillService(tastytrade_client=FakeTastyClient())

    fake_option = SimpleNamespace(symbol="UBER_121220P78")

    # Monkeypatch to simulate account.get_live_orders returning the entry order
    fake_order = FakeOrder(id=123, price=Decimal('0.75'))

    class FakeAccount:
        def get_live_orders(self, session):
            return [fake_order]

    def fake_get_session():
        return object()

    monkeypatch.setattr("services.options_fill_service.Account.get", lambda session: [FakeAccount()])

    captured = {}

    async def fake_place_limit_order(option, quantity, action, price):
        captured['exit_price'] = price
        return 'exit123'

    monkeypatch.setattr(svc, "place_limit_order", fake_place_limit_order)

    exit_id = await svc.create_profit_exit(fake_option, Decimal('1'), '123')
    assert exit_id == 'exit123'
    assert captured['exit_price'] == Decimal('1.50')


@pytest.mark.asyncio
async def test_create_profit_exit_prefers_fill_price(monkeypatch):
    svc = OptionsFillService(tastytrade_client=FakeTastyClient())

    fake_option = SimpleNamespace(symbol="UBER_121220P78")

    # Create a fake leg with fills containing a fill_price
    fake_leg = SimpleNamespace(price=Decimal('0.75'), fills=[SimpleNamespace(fill_price=Decimal('0.80'))])

    # Monkeypatch to simulate account.get_live_orders returning the entry order with legs and fills
    fake_order = FakeOrder(id=123, price=Decimal('0.75'), legs=[fake_leg])

    class FakeAccount:
        def get_live_orders(self, session):
            return [fake_order]

    monkeypatch.setattr("services.options_fill_service.Account.get", lambda session: [FakeAccount()])

    captured = {}

    async def fake_place_limit_order(option, quantity, action, price):
        captured['exit_price'] = price
        return 'exit123'

    monkeypatch.setattr(svc, "place_limit_order", fake_place_limit_order)

    exit_id = await svc.create_profit_exit(fake_option, Decimal('1'), '123')
    assert exit_id == 'exit123'
    # fill price should be used (0.80 -> exit 1.60)
    assert captured['exit_price'] == Decimal('1.60')
