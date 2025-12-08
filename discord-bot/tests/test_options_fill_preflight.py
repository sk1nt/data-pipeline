import os
import sys
import pytest
from decimal import Decimal

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from services.options_fill_service import OptionsFillService
from tastytrade.order import OrderAction


class FakeOrder:
    def __init__(self, id):
        self.order = type("O", (), {"id": id})


def test_place_limit_aborts_when_options_closing_only(monkeypatch):
    class StubClient:
        def ensure_authorized(self):
            return True
        def get_session(self):
            return object()

    svc = OptionsFillService(tastytrade_client=StubClient())

    class FakeOption:
        def __init__(self):
            self.symbol = "SPY"
        def build_leg(self, quantity, action):
            return {"instrument_type": "Equity Option", "symbol": self.symbol, "action": action, "quantity": quantity}

    class FakeAccount:
        def get_trading_status(self, session):
            return type("T", (), {"is_options_closing_only": True})()
        def get_balances(self, session):
            return type("B", (), {"available_trading_funds": 100000})()
        def place_order(self, session, order, dry_run=False):
            return FakeOrder(1)

    monkeypatch.setattr('services.options_fill_service.Account.get', lambda session: [FakeAccount()])

    import asyncio
    res = asyncio.get_event_loop().run_until_complete(svc.place_limit_order(FakeOption(), Decimal('1'), OrderAction.BUY_TO_OPEN, Decimal('1.0')))
    assert res is None


def test_place_limit_aborts_on_insufficient_buying_power(monkeypatch):
    class StubClient:
        def ensure_authorized(self):
            return True
        def get_session(self):
            return object()

    svc = OptionsFillService(tastytrade_client=StubClient())

    class FakeOption:
        def __init__(self):
            self.symbol = "SPY"
        def build_leg(self, quantity, action):
            return {"instrument_type": "Equity Option", "symbol": self.symbol, "action": action, "quantity": quantity}

    class FakeAccount:
        def get_trading_status(self, session):
            return type("T", (), {"is_options_closing_only": False})()
        def get_balances(self, session):
            return type("B", (), {"available_trading_funds": 1})()
        def place_order(self, session, order, dry_run=False):
            return FakeOrder(1)

    monkeypatch.setattr('services.options_fill_service.Account.get', lambda session: [FakeAccount()])

    import asyncio
    res = asyncio.get_event_loop().run_until_complete(svc.place_limit_order(FakeOption(), Decimal('10'), OrderAction.BUY_TO_OPEN, Decimal('1.0')))
    assert res is None
