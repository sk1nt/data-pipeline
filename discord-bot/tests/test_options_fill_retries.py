import os
import sys
import pytest
from decimal import Decimal

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from services.options_fill_service import OptionsFillService
from src.lib.retries import TransientError


class FakeOrder:
    def __init__(self, id):
        self.order = type('O', (), {'id': id})


def test_place_limit_order_retries_and_succeeds(monkeypatch):
    class StubClient:
        def ensure_authorized(self):
            return True

        def get_session(self):
            return object()

    svc = OptionsFillService(tastytrade_client=StubClient())

    # Fake account with place_order that fails once then succeeds
    class FakeAccount:
        def __init__(self):
            self.calls = 0

        def place_order(self, session, order, dry_run=False):
            self.calls += 1
            if self.calls == 1:
                raise TransientError("network timed out")
            return FakeOrder(999)

    monkeypatch.setattr('services.options_fill_service.Account.get', lambda session: [FakeAccount()])
    # Avoid auth/session initialization during unit test
    # Replace the tastytrade_client object used in the module with a minimal stub
    class StubClient:
        def ensure_authorized(self):
            return True

        def get_session(self):
            return object()

    monkeypatch.setattr('services.options_fill_service.tastytrade_client', StubClient())
    # Provide a fake option object that has the required interface for Option.build_leg
    class FakeOption:
        def build_leg(self, quantity, action):
            return {
                "instrument_type": "Equity Option",
                "symbol": "SPY",
                "action": action,
                "quantity": quantity,
            }

    # Call place_limit_order; it should retry and return id
    order_id = pytest.raises is None
    order_id = pytest.mark.asyncio and None
    # run as sync wrapper for our test environment
    res = None
    # We can call place_limit_order synchronously via asyncio.run in test
    import asyncio
    res = asyncio.get_event_loop().run_until_complete(
        svc.place_limit_order(FakeOption(), Decimal('1'), 'BTO', Decimal('0.75'))
    )
    assert res == '999'


def test_cancel_order_retries(monkeypatch):
    class StubClient:
        def ensure_authorized(self):
            return True

        def get_session(self):
            return object()

    svc = OptionsFillService(tastytrade_client=StubClient())

    class FakeAccount:
        def __init__(self):
            self.calls = 0

        def delete_order(self, session, order_id):
            self.calls += 1
            if self.calls == 1:
                raise TransientError("502 Bad Gateway")
            return True

    monkeypatch.setattr('services.options_fill_service.Account.get', lambda session: [FakeAccount()])
    class StubClient:
        def ensure_authorized(self):
            return True

        def get_session(self):
            return object()

    monkeypatch.setattr('services.options_fill_service.tastytrade_client', StubClient())
    import asyncio
    res = asyncio.get_event_loop().run_until_complete(svc.cancel_order('123'))
    assert res is True
