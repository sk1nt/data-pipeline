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
    """Test that create_profit_exit correctly uses provided entry_price parameter."""
    svc = OptionsFillService(tastytrade_client=FakeTastyClient())

    fake_option = SimpleNamespace(symbol="UBER_121220P78")

    captured = {}

    async def fake_place_limit_order(option, quantity, action, price):
        captured["exit_price"] = price
        captured["quantity"] = quantity
        return "exit123"

    monkeypatch.setattr(svc, "place_limit_order", fake_place_limit_order)

    # Pass entry_price directly - no API lookup needed
    exit_id = await svc.create_profit_exit(
        fake_option, Decimal("2"), "123", entry_price=Decimal("0.75")
    )
    assert exit_id == "exit123"
    assert captured["exit_price"] == Decimal("1.50")  # 100% profit = 2x entry price
    assert captured["quantity"] == Decimal("1")  # 50% of 2 = 1


@pytest.mark.asyncio
async def test_create_profit_exit_looks_up_price_when_not_provided(monkeypatch):
    """Test that create_profit_exit looks up entry price from order when not provided."""
    svc = OptionsFillService(tastytrade_client=FakeTastyClient())

    fake_option = SimpleNamespace(symbol="UBER_121220P78")

    # Monkeypatch to simulate account.get_live_orders returning the entry order
    fake_order = FakeOrder(id=123, price=Decimal("0.75"))

    class FakeAccount:
        def get_live_orders(self, session):
            return [fake_order]

    monkeypatch.setattr(
        "services.options_fill_service.Account.get", lambda session: [FakeAccount()]
    )

    captured = {}

    async def fake_place_limit_order(option, quantity, action, price):
        captured["exit_price"] = price
        return "exit123"

    monkeypatch.setattr(svc, "place_limit_order", fake_place_limit_order)

    # No entry_price provided - should look it up from order
    exit_id = await svc.create_profit_exit(fake_option, Decimal("1"), "123")
    assert exit_id == "exit123"
    assert captured["exit_price"] == Decimal("1.50")


@pytest.mark.asyncio
async def test_create_profit_exit_prefers_fill_price(monkeypatch):
    """Test that when looking up entry price, fill price is preferred over order price."""
    svc = OptionsFillService(tastytrade_client=FakeTastyClient())

    fake_option = SimpleNamespace(symbol="UBER_121220P78")

    # Create a fake leg with fills containing a fill_price
    fake_leg = SimpleNamespace(
        price=Decimal("0.75"), fills=[SimpleNamespace(fill_price=Decimal("0.80"))]
    )

    # Monkeypatch to simulate account.get_live_orders returning the entry order with legs and fills
    fake_order = FakeOrder(id=123, price=Decimal("0.75"), legs=[fake_leg])

    class FakeAccount:
        def get_live_orders(self, session):
            return [fake_order]

    monkeypatch.setattr(
        "services.options_fill_service.Account.get", lambda session: [FakeAccount()]
    )

    captured = {}

    async def fake_place_limit_order(option, quantity, action, price):
        captured["exit_price"] = price
        return "exit123"

    monkeypatch.setattr(svc, "place_limit_order", fake_place_limit_order)

    # No entry_price provided - should look up and prefer fill_price over order.price
    exit_id = await svc.create_profit_exit(fake_option, Decimal("1"), "123")
    assert exit_id == "exit123"
    # fill price should be used (0.80 -> exit 1.60)
    assert captured["exit_price"] == Decimal("1.60")


@pytest.mark.asyncio
async def test_create_profit_exit_with_direct_entry_price(monkeypatch):
    """Test that provided entry_price takes precedence over any lookup."""
    svc = OptionsFillService(tastytrade_client=FakeTastyClient())

    fake_option = SimpleNamespace(symbol="UBER_121220P78")

    captured = {}

    async def fake_place_limit_order(option, quantity, action, price):
        captured["exit_price"] = price
        captured["quantity"] = quantity
        return "exit456"

    monkeypatch.setattr(svc, "place_limit_order", fake_place_limit_order)

    # Pass entry_price directly at 1.25 -> exit should be 2.50
    exit_id = await svc.create_profit_exit(
        fake_option, Decimal("4"), "order123", entry_price=Decimal("1.25")
    )
    assert exit_id == "exit456"
    assert captured["exit_price"] == Decimal("2.50")  # 100% profit = 2x
    assert captured["quantity"] == Decimal("2")  # 50% of 4 = 2
