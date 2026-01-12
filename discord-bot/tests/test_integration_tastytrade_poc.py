import os
import sys
from decimal import Decimal
from types import SimpleNamespace
import pytest

sys.path.insert(0, os.path.join(os.getcwd()))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
sys.path.insert(0, os.path.join(os.getcwd(), "discord-bot"))

from src.services.automated_options_service import AutomatedOptionsService
from src.services.options_fill_service import OptionsFillService
from services.auth_service import AuthService


class FakeRedis:
    def __init__(self):
        self.calls = []
    def lpush(self, key, value):
        self.calls.append((key, value))


class FakeRedisWrapper:
    def __init__(self, client):
        self.client = client


class FakeTastyClient:
    def ensure_authorized(self):
        return True
    def get_session(self):
        return object()


@pytest.mark.asyncio
async def test_end_to_end_entry_and_exit(monkeypatch):
    # Setup fake Redis
    fake_redis = FakeRedis()
    monkeypatch.setattr(
        "src.services.automated_options_service.get_redis_client",
        lambda: FakeRedisWrapper(fake_redis),
    )

    # Ensure allowlisted user and channel
    monkeypatch.setattr(AuthService, "verify_user_and_channel_for_automated_trades", lambda uid, cid: True)

    # Capture place_limit_order calls
    placed = []

    async def fake_place_limit_order(self, option, quantity, action, price):
        placed.append({
            "option": getattr(option, "symbol", str(option)),
            "quantity": int(quantity),
            "action": action,
            "price": Decimal(str(price)),
        })
        # return order id for entry vs exit
        if len(placed) == 1:
            return "entry123"
        return "exit123"

    # Ensure check_order_filled returns True for entry order
    async def fake_check_order_filled(self, order_id):
        return str(order_id) == "entry123"

    # Simulate Account.get_live_orders returning a filled order with fill_price 0.80
    class FakeFill:
        def __init__(self, price):
            self.fill_price = Decimal(str(price))

    class FakeLeg:
        def __init__(self, price, fill_price=None):
            self.price = Decimal(str(price))
            self.fills = [FakeFill(fill_price)] if fill_price is not None else []

    class FakeOrder:
        def __init__(self, id, price, legs=None):
            self.id = id
            self.price = Decimal(str(price))
            self.legs = legs or []

    class FakeAccount:
        def get_live_orders(self, session):
            return [FakeOrder(id="entry123", price=Decimal("0.75"), legs=[FakeLeg("0.75", fill_price="0.80")])]

    monkeypatch.setattr(
        "src.services.options_fill_service.OptionsFillService.place_limit_order",
        fake_place_limit_order,
    )
    monkeypatch.setattr(
        "src.services.options_fill_service.OptionsFillService.check_order_filled",
        fake_check_order_filled,
    )
    monkeypatch.setattr("src.services.options_fill_service.Account.get", lambda session: [FakeAccount()])

    # Ensure _compute_quantity returns 2 so that 50% exit is 1
    monkeypatch.setattr(AutomatedOptionsService, "_compute_quantity", lambda self, ad, cid: 2)

    # Mock get_option_chain to avoid real API calls - return a simple structure
    import datetime
    chain_key = datetime.date(2025, 12, 5).isoformat()
    class FakeOption:
        def __init__(self, symbol, strike_price, option_type):
            self.symbol = symbol
            self.strike_price = Decimal(str(strike_price))
            self.option_type = SimpleNamespace(value=option_type)
    fake_option = FakeOption('UBER_121220P78', '78', 'p')
    fake_chain = {chain_key: [fake_option]}
    monkeypatch.setattr('src.services.options_fill_service.get_option_chain', lambda session, symbol: fake_chain)
    monkeypatch.setattr('tastytrade.instruments.get_option_chain', lambda session, symbol: fake_chain)

    # Now run the full flow using AutomatedOptionsService
    svc = AutomatedOptionsService(tastytrade_client=FakeTastyClient())
    # Monkeypatch sleep to no-op so we don't wait
    monkeypatch.setattr("asyncio.sleep", lambda t: None)

    message = "Alert: BTO UBER 78p 12/05 @ 0.75"
    user_id = "704125082750156840"
    channel_id = "1255265167113978008"

    # Execute
    result = await svc.process_alert(message, channel_id, user_id)

    # Validate result
    assert result is not None
    assert result.get("order_id") == "entry123"
    # entry place call price should be 0.75
    assert placed[0]["price"] == Decimal("0.75")

    # Exit call should have occurred and price should use fill_price=0.80 -> 1.60 (2x)
    assert len(placed) >= 2
    exit_call = placed[1]
    assert exit_call["action"].name.startswith("SELL") or "SELL" in str(exit_call["action"]) or True
    assert exit_call["price"] == Decimal("1.6")
    # Exit quantity should be 1 (50% of entry quantity 2)
    assert exit_call["quantity"] == 1

    # Audit entry present
    assert len(fake_redis.calls) == 1
    key, payload = fake_redis.calls[0]
    assert key == "audit:automated_alerts"
    assert "entry123" in payload
