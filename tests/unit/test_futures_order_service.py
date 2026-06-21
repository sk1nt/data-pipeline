import asyncio
from types import SimpleNamespace

from src.services.futures_order_parser import FuturesAction, FuturesOrderParams
from src.services.futures_order_service import FuturesOrderService


def test_flat_futures_closes_short_positions_with_buy_to_close(monkeypatch):
    captured = {}

    class StubClient:
        def ensure_authorized(self):
            return True

        def get_session(self):
            return object()

    class FakeFuture:
        def build_leg(self, quantity, action):
            captured["action"] = action
            captured["quantity"] = quantity
            return SimpleNamespace(quantity=quantity, action=action)

    class FakeOrder:
        def __init__(self, order_id):
            self.order = SimpleNamespace(id=order_id)

    class FakeAccount:
        account_number = "123"
        nickname = "test"
        margin_or_cash = "margin"

        def get_positions(self, session):
            return [
                {
                    "symbol": "/MNQZ5",
                    "quantity": "1",
                    "quantity_direction": "Short",
                }
            ]

        def place_order(self, session, order, dry_run=False):
            captured["order"] = order
            captured["dry_run"] = dry_run
            return FakeOrder("abc123")

        def get_balances(self, session):
            return SimpleNamespace(
                derivative_buying_power=100000,
                equity_buying_power=100000,
                day_trading_buying_power=100000,
                net_liquidating_value=100000,
                cash_balance=100000,
            )

    monkeypatch.setattr("src.services.futures_order_service.tastytrade_client", StubClient())
    monkeypatch.setattr("src.services.futures_order_service.select_account", lambda accounts: accounts[0] if accounts else None)
    monkeypatch.setattr("src.services.futures_order_service.Account.get", lambda session: [FakeAccount()])
    monkeypatch.setattr("src.services.futures_order_service.Future.get", lambda session, symbol: FakeFuture())
    monkeypatch.setattr(
        "src.services.futures_order_service.NewOrder",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    service = FuturesOrderService()
    params = FuturesOrderParams(FuturesAction.FLAT, "/MNQZ5", 0, 1, "dry")

    result = asyncio.get_event_loop().run_until_complete(service.place_order(params, user_id="42"))

    assert "abc123" in result["message"]
    assert captured["action"].name == "BUY_TO_CLOSE"
    assert captured["quantity"] == 1


def test_flat_futures_closes_long_positions_with_sell_to_close(monkeypatch):
    captured = {}

    class StubClient:
        def ensure_authorized(self):
            return True

        def get_session(self):
            return object()

    class FakeFuture:
        def build_leg(self, quantity, action):
            captured["action"] = action
            return SimpleNamespace(quantity=quantity, action=action)

    class FakeOrder:
        def __init__(self, order_id):
            self.order = SimpleNamespace(id=order_id)

    class FakeAccount:
        account_number = "123"
        nickname = "test"
        margin_or_cash = "margin"

        def get_positions(self, session):
            return [
                {
                    "symbol": "/MNQZ5",
                    "quantity": "1",
                    "quantity_direction": "Long",
                }
            ]

        def place_order(self, session, order, dry_run=False):
            return FakeOrder("xyz789")

        def get_balances(self, session):
            return SimpleNamespace(
                derivative_buying_power=100000,
                equity_buying_power=100000,
                day_trading_buying_power=100000,
                net_liquidating_value=100000,
                cash_balance=100000,
            )

    monkeypatch.setattr("src.services.futures_order_service.tastytrade_client", StubClient())
    monkeypatch.setattr("src.services.futures_order_service.select_account", lambda accounts: accounts[0] if accounts else None)
    monkeypatch.setattr("src.services.futures_order_service.Account.get", lambda session: [FakeAccount()])
    monkeypatch.setattr("src.services.futures_order_service.Future.get", lambda session, symbol: FakeFuture())
    monkeypatch.setattr(
        "src.services.futures_order_service.NewOrder",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    service = FuturesOrderService()
    params = FuturesOrderParams(FuturesAction.FLAT, "/MNQZ5", 0, 1, "dry")

    result = asyncio.get_event_loop().run_until_complete(service.place_order(params, user_id="42"))

    assert captured["action"].name == "SELL_TO_CLOSE"
    assert "xyz789" in result["message"]
