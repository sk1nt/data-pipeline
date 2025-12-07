import pytest
from decimal import Decimal
from types import SimpleNamespace


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


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture
def fake_redis_wrapper(fake_redis):
    return FakeRedisWrapper(fake_redis)


@pytest.fixture
def fake_tasty_client():
    return FakeTastyClient()


@pytest.fixture
def fake_account_and_order():
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

    return FakeAccount


@pytest.fixture
def fake_option_chain(monkeypatch):
    import datetime
    from src.services.options_fill_service import OptionsFillService

    class FakeOption:
        def __init__(self, symbol, strike_price, option_type):
            self.symbol = symbol
            self.strike_price = Decimal(str(strike_price))
            self.option_type = SimpleNamespace(value=option_type)

    fake_option = FakeOption('UBER_121220P78', '78', 'p')
    chain_key_date = OptionsFillService()._normalize_expiry('12/05')
    chain_key = chain_key_date.isoformat() if chain_key_date is not None else '2025-12-05'
    fake_chain = {chain_key: [fake_option]}

    # Patch both tastytrade and module-level bindings
    monkeypatch.setattr('tastytrade.instruments.get_option_chain', lambda session, symbol: fake_chain)
    monkeypatch.setattr('src.services.options_fill_service.get_option_chain', lambda session, symbol: fake_chain)
    monkeypatch.setattr('services.options_fill_service.get_option_chain', lambda session, symbol: fake_chain)

    # Patch get_market_data to return a simple bid/ask
    monkeypatch.setattr('tastytrade.market_data.get_market_data', lambda session, symbol, instrument_type=None: SimpleNamespace(bid=Decimal('0.5'), ask=Decimal('1.0')))

    return fake_chain


@pytest.fixture
def monkeypatch_options_fill_methods(monkeypatch, fake_account_and_order):
    # Keep reference to calls for assertions
    placed = []

    async def fake_place_limit_order(self, option, quantity, action, price):
        placed.append({
            "option": getattr(option, "symbol", str(option)),
            "quantity": int(quantity),
            "action": action,
            "price": Decimal(str(price)),
        })
        if len(placed) == 1:
            return "entry123"
        return "exit123"

    async def fake_check_order_filled(self, order_id):
        return str(order_id) == "entry123"

    # Patch both src and non-src module paths to be defensive
    monkeypatch.setattr('src.services.options_fill_service.OptionsFillService.place_limit_order', fake_place_limit_order)
    monkeypatch.setattr('services.options_fill_service.OptionsFillService.place_limit_order', fake_place_limit_order)
    monkeypatch.setattr('src.services.options_fill_service.OptionsFillService.check_order_filled', fake_check_order_filled)
    monkeypatch.setattr('services.options_fill_service.OptionsFillService.check_order_filled', fake_check_order_filled)

    monkeypatch.setattr('src.services.options_fill_service.Account.get', lambda session: [fake_account_and_order()])
    monkeypatch.setattr('services.options_fill_service.Account.get', lambda session: [fake_account_and_order()])

    return placed


@pytest.fixture
def monkeypatch_redis_client(monkeypatch, fake_redis_wrapper, fake_redis):
    # monkeypatch the get_redis_client used by our services
    monkeypatch.setattr('src.services.automated_options_service.get_redis_client', lambda: fake_redis_wrapper)
    return fake_redis


@pytest.fixture
def noop_sleep(monkeypatch):
    async def noop_sleep(t):
        return None

    monkeypatch.setattr('asyncio.sleep', noop_sleep)


@pytest.fixture
def allowlist_ok(monkeypatch):
    from services.auth_service import AuthService

    monkeypatch.setattr(AuthService, "verify_user_and_channel_for_automated_trades", lambda uid, cid: True)
    return True
