import os
import sys
from types import SimpleNamespace
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from bot.tastytrade_client import TastyTradeClient


def test_extract_order_id_variants():
    client = TastyTradeClient(client_secret='x', refresh_token='y', default_account=None, use_sandbox=True, dry_run=True)
    # Simple object with id attribute
    obj = SimpleNamespace(id=123)
    assert client._extract_order_id(obj) == '123'
    # Dict with top-level id
    assert client._extract_order_id({'id': 'abc'}) == 'abc'
    # Dict with data.id
    assert client._extract_order_id({'data': {'id': 456}}) == '456'
    # None returns None
    assert client._extract_order_id(None) is None
    # Object without id returns None
    assert client._extract_order_id(SimpleNamespace(foo=1)) is None


def test_list_futures_mock(monkeypatch):
    client = TastyTradeClient(client_secret='x', refresh_token='y', default_account=None, use_sandbox=True, dry_run=True)
    # Mock session creation
    monkeypatch.setattr(client, '_ensure_session', lambda: object())

    # Prepare a fake Future.get return
    fake_future = SimpleNamespace(
        symbol='/NQZ5',
        streamer_symbol='/NQZ25:XCME',
        expiration_date=date(2025, 12, 19),
        is_tradeable=True,
        product_code='NQ',
        description='E-mini Nasdaq-100',
    )

    def fake_future_get(session, symbols=None, product_codes=None):
        return [fake_future]

    # Monkeypatch the Future.get used in the module
    import builtins
    try:
        from tastytrade.instruments import Future
    except Exception:
        Future = None

    if Future:
        monkeypatch.setattr(Future, 'get', fake_future_get)

    results = client.list_futures(['NQ'])
    assert isinstance(results, list)
    assert any(r.get('symbol') == '/NQZ5' for r in results)
    # returned items contain expected keys
    item = results[0]
    assert item.get('symbol') == '/NQZ5'
    assert item.get('streamer_symbol') == '/NQZ25:XCME'
    assert item.get('is_tradeable') is True
    assert item.get('product_code') == 'NQ'
