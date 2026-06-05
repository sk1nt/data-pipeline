import os
import sys
from types import SimpleNamespace
import pytest

# Ensure project root and src dir are on sys.path
sys.path.insert(0, os.path.join(os.getcwd()))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from src.services.alert_parser import AlertParser
import services.auth_service as svc_auth


def test_parse_various_formats(monkeypatch):
    parser = AlertParser()
    # Ensure user is allowed
    monkeypatch.setattr(svc_auth.AuthService, "verify_user_for_alerts", lambda uid: True)

    msg1 = "Alert: BTO UBER 78p 12/05 @ 0.75"
    res1 = parser.parse_alert(msg1, "someuser")
    assert res1["action"] == "BTO"
    assert res1["symbol"] == "UBER"
    assert res1["strike"] == 78.0
    assert res1["option_type"] == "p"
    assert res1["price"] == 0.75

    msg2 = "BTO UBER 78p 12/05 0.75"
    res2 = parser.parse_alert(msg2, "someuser")
    assert res2["price"] == 0.75

    msg3 = "BUY UBER 78p 12/05 0.75"
    res3 = parser.parse_alert(msg3, "someuser")
    assert res3["action"] in ("BUY", "BTO")

    msg4 = "Alert: BTO UBER 78p 12/05"
    res4 = parser.parse_alert(msg4, "someuser")
    assert res4["price"] is None


def test_parse_unauthorized_user(monkeypatch):
    parser = AlertParser()
    monkeypatch.setattr(svc_auth.AuthService, "verify_user_for_alerts", lambda uid: False)
    msg = "Alert: BTO UBER 78p 12/05 @ 0.75"
    assert parser.parse_alert(msg, "blocked") is None



import pytest

@pytest.mark.parametrize("msg,expected_label,expected_action", [
    ("Lotto: BTO UNH 397.5c 06/12 @ 1.90",         "lotto",       "BTO"),
    ("Super Lotto: BTO AAPL 200c 06/20 @ 0.50",    "super_lotto", "BTO"),
    ("Alert: BTO UBER 78p 12/05 @ 0.75",            "alert",       "BTO"),
    ("BTO TSLA 250c 07/18 @ 1.25",                  None,          "BTO"),
    ("STC TSLA 250c 07/18 @ 2.50",                  None,          "STC"),
    ("BUY SPY 530c 06/20 @ 0.80",                   None,          "BUY"),
    ("SELL SPY 530c 06/20 @ 1.60",                  None,          "SELL"),
])
def test_parse_all_prefixes(msg, expected_label, expected_action, monkeypatch):
    """Parser must correctly parse action and trade_label for all supported prefixes."""
    parser = AlertParser()
    monkeypatch.setattr(svc_auth.AuthService, "verify_user_for_alerts", lambda uid: True)
    result = parser.parse_alert(msg, "someuser")
    assert result is not None, f"Parser returned None for: {msg!r}"
    assert result["action"].upper() == expected_action, \
        f"Expected action {expected_action!r}, got {result['action']!r} for: {msg!r}"
    assert result["trade_label"] == expected_label, \
        f"Expected trade_label {expected_label!r}, got {result['trade_label']!r} for: {msg!r}"


def test_lotto_trade_label_is_lotto(monkeypatch):
    """Lotto prefix sets trade_label='lotto' which triggers 5% allocation."""
    parser = AlertParser()
    monkeypatch.setattr(svc_auth.AuthService, "verify_user_for_alerts", lambda uid: True)
    result = parser.parse_alert("Lotto: BTO UNH 397.5c 06/12 @ 1.90", "user")
    assert result is not None
    assert result["trade_label"] == "lotto"
    assert result["symbol"] == "UNH"
    assert result["strike"] == 397.5
    assert result["option_type"] == "c"
    assert result["price"] == 1.90


def test_super_lotto_trade_label(monkeypatch):
    """Super Lotto prefix sets trade_label='super_lotto'."""
    parser = AlertParser()
    monkeypatch.setattr(svc_auth.AuthService, "verify_user_for_alerts", lambda uid: True)
    result = parser.parse_alert("Super Lotto: BTO AAPL 200c 06/20 @ 0.50", "user")
    assert result is not None
    assert result["trade_label"] == "super_lotto"
    assert result["symbol"] == "AAPL"
    assert result["price"] == 0.50
