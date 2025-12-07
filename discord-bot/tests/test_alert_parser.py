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

