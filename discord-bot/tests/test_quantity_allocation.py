import os
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, os.path.join(os.getcwd()))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from src.services.automated_options_service import AutomatedOptionsService


def test_compute_quantity_default_buying_power(monkeypatch):
    svc = AutomatedOptionsService(tastytrade_client=SimpleNamespace())
    # Minimal alert_data with price
    alert_data = {"symbol": "UBER", "price": 7.8, "action": "BUY", "option_type": "PUT", "strike": 78, "expiry": "2025-12-05"}
    qty = svc._compute_quantity(alert_data, "1255265167113978008")
    assert isinstance(qty, int)
    assert qty >= 1
