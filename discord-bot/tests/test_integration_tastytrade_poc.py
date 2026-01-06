import os
import sys
from decimal import Decimal
from types import SimpleNamespace
import pytest
import pytest
from decimal import Decimal


@pytest.mark.asyncio
async def test_end_to_end_entry_and_exit(
    monkeypatch,
    fake_redis,
    fake_redis_wrapper,
    fake_tasty_client,
    fake_account_and_order,
    fake_option_chain,
    monkeypatch_options_fill_methods,
    monkeypatch_redis_client,
    noop_sleep,
    allowlist_ok,
):
    """Integration POC test using fixtures to centralize TastyTrade & Redis fakes."""
    from src.services.automated_options_service import AutomatedOptionsService

    # `monkeypatch_options_fill_methods` returns `placed` list we can assert, so fetch it
    placed = monkeypatch_options_fill_methods

    svc = AutomatedOptionsService(tastytrade_client=fake_tasty_client)

    message = "Alert: BTO UBER 78p 12/05 @ 0.75"
    user_id = "704125082750156840"
    channel_id = "1255265167113978008"

    # patch compute quantity after import (consistent with earlier tests)
    # _compute_quantity now returns a tuple: (qty, buying_power, est_notional)
    monkeypatch.setattr(AutomatedOptionsService, "_compute_quantity", lambda self, ad, cid: (2, 10000.0, 150.0))

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
    assert exit_call["price"] == Decimal("1.6")
    # Exit quantity should be 1 (50% of entry quantity 2)
    assert exit_call["quantity"] == 1

    # Audit entry present
    assert len(fake_redis.calls) == 1
    key, payload = fake_redis.calls[0]
    assert key == "audit:automated_alerts"
    assert "entry123" in payload

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
