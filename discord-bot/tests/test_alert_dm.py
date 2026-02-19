import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.getcwd()))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
sys.path.insert(0, os.path.join(os.getcwd(), "discord-bot"))

from bot.trade_bot import TradeBot


class FakeMessage:
    def __init__(self, content, author_id, channel_id):
        self.content = content
        self.author = SimpleNamespace(id=author_id, bot=False)
        self.channel = SimpleNamespace(id=channel_id)
        self.reference = None


@pytest.mark.asyncio
async def test_alert_dm_on_success(monkeypatch):
    config = SimpleNamespace(
        allowed_channel_ids=[970439141512871956],
        option_alert_channel_ids=[970439141512871956],
    )
    monkeypatch.setattr(
        "bot.trade_bot.TradeBot._init_tastytrade_client",
        lambda self: SimpleNamespace(),
    )
    bot = TradeBot(config)

    sent = {}

    async def fake_send_alert_dm(self, content):
        sent["content"] = content

    monkeypatch.setattr(
        "bot.trade_bot.TradeBot._send_alert_dm", fake_send_alert_dm, raising=False
    )

    class FakeAutomatedOptionsService:
        def __init__(self, tastytrade_client):
            pass

        async def process_alert(self, message, channel_id, user_id):
            return {"order_id": "OID-123", "quantity": 1, "entry_price": 3.75}

    monkeypatch.setattr(
        "services.automated_options_service.AutomatedOptionsService",
        FakeAutomatedOptionsService,
    )
    monkeypatch.setattr(
        "services.auth_service.AuthService.verify_user_and_channel_for_automated_trades",
        lambda user_id, channel_id: True,
    )
    monkeypatch.setattr(
        "services.auth_service.AuthService.verify_user_for_automated_trades",
        lambda user_id: True,
    )
    monkeypatch.setattr(
        "services.auth_service.AuthService.verify_channel_for_automated_trades",
        lambda channel_id: True,
    )

    msg = FakeMessage(
        "Alert: BTO CRM 215c 02/6 @ 3.75",
        author_id=700068629626224700,
        channel_id=970439141512871956,
    )
    await bot._process_alert_message(msg)

    assert "Automated order placed" in sent["content"]
