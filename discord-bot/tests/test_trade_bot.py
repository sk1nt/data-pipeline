import os
import sys
import asyncio
from types import SimpleNamespace
import pytest

sys.path.insert(0, os.path.join(os.getcwd()))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
sys.path.insert(0, os.path.join(os.getcwd(), "discord-bot"))

from discord import Message
from discord import ChannelType
from discord import Object

from bot.trade_bot import TradeBot


class FakeMessage:
    def __init__(self, content, author_id, channel_id, is_reply=False):
        self.content = content
        self.author = SimpleNamespace(id=author_id, bot=False)
        self.channel = SimpleNamespace(id=channel_id)
        self.reference = SimpleNamespace(message_id=1) if is_reply else None


@pytest.mark.asyncio
async def test_on_message_skips_reply(monkeypatch):
    # Create a simple config object with allowed channel
    config = SimpleNamespace(allowed_channel_ids=[1255265167113978008], option_alert_channel_ids=[1255265167113978008]) 

    # Monkeypatch _init_tastytrade_client to avoid external dependencies
    monkeypatch.setattr("bot.trade_bot.TradeBot._init_tastytrade_client", lambda self: SimpleNamespace())

    bot = TradeBot(config)
    # Avoid discord.utils processing by stubbing out process_commands
    async def noop_process_commands(self, message):
        return None
    monkeypatch.setattr("bot.trade_bot.TradeBot.process_commands", noop_process_commands, raising=False)

    called = {"count": 0}

    async def fake_process_alert(self, message):
        called["count"] += 1

    monkeypatch.setattr("bot.trade_bot.TradeBot._process_alert_message", fake_process_alert, raising=False)

    msg = FakeMessage("Alert: BTO UBER 78p 12/05 @ 0.75", author_id=12345, channel_id="1255265167113978008", is_reply=True)
    await bot.on_message(msg)
    assert called["count"] == 0


@pytest.mark.asyncio
async def test_on_message_processes_first_message(monkeypatch):
    config = SimpleNamespace(allowed_channel_ids=[1255265167113978008], option_alert_channel_ids=[1255265167113978008]) 
    monkeypatch.setattr("bot.trade_bot.TradeBot._init_tastytrade_client", lambda self: SimpleNamespace())
    bot = TradeBot(config)
    # Avoid discord.utils processing by stubbing out process_commands
    async def noop_process_commands(self, message):
        return None
    monkeypatch.setattr("bot.trade_bot.TradeBot.process_commands", noop_process_commands, raising=False)

    called = {"count": 0}

    async def fake_process_alert(self, message):
        called["count"] += 1

    monkeypatch.setattr("bot.trade_bot.TradeBot._process_alert_message", fake_process_alert, raising=False)
    msg = FakeMessage("Alert: BTO UBER 78p 12/05 @ 0.75", author_id=12345, channel_id="1255265167113978008", is_reply=False)
    await bot.on_message(msg)
    assert called["count"] == 1

