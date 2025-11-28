# ruff: noqa: E402
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest

from src.services.tastytrade_streamer import StreamerSettings, TastyTradeStreamer


class FakeDXLinkStreamer:
    instances = []

    def __init__(self, session):
        self.session = session
        self.subscribed = []
        FakeDXLinkStreamer.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def subscribe(self, event_cls, symbols):
        self.subscribed.append((event_cls, symbols))

    async def get_event(self, event_cls):
        # Always timeout to avoid long-running loops
        await asyncio.sleep(0)
        raise asyncio.TimeoutError()


@pytest.mark.asyncio
async def test_enable_depth_false(monkeypatch):
    monkeypatch.setenv("TASTYTRADE_ENABLE_DEPTH", "false")
    # Patch the DXLinkStreamer class used in the module
    import importlib

    mod = importlib.import_module("src.services.tastytrade_streamer")
    monkeypatch.setattr(mod, "DXLinkStreamer", FakeDXLinkStreamer)
    monkeypatch.setattr(mod, "Session", lambda **kwargs: object())

    settings = StreamerSettings(
        client_id="id",
        client_secret="secret",
        refresh_token="tok",
        symbols=["MNQ", "ES"],
        depth_levels=5,
        enable_depth=False,
    )
    FakeDXLinkStreamer.instances.clear()
    streamer = TastyTradeStreamer(settings)
    streamer.start()
    await asyncio.sleep(0.1)
    await streamer.stop()
    assert isinstance(streamer._task, type(None)) or not streamer.is_running
    assert len(FakeDXLinkStreamer.instances) == 1
    subs = FakeDXLinkStreamer.instances[0].subscribed
    # only trades should be subscribed when enable_depth=False
    assert (
        len([s for s in subs if "Quote" in str(s[0]) or s[0].__name__ == "Quote"]) == 0
    )


@pytest.mark.asyncio
async def test_enable_depth_true(monkeypatch):
    monkeypatch.setenv("TASTYTRADE_ENABLE_DEPTH", "true")
    import importlib

    mod = importlib.import_module("src.services.tastytrade_streamer")
    monkeypatch.setattr(mod, "DXLinkStreamer", FakeDXLinkStreamer)
    monkeypatch.setattr(mod, "Session", lambda **kwargs: object())

    settings = StreamerSettings(
        client_id="id",
        client_secret="secret",
        refresh_token="tok",
        symbols=["MNQ", "ES"],
        depth_levels=5,
        enable_depth=True,
    )
    FakeDXLinkStreamer.instances.clear()
    streamer = TastyTradeStreamer(settings)
    streamer.start()
    await asyncio.sleep(0.1)
    await streamer.stop()
    assert isinstance(streamer._task, type(None)) or not streamer.is_running
    assert len(FakeDXLinkStreamer.instances) == 1
    subs = FakeDXLinkStreamer.instances[0].subscribed
    # Quote should be subscribed when enable_depth=True
    assert (
        len([s for s in subs if "Quote" in str(s[0]) or s[0].__name__ == "Quote"]) >= 1
    )
