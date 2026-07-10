import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.config import settings
from src.data_pipeline import ServiceManager


def test_gex_nq_poller_uses_env_rth_interval(monkeypatch):
    # Configure environment-backed settings
    old_api_key = settings.gexbot_api_key
    old_enabled = settings.gex_nq_polling_enabled
    old_rth = settings.gex_nq_poll_rth_interval_seconds
    try:
        settings.gexbot_api_key = "fake"
        settings.gex_nq_polling_enabled = True
        settings.gex_nq_poll_symbols = "NQ_NDX,SPX,VIX"
        settings.gex_nq_poll_rth_interval_seconds = 0.8

        manager = ServiceManager()

        # Replace GEXBotPoller in the data-pipeline module with a fake that
        # captures the settings so we can assert on the constructed interval.
        class FakeGEXBotPoller:
            def __init__(self, settings_obj, *, redis_client=None, ts_client=None, **_):
                self.settings = settings_obj

            def start(self):
                return None

        # Patch the entrypoint module's GEXBotPoller symbol so ServiceManager
        # instantiates our fake. Use ServiceManager.__module__ to find module.
        # Since the module is dynamically loaded, patch the loaded module
        import src.data_pipeline

        monkeypatch.setattr(src.data_pipeline._module, "GEXBotPoller", FakeGEXBotPoller)

        # Start NQ poller
        manager.start_service("gex_nq_poller")
        assert manager.gex_nq_poller is not None
        assert manager.gex_nq_poller.settings.symbols == ["NQ_NDX", "SPX", "VIX"]
        assert manager.gex_nq_poller.settings.rth_interval_seconds == 0.8
    finally:
        settings.gexbot_api_key = old_api_key
        settings.gex_nq_polling_enabled = old_enabled
        settings.gex_nq_poll_rth_interval_seconds = old_rth
        try:
            manager.gex_nq_poller = None
        except Exception:
            pass


def test_gex_poller_uses_configured_symbols_when_env_missing(monkeypatch):
    old_api_key = settings.gexbot_api_key
    old_enabled = settings.gex_polling_enabled
    old_symbols = settings.gex_poll_symbols
    try:
        settings.gexbot_api_key = "fake"
        settings.gex_polling_enabled = True
        settings.gex_poll_symbols = "ES_SPX,SPY,QQQ,SPX,NDX"
        monkeypatch.delenv("GEXBOT_POLL_SYMBOLS", raising=False)

        manager = ServiceManager()

        class FakeGEXBotPoller:
            def __init__(self, settings_obj, *, redis_client=None, ts_client=None, **_):
                self.settings = settings_obj

            def start(self):
                return None

        import src.data_pipeline

        monkeypatch.setattr(src.data_pipeline._module, "GEXBotPoller", FakeGEXBotPoller)

        manager.start_service("gex_poller")
        assert manager.gex_poller is not None
        assert manager.gex_poller.settings.symbols == [
            "ES_SPX",
            "SPY",
            "QQQ",
            "SPX",
            "NDX",
        ]
        assert manager.gex_poller.settings.exclude_symbols == ["NQ_NDX", "VIX"]
    finally:
        settings.gexbot_api_key = old_api_key
        settings.gex_polling_enabled = old_enabled
        settings.gex_poll_symbols = old_symbols


def test_gex_pollers_share_one_duckdb_writer(monkeypatch):
    old_api_key = settings.gexbot_api_key
    old_gex_enabled = settings.gex_polling_enabled
    old_nq_enabled = settings.gex_nq_polling_enabled
    try:
        settings.gexbot_api_key = "fake"
        settings.gex_polling_enabled = True
        settings.gex_nq_polling_enabled = True

        manager = ServiceManager()

        class FakeWriter:
            def __init__(self, settings_obj=None):
                self.settings = settings_obj
                self.started = False
                self.stopped = False

            def start(self):
                self.started = True

            async def stop(self):
                self.stopped = True

            def status(self):
                return {"running": self.started and not self.stopped}

        class FakeGEXBotPoller:
            def __init__(self, settings_obj, *, redis_client=None, ts_client=None, duckdb_writer=None, **_):
                self.settings = settings_obj
                self.duckdb_writer = duckdb_writer
                self.redis_client = redis_client
                self.ts_client = ts_client

            def start(self):
                return None

        import src.data_pipeline

        monkeypatch.setattr(src.data_pipeline._module, "GEXDuckDBWriter", FakeWriter)
        monkeypatch.setattr(src.data_pipeline._module, "GEXBotPoller", FakeGEXBotPoller)

        manager.start_service("gex_poller")
        manager.start_service("gex_nq_poller")

        assert manager.gex_duckdb_writer is not None
        assert manager.gex_duckdb_writer.started is True
        assert manager.gex_poller is not None
        assert manager.gex_nq_poller is not None
        assert manager.gex_poller.duckdb_writer is manager.gex_duckdb_writer
        assert manager.gex_nq_poller.duckdb_writer is manager.gex_duckdb_writer
    finally:
        settings.gexbot_api_key = old_api_key
        settings.gex_polling_enabled = old_gex_enabled
        settings.gex_nq_polling_enabled = old_nq_enabled


@pytest.mark.asyncio
async def test_gex_nq_poller_poll_now_uses_one_shot_fetch(monkeypatch):
    old_api_key = settings.gexbot_api_key
    old_enabled = settings.gex_nq_polling_enabled
    try:
        settings.gexbot_api_key = "fake"
        settings.gex_nq_polling_enabled = True
        settings.gex_nq_poll_symbols = "NQ_NDX,SPX,VIX"

        manager = ServiceManager()

        class FakeGEXBotPoller:
            def __init__(self, settings_obj, *, redis_client=None, ts_client=None, **_):
                self.settings = settings_obj
                self.redis_client = redis_client
                self.ts_client = ts_client
                self.fetched = []

            async def fetch_symbol_now(self, symbol):
                self.fetched.append(symbol)
                return {"symbol": symbol, "timestamp": "2024-01-02T12:00:00+00:00"}

        import src.data_pipeline

        monkeypatch.setattr(src.data_pipeline._module, "GEXBotPoller", FakeGEXBotPoller)

        result = await manager.poll_service_now("gex_nq_poller", symbol="nq_ndx")
        assert result["service"] == "gex_nq_poller"
        assert result["symbol"] == "NQ_NDX"
        assert result["fetched"] is True
        assert result["snapshot"]["symbol"] == "NQ_NDX"
    finally:
        settings.gexbot_api_key = old_api_key
        settings.gex_nq_polling_enabled = old_enabled
