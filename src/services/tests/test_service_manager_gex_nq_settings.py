import os
import sys

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
