from src.services.schwab_streamer import build_streamer

def test_schwab_streamer_start(monkeypatch):
    # Monkeypatch streamer dependencies to avoid real network calls
    class DummyStream:
        def start(self, symbols=None, on_tick=None, on_level2=None):
            assert symbols is not None
            # Simulate successful start
        def stop(self):
            pass
    class DummyAuthClient:
        def stream(self):
            return DummyStream()
        def start_auto_refresh(self):
            pass
        def stop_auto_refresh(self):
            pass
    class DummyPublisher:
        def publish_tick(self, event):
            pass
        def publish_level2(self, event):
            pass
    # Patch build_streamer to use dummy classes
    monkeypatch.setattr('src.services.schwab_streamer.SchwabAuthClient', lambda *a, **kw: DummyAuthClient())
    monkeypatch.setattr('src.services.schwab_streamer.TradingEventPublisher', lambda *a, **kw: DummyPublisher())
    streamer = build_streamer(interactive=False)
    streamer.start()
    streamer.stop()
