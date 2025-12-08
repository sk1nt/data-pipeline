from src.services.notifications import notify_operator


def test_notify_operator(monkeypatch):
    calls = []

    class FakeRedis:
        def lpush(self, key, value):
            calls.append((key, value))

    monkeypatch.setattr("src.services.notifications.get_redis_client", lambda: type("W", (), {"client": FakeRedis()})())
    assert notify_operator("test") is True
    assert calls[0][0] == "notifications:operators"
    assert calls[0][1] == "test"
