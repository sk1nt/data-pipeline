from datetime import datetime, timedelta, timezone

import pytest
import json

from src.services.tastytrade_auth_service import (
    AUTH_REFRESH_LOCK_KEY,
    AUTH_STATE_KEY,
    TastytradeAuthError,
    TastytradeAuthService,
    TastytradeAuthSettings,
    is_hard_auth_error,
)


class FakeRedis:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.values[key] = value
        return True

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    def delete(self, key):
        self.values.pop(key, None)


class FakeRedisWrapper:
    def __init__(self, client):
        self.client = client


class FakeSession:
    created = 0

    def __init__(self, **kwargs):
        FakeSession.created += 1
        self.kwargs = kwargs
        self.refresh_count = 0
        self.session_expiration = datetime.now(timezone.utc) + timedelta(minutes=20)

    def refresh(self):
        self.refresh_count += 1
        self.session_expiration = datetime.now(timezone.utc) + timedelta(minutes=20)


def make_service(tmp_path, redis_client=None, session_factory=FakeSession, buffer=300):
    return TastytradeAuthService(
        TastytradeAuthSettings(
            client_secret="secret",
            refresh_token="refresh",
            use_sandbox=True,
            refresh_buffer_seconds=buffer,
            fallback_state_path=tmp_path / "tt_auth.json",
        ),
        redis_client=FakeRedisWrapper(redis_client or FakeRedis()),
        session_factory=session_factory,
    )


def test_get_session_does_not_recreate_when_valid(tmp_path):
    FakeSession.created = 0
    service = make_service(tmp_path)

    first = service.get_session()
    second = service.get_session()

    assert first is second
    assert FakeSession.created == 1


def test_refreshes_before_expiry(tmp_path):
    service = make_service(tmp_path, buffer=300)
    session = service.get_session()
    session.session_expiration = datetime.now(timezone.utc) + timedelta(seconds=100)
    service._session_expiration = session.session_expiration

    service.get_session()

    assert session.refresh_count == 1


def test_writes_redis_and_json_state(tmp_path):
    redis_client = FakeRedis()
    service = make_service(tmp_path, redis_client=redis_client)

    service.get_session()

    assert AUTH_STATE_KEY in redis_client.values
    assert service.settings.fallback_state_path.exists()


def test_uses_lock_to_avoid_duplicate_refresh(tmp_path):
    redis_client = FakeRedis()
    redis_client.values[AUTH_REFRESH_LOCK_KEY] = "other"
    service = make_service(tmp_path, redis_client=redis_client, buffer=300)
    session = service.get_session()
    session.session_expiration = datetime.now(timezone.utc) + timedelta(seconds=100)
    service._session_expiration = session.session_expiration

    service.refresh_session()

    assert session.refresh_count == 0


def test_invalid_grant_marks_needs_reauth(tmp_path):
    redis_client = FakeRedis()

    class InvalidSession(FakeSession):
        def __init__(self, **kwargs):
            raise RuntimeError("invalid_grant")

    service = make_service(tmp_path, redis_client=redis_client, session_factory=InvalidSession)

    with pytest.raises(TastytradeAuthError):
        service.get_session()

    assert service.status()["needs_reauth"] is True
    redis_state = json.loads(redis_client.values[AUTH_STATE_KEY])
    assert redis_state["needs_reauth"] is True
    assert redis_state["session_valid"] is False


def test_generic_unauthorized_is_not_hard_refresh_token_failure():
    assert is_hard_auth_error("401 unauthorized") is False
    assert is_hard_auth_error("invalid_grant") is True


def test_hard_refresh_failure_keeps_still_valid_session(tmp_path):
    service = make_service(tmp_path)
    session = service.get_session()

    def fail_refresh():
        raise RuntimeError("invalid_grant")

    session.refresh = fail_refresh
    session.session_expiration = datetime.now(timezone.utc) + timedelta(seconds=100)
    service._session_expiration = session.session_expiration

    with pytest.raises(TastytradeAuthError):
        service.refresh_session(force=True)

    assert service._session is session
    assert service.status()["needs_reauth"] is True
