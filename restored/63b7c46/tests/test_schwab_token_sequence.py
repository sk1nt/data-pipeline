from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import schwab_token_sequence as sequence


def test_token_store_round_trip(tmp_path: Path) -> None:
    store = sequence.TokenStore(path=tmp_path / "token.json")
    payload = {"refresh_token": "abc", "access_token": "xyz"}
    store.save(payload)
    loaded = store.load()
    assert loaded["refresh_token"] == "abc"
    assert "saved_at" in loaded
    assert store.get_refresh_token() == "abc"


def test_extract_code_parses_full_url() -> None:
    url = "https://127.0.0.1:8182/?state=foo&code=bar123"
    assert sequence.extract_code(url) == "bar123"


def test_exchange_code_for_tokens_makes_request(tmp_path: Path) -> None:
    response_payload = {"access_token": "a", "refresh_token": "b"}

    def fake_post(url, headers, data, timeout):  # noqa: ANN001 - signature shaped like requests.post
        assert url.endswith("/oauth/token")
        assert data["grant_type"] == "authorization_code"
        return SimpleNamespace(
            status_code=200,
            json=lambda: response_payload,
            raise_for_status=lambda: None,
        )

    with mock.patch.object(sequence.requests, "post", side_effect=fake_post):
        tokens = sequence.exchange_code_for_tokens(
            client_id="id",
            client_secret="secret",
            redirect_uri="https://127.0.0.1",
            code="abc",
        )
    assert tokens == response_payload


def test_refresh_with_token_updates_store(tmp_path: Path) -> None:
    store = sequence.TokenStore(path=tmp_path / "token.json")
    store.save({"refresh_token": "old", "access_token": "stale"})
    response_payload = {"refresh_token": "new", "access_token": "fresh"}

    def fake_post(url, headers, data, timeout):
        assert data["refresh_token"] == "old"
        return SimpleNamespace(
            status_code=200,
            json=lambda: response_payload,
            raise_for_status=lambda: None,
        )

    with mock.patch.object(sequence.requests, "post", side_effect=fake_post):
        refreshed = sequence.refresh_with_token(
            client_id="id",
            client_secret="secret",
            refresh_token=store.get_refresh_token(),
        )
    assert refreshed == response_payload


def test_token_store_migrates_legacy_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / ".tokens" / "schwab.token"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("{'refresh_token': 'legacy', 'access_token': 'old'}")
    new_path = tmp_path / "tokens" / "schwab_token.json"
    monkeypatch.setenv("SCHWAB_TOKEN_PATH", str(new_path))
    import importlib

    module = importlib.reload(sequence)
    store = module.TokenStore()
    data = store.load()
    assert data["refresh_token"] == "legacy"
    assert store.path.exists()
