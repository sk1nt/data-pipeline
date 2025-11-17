import json
import os
import sys
from pathlib import Path
import time

# Ensure project root appears on sys.path so we can import top-level `src` modules
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.schwab_streamer import SchwabAuthClient
import time


def test_schwab_token_file_is_loaded_and_used(tmp_path):
    """Integration test: create a persisted token file and ensure the client loads it."""
    # Setup: repo root / .tokens path (SchwabAuthClient computes from src/services file location)
    repo_root = Path(__file__).resolve().parents[3]
    tok_dir = repo_root / ".tokens"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok_path = tok_dir / "schwab_token.json"

    tokens = {
        "access_token": "persisted_access",
        "refresh_token": "persisted_refresh",
        "expires_in": 3600,
    }
    with open(tok_path, "w") as fh:
        json.dump(tokens, fh)

    # Create a quick dummy schwab client that can accept `tokens` write and expose them
    class FileSchwabDummy:
        def __init__(self):
            self.tokens = {"access_token": "initial", "refresh_token": "initialR", "expires_in": 10}

        def refresh_token(self):
            # emulate update
            self.tokens["access_token"] = "refreshed"

        @property
        def access_token(self):
            return self.tokens.get("access_token")

    dummy = FileSchwabDummy()
    client = SchwabAuthClient(
        client_id="test",
        client_secret="secret",
        refresh_token="r",
        rest_url="https://api.test",
        schwab_client=dummy,
    )

    # After initialization, the persisted tokens should have been loaded into `dummy.tokens`.
    assert dummy.tokens.get("access_token") == "persisted_access"
    assert dummy.tokens.get("refresh_token") == "persisted_refresh"

    # Clean up
    try:
        tok_path.unlink()
        if not any(tok_dir.iterdir()):
            tok_dir.rmdir()
    except Exception:
        pass


def test_manual_refresh_persists_tokens_to_file(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    tok_dir = repo_root / ".tokens"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok_path = tok_dir / "schwab_token.json"
    # start with empty tokens
    if tok_path.exists():
        tok_path.unlink()

    class ManualDummy:
        def __init__(self):
            self.tokens = {"access_token": "a", "refresh_token": "r", "expires_in": 3600}

        def refresh_token(self):
            self.tokens["access_token"] = "b"

        @property
        def access_token(self):
            return self.tokens['access_token']

    dummy = ManualDummy()
    client = SchwabAuthClient(
        client_id="test",
        client_secret="secret",
        refresh_token="r",
        rest_url="https://api.test",
        schwab_client=dummy,
    )
    try:
        assert not tok_path.exists()
        client.refresh_tokens()
        # read persisted file
        with open(tok_path, "r") as fh:
            persisted = json.load(fh)
        assert persisted.get("access_token") == "b"
    finally:
        try:
            tok_path.unlink()
            if not any(tok_dir.iterdir()):
                tok_dir.rmdir()
        except Exception:
            pass


def test_auto_refresh_persists_tokens_under_short_interval(tmp_path):
    """Integration test: a short auto-refresh interval persists updated tokens to disk."""
    repo_root = Path(__file__).resolve().parents[3]
    tok_dir = repo_root / ".tokens"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok_path = tok_dir / "schwab_token.json"
    if tok_path.exists():
        tok_path.unlink()

    class AutoSchwabDummy:
        def __init__(self):
            self.tokens = {"access_token": "initial", "refresh_token": "r", "expires_in": 3600}
            self.calls = 0

        def refresh_token(self):
            self.calls += 1
            self.tokens["access_token"] = f"refreshed_{self.calls}"

        @property
        def access_token(self):
            return self.tokens.get("access_token")

    dummy = AutoSchwabDummy()
    client = SchwabAuthClient(
        client_id="test",
        client_secret="secret",
        refresh_token="r",
        rest_url="https://api.test",
        schwab_client=dummy,
        access_refresh_interval_seconds=0.05,
        refresh_token_rotate_interval_seconds=10,
    )
    try:
        client.start_auto_refresh()
        # Allow a few refresh cycles
        time.sleep(0.3)
        client.stop_auto_refresh()
        # tokens file should now exist and reflect a refreshed access_token
        assert tok_path.exists()
        with open(tok_path, "r") as fh:
            persisted = json.load(fh)
        assert persisted.get("access_token", "") != "initial"
    finally:
        try:
            tok_path.unlink()
            if not any(tok_dir.iterdir()):
                tok_dir.rmdir()
        except Exception:
            pass
