import json
import os
import sys
from pathlib import Path
import time

# Ensure project root appears on sys.path so we can import top-level `src` modules
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.schwab_streamer import SchwabAuthClient


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
