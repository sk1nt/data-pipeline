import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.getcwd()))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))


@pytest.fixture(autouse=True)
def ensure_test_env(monkeypatch):
    # Provide a simple test environment that imports work with
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    return True
