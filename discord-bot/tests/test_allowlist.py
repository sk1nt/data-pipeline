import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from services.auth_service import AuthService
from src.config.settings import config


def test_verify_user_and_channel_config(monkeypatch):
    # Setup config allowlists via monkeypatch
    monkeypatch.setattr(config, "allowed_users", "U100")
    monkeypatch.setattr(config, "allowed_channels", "C100")

    assert AuthService.verify_user_and_channel_for_automated_trades("U100", "C100")
    assert not AuthService.verify_user_and_channel_for_automated_trades("U101", "C100")
    assert not AuthService.verify_user_and_channel_for_automated_trades("U100", "C101")
