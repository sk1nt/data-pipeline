from typing import List, Optional

from lib.redis_client import get_redis_client
from src.config.settings import config

# Import via absolute path so it works when `services` is used as a top-level package
from models.trader import Trader


class AuthService:
    # Authorized user IDs for different actions
    FUTURES_USERS = ["704125082750156840"]
    ALERT_USERS = ["700068629626224700", "704125082750156840"]
    # Users allowed to trigger automated trades from alerts
    AUTOMATED_TRADE_USERS = ALERT_USERS
    # Channels allowed to receive automated alerts
    ALERT_CHANNELS = ["1255265167113978008"]

    @staticmethod
    def verify_user_for_futures(discord_id: str) -> bool:
        """Check if user can place futures orders."""
        return discord_id in AuthService.FUTURES_USERS

    @staticmethod
    def verify_user_for_alerts(discord_id: str) -> bool:
        """Check if user can send options alerts."""
        return discord_id in AuthService.ALERT_USERS

    @staticmethod
    def verify_user_for_automated_trades(discord_id: str) -> bool:
        """Check if user can trigger automated trades from alerts."""
        try:
            rc = get_redis_client().client
            if rc.sismember(AuthService._users_redis_key(), str(discord_id)):
                return True
        except Exception:
            pass
        cfg_users = getattr(config, "allowed_user_list", None)
        if cfg_users:
            return discord_id in cfg_users
        return discord_id in AuthService.AUTOMATED_TRADE_USERS

    @staticmethod
    def verify_channel_for_automated_trades(channel_id: str) -> bool:
        """Check if channel is allowed for automated trades via alerts."""
        return str(channel_id) in AuthService.ALERT_CHANNELS

    @staticmethod
    def verify_user_and_channel_for_automated_trades(discord_id: str, channel_id: str) -> bool:
        """Combined user and channel check for automated trades."""
        return (
            AuthService.verify_user_for_automated_trades(discord_id)
            and AuthService.verify_channel_for_automated_trades(channel_id)
        )

    # Redis-backed allowlist helpers
    @staticmethod
    def _users_redis_key() -> str:
        return "allowlist:automated_alerts:users"

    @staticmethod
    def _channels_redis_key() -> str:
        return "allowlist:automated_alerts:channels"

    @staticmethod
    def add_user_to_allowlist(discord_id: str) -> bool:
        try:
            rc = get_redis_client().client
            rc.sadd(AuthService._users_redis_key(), str(discord_id))
            return True
        except Exception:
            return False

    @staticmethod
    def remove_user_from_allowlist(discord_id: str) -> bool:
        try:
            rc = get_redis_client().client
            rc.srem(AuthService._users_redis_key(), str(discord_id))
            return True
        except Exception:
            return False

    @staticmethod
    def list_users_allowlist() -> List[str]:
        try:
            rc = get_redis_client().client
            users = rc.smembers(AuthService._users_redis_key()) or set()
            return sorted(list(users))
        except Exception:
            return list(AuthService.ALERT_USERS)

    @staticmethod
    def add_channel_to_allowlist(channel_id: str) -> bool:
        try:
            rc = get_redis_client().client
            rc.sadd(AuthService._channels_redis_key(), str(channel_id))
            return True
        except Exception:
            return False

    @staticmethod
    def remove_channel_from_allowlist(channel_id: str) -> bool:
        try:
            rc = get_redis_client().client
            rc.srem(AuthService._channels_redis_key(), str(channel_id))
            return True
        except Exception:
            return False

    @staticmethod
    def list_channels_allowlist() -> List[str]:
        try:
            rc = get_redis_client().client
            channels = rc.smembers(AuthService._channels_redis_key()) or set()
            return sorted(list(channels))
        except Exception:
            return list(AuthService.ALERT_CHANNELS)

    # Checking functions updated to consult Redis first
    @staticmethod
    def verify_user_for_alerts(discord_id: str) -> bool:
        try:
            rc = get_redis_client().client
            if rc.sismember(AuthService._users_redis_key(), str(discord_id)):
                return True
        except Exception:
            pass
        # Also consult config if set
        cfg_users = getattr(config, "allowed_user_list", None)
        if cfg_users:
            return discord_id in cfg_users
        return discord_id in AuthService.ALERT_USERS

    @staticmethod
    def verify_channel_for_automated_trades(channel_id: str) -> bool:
        try:
            rc = get_redis_client().client
            if rc.sismember(AuthService._channels_redis_key(), str(channel_id)):
                return True
        except Exception:
            pass
        cfg_channels = getattr(config, "allowed_channel_list", None)
        if cfg_channels:
            return str(channel_id) in cfg_channels
        return str(channel_id) in AuthService.ALERT_CHANNELS

    @staticmethod
    def get_user_permissions(discord_id: str) -> List[str]:
        """Get list of permissions for a user."""
        permissions = []
        if AuthService.verify_user_for_futures(discord_id):
            permissions.append("FUTURES")
        if AuthService.verify_user_for_alerts(discord_id):
            permissions.append("ALERTS")
        return permissions

    @staticmethod
    def get_trader(discord_id: str) -> Optional[Trader]:
        """Get trader info for user (mock implementation)."""
        permissions = AuthService.get_user_permissions(discord_id)
        if permissions:
            return Trader(
                discord_id=discord_id,
                permissions=permissions,
                account_id="default_account",  # TODO: map to actual accounts
                allocation_percentage=0.1,  # TODO: configurable
            )
        return None
