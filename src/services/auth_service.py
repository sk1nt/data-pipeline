from typing import List, Optional

# Import via absolute path so it works when `services` is used as a top-level package
from models.trader import Trader


class AuthService:
    # Authorized user IDs for different actions
    FUTURES_USERS = ["704125082750156840"]
    ALERT_USERS = ["700068629626224700", "704125082750156840"]
    # Users allowed to trigger automated trades from alerts
    AUTOMATED_TRADE_USERS = ALERT_USERS

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
        return discord_id in AuthService.AUTOMATED_TRADE_USERS

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
