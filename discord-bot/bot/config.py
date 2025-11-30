from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os


@dataclass
class BrokerAllocation:
    percentage: float
    enabled: bool


@dataclass
class TastyTradeCredentials:
    client_id: str
    client_secret: str
    oauth_scope: Optional[str] = None
    api_base_url: Optional[str] = None
    account_whitelist: Tuple[str, ...] = ()
    default_account: Optional[str] = None
    use_cert_environment: bool = False
    use_sandbox_environment: bool = False
    sandbox_api_url: Optional[str] = None
    sandbox_uri: Optional[str] = None
    sandbox_client_id: Optional[str] = None
    sandbox_client_secret: Optional[str] = None
    sandbox_username: Optional[str] = None
    sandbox_password: Optional[str] = None
    refresh_token: str = ""
    sandbox_refresh_token: Optional[str] = None
    dry_run: bool = False


@dataclass
class BotConfig:
    discord_token: str
    thinkorswim_allocation: BrokerAllocation
    tastytrade_allocation: BrokerAllocation
    tastytrade_credentials: Optional[TastyTradeCredentials]
    status_api_secret: Optional[str]
    scid_data_directory: Optional[str]
    allowed_channel_ids: Optional[Tuple[int, ...]]
    status_channel_id: Optional[int]
    uw_channel_ids: Optional[Tuple[int, ...]]
    gex_feed_enabled: bool = False
    gex_feed_channel_ids: Optional[Tuple[int, ...]] = None
    gex_feed_symbol: str = "NQ_NDX"
    gex_feed_symbols: Optional[Tuple[str, ...]] = None
    gex_feed_channel_map: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    # Faster default cadence; can be overridden via env
    gex_feed_update_seconds: float = 0.5
    gex_feed_refresh_minutes: int = 5
    gex_feed_window_seconds: int = 60
    gex_feed_aggregation_seconds: float = 0.25
    gex_feed_metrics_enabled: bool = False
    gex_feed_metrics_key: str = "metrics:gex_feed"
    gex_feed_force_window: bool = False
    # Keep Discord rate-limit backoff small to avoid stretching cadence
    gex_feed_backoff_base_seconds: float = 0.25
    gex_feed_backoff_max_seconds: float = 1.0


def _parse_channel_ids(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    return [int(cid.strip()) for cid in value.split(",") if cid.strip()]


def _parse_symbol_list(value: Optional[str]) -> Optional[Tuple[str, ...]]:
    if not value:
        return None
    symbols = [token.strip().upper() for token in value.split(",") if token.strip()]
    return tuple(symbols) if symbols else None


def _parse_channel_map(raw: Optional[str]) -> Dict[str, Tuple[int, ...]]:
    """Parse mapping like 'NQ_NDX:123|456,ES_SPX:789,SPX:111|222'."""
    mapping: Dict[str, Tuple[int, ...]] = {}
    if not raw:
        return mapping
    for token in raw.split(","):
        if ":" not in token:
            continue
        sym, ids_raw = token.split(":", 1)
        sym = sym.strip().upper()
        if not sym:
            continue
        ids: List[int] = []
        for part in ids_raw.split("|"):
            part = part.strip()
            if not part:
                continue
            try:
                ids.append(int(part))
            except ValueError:
                continue
        if ids:
            mapping[sym] = tuple(ids)
    return mapping


def _parse_account_list(value: Optional[str]) -> Tuple[str, ...]:
    if not value:
        return ()
    return tuple(acc.strip() for acc in value.split(",") if acc.strip())


def create_config_from_env() -> BotConfig:
    discord_token = os.getenv("DISCORD_BOT_TOKEN", "")

    tos_percentage = float(os.getenv("TOS_ALLOCATION_PERCENTAGE", "50.0"))
    tos_enabled = os.getenv("TOS_ENABLED", "false").lower() == "true"

    tt_percentage = float(os.getenv("TASTYTRADE_ALLOCATION_PERCENTAGE", "10.0"))
    tt_enabled = os.getenv("TASTYTRADE_ENABLED", "true").lower() == "true"

    # TastyTrade credentials
    tt_client_id = os.getenv("TASTYTRADE_CLIENT_ID", "")
    tt_client_secret = os.getenv("TASTYTRADE_CLIENT_SECRET", "")
    tt_scope = os.getenv("TASTYTRADE_OAUTH_SCOPE")
    tt_api_base_url = os.getenv("TASTYTRADE_API_URL") or os.getenv("TASTYTRADE_URI")
    tt_account_whitelist = _parse_account_list(
        os.getenv("TASTYTRADE_ACCOUNT_WHITELIST")
    )
    tt_default_account = os.getenv("TASTYTRADE_DEFAULT_ACCOUNT") or os.getenv(
        "TASTYTRADE_ACCOUNT"
    )
    tt_use_cert = os.getenv("TASTYTRADE_USE_CERT", "false").lower() == "true"
    tt_use_sandbox = os.getenv("TASTYTRADE_USE_SANDBOX", "false").lower() == "true"
    tt_sandbox_url = (
        os.getenv("TASTYTRADE_SANDBOX_API_URL") or "https://api.sandbox.tastytrade.com"
    )
    tt_sandbox_uri = os.getenv("TASTYTRADE_SANDBOX_URI")
    tt_sandbox_client_id = os.getenv("TASTYTRADE_SANDBOX_CLIENT_ID")
    tt_sandbox_client_secret = os.getenv("TASTYTRADE_SANDBOX_CLIENT_SECRET")
    tt_sandbox_username = os.getenv("TASTYTRADE_SANDBOX_USERNAME")
    tt_sandbox_password = os.getenv("TASTYTRADE_SANDBOX_PASSWORD")
    tt_refresh_token = os.getenv("TASTYTRADE_REFRESH_TOKEN", "")
    tt_sandbox_refresh_token = os.getenv("TASTYTRADE_SANDBOX_REFRESH_TOKEN")
    tt_dry_run = os.getenv("TASTYTRADE_DRY_RUN", "true").lower() == "true"

    if tt_use_cert and tt_use_sandbox:
        raise ValueError(
            "Cannot enable both TastyTrade certification and sandbox environments"
        )

    tt_credentials = None
    if tt_client_id and tt_client_secret and tt_refresh_token:
        tt_credentials = TastyTradeCredentials(
            client_id=tt_client_id,
            client_secret=tt_client_secret,
            oauth_scope=tt_scope,
            api_base_url=tt_api_base_url,
            account_whitelist=tt_account_whitelist,
            default_account=tt_default_account,
            use_cert_environment=tt_use_cert,
            use_sandbox_environment=tt_use_sandbox,
            sandbox_api_url=tt_sandbox_url,
            sandbox_uri=tt_sandbox_uri,
            sandbox_client_id=tt_sandbox_client_id,
            sandbox_client_secret=tt_sandbox_client_secret,
            sandbox_username=tt_sandbox_username,
            sandbox_password=tt_sandbox_password,
            refresh_token=tt_refresh_token,
            sandbox_refresh_token=tt_sandbox_refresh_token,
            dry_run=tt_dry_run,
        )

    if tt_enabled and not tt_credentials:
        raise ValueError("TastyTrade is enabled but credentials are missing")

    allowed_channels = _parse_channel_ids(os.getenv("DISCORD_ALLOWED_CHANNEL_IDS"))
    uw_channel_ids = _parse_channel_ids(os.getenv("DISCORD_UW_CHANNEL_IDS"))
    status_channel_id = int(os.getenv("DISCORD_STATUS_CHANNEL_ID", "0")) or None
    gex_feed_channel_ids = _parse_channel_ids(os.getenv("DISCORD_GEX_FEED_CHANNEL_IDS"))
    gex_feed_symbol_list = _parse_symbol_list(os.getenv("GEX_FEED_SYMBOLS"))
    gex_feed_channel_map = _parse_channel_map(os.getenv("GEX_FEED_CHANNEL_MAP"))

    gex_feed_enabled = os.getenv("GEX_FEED_ENABLED", "false").lower() == "true"
    gex_feed_symbol = os.getenv("GEX_FEED_SYMBOL", "NQ_NDX")
    gex_feed_update_seconds = float(os.getenv("GEX_FEED_UPDATE_SECONDS", "0.8"))
    gex_feed_refresh_minutes = int(os.getenv("GEX_FEED_REFRESH_MINUTES", "5"))
    gex_feed_window_seconds = int(os.getenv("GEX_FEED_WINDOW_SECONDS", "60"))
    gex_feed_aggregation_seconds = float(
        os.getenv("GEX_FEED_AGGREGATION_SECONDS", "0.2")
    )
    gex_feed_metrics_enabled = (
        os.getenv("GEX_FEED_METRICS_ENABLED", "false").lower() == "true"
    )
    gex_feed_metrics_key = os.getenv("GEX_FEED_METRICS_KEY", "metrics:gex_feed")
    gex_feed_force_window = (
        os.getenv("GEX_FEED_FORCE_WINDOW", "false").lower() == "true"
    )
    gex_feed_backoff_base_seconds = float(
        os.getenv("GEX_FEED_BACKOFF_BASE_SECONDS", str(gex_feed_update_seconds))
    )
    gex_feed_backoff_max_seconds = float(os.getenv("GEX_FEED_BACKOFF_MAX_SECONDS", "5"))

    return BotConfig(
        discord_token=discord_token,
        thinkorswim_allocation=BrokerAllocation(
            percentage=tos_percentage, enabled=tos_enabled
        ),
        tastytrade_allocation=BrokerAllocation(
            percentage=tt_percentage, enabled=tt_enabled
        ),
        tastytrade_credentials=tt_credentials,
        status_api_secret=os.getenv("STATUS_API_SECRET"),
        scid_data_directory=os.getenv("SCID_DATA_DIRECTORY"),
        allowed_channel_ids=tuple(allowed_channels) if allowed_channels else None,
        uw_channel_ids=tuple(uw_channel_ids) if uw_channel_ids else None,
        status_channel_id=status_channel_id,
        gex_feed_enabled=gex_feed_enabled,
        gex_feed_channel_ids=tuple(gex_feed_channel_ids)
        if gex_feed_channel_ids
        else None,
        gex_feed_symbol=gex_feed_symbol,
        gex_feed_symbols=gex_feed_symbol_list,
        gex_feed_channel_map=gex_feed_channel_map,
        gex_feed_update_seconds=gex_feed_update_seconds,
        gex_feed_refresh_minutes=gex_feed_refresh_minutes,
        gex_feed_window_seconds=gex_feed_window_seconds,
        gex_feed_aggregation_seconds=gex_feed_aggregation_seconds,
        gex_feed_metrics_enabled=gex_feed_metrics_enabled,
        gex_feed_metrics_key=gex_feed_metrics_key,
        gex_feed_force_window=gex_feed_force_window,
        gex_feed_backoff_base_seconds=gex_feed_backoff_base_seconds,
        gex_feed_backoff_max_seconds=gex_feed_backoff_max_seconds,
    )
