import asyncio
import json
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import discord
import httpx
import redis
from discord.ext import commands
from zoneinfo import ZoneInfo

# Add src to path for importing services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from .tastytrade_client import TastyTradeClient, TastytradeAuthError
from services.auth_service import AuthService


class TradeBot(commands.Bot):
    def __init__(self, config):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
        self.config = config
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
        )
        self.command_admin_ids = self._parse_admin_ids()
        self.command_admin_names = self._parse_admin_names()
        self.tastytrade_client = self._init_tastytrade_client()
        self.uw_option_latest_key = os.getenv(
            "UW_OPTION_LATEST_KEY", "uw:options-trade:latest"
        )
        self.uw_option_history_key = os.getenv(
            "UW_OPTION_HISTORY_KEY", "uw:options-trade:history"
        )
        self.uw_market_latest_key = os.getenv(
            "UW_MARKET_LATEST_KEY", "uw:market-state:latest"
        )
        self.uw_market_history_key = os.getenv(
            "UW_MARKET_HISTORY_KEY", "uw:market-state:history"
        )
        self.uw_option_stream_channel = os.getenv(
            "UW_OPTION_STREAM_CHANNEL", "uw:options-trade:stream"
        )
        # Canonical tickers for GEXBot futures endpoints so cache/API reuse shared contracts
        self.ticker_aliases = {
            "NQ": "NQ_NDX",
            "MNQ": "NQ_NDX",
            "ES": "ES_SPX",
            "MES": "ES_SPX",
        }
        self.display_zone = ZoneInfo(os.getenv("DISPLAY_TIMEZONE", "America/New_York"))
        # Accept legacy feed-specific env overrides so Discord-only deployments stay in sync
        self.redis_snapshot_prefix = os.getenv(
            "GEX_SNAPSHOT_PREFIX",
            os.getenv("GEX_FEED_SNAPSHOT_PREFIX", "gex:snapshot:"),
        )
        self.gex_snapshot_channel = os.getenv(
            "GEX_SNAPSHOT_CHANNEL",
            os.getenv("GEX_FEED_SNAPSHOT_CHANNEL", "gex:snapshot:stream"),
        )
        self.allowed_channel_ids = tuple(
            getattr(config, "allowed_channel_ids", ()) or ()
        )
        self.status_channel_id = getattr(config, "status_channel_id", None)
        self.status_command_user = os.getenv("DISCORD_STATUS_USER", "skint0552").lower()
        self.option_alert_channel_ids = self._init_alert_channels()
        self._uw_listener_task: Optional[asyncio.Task] = None
        self._uw_listener_stop = asyncio.Event()
        self._gex_feed_task: Optional[asyncio.Task] = None
        self._gex_feed_stop = asyncio.Event()
        self.gex_feed_enabled = bool(getattr(config, "gex_feed_enabled", False))
        self.gex_feed_channel_ids = getattr(config, "gex_feed_channel_ids", None)
        raw_feed_symbol = getattr(config, "gex_feed_symbol", "NQ_NDX") or "NQ_NDX"
        # Gracefully handle comma-separated symbols passed via GEX_FEED_SYMBOL
        raw_feed_symbol_list = None
        if isinstance(raw_feed_symbol, str) and "," in raw_feed_symbol:
            raw_feed_symbol_list = [
                sym.strip().upper()
                for sym in raw_feed_symbol.split(",")
                if sym.strip()
            ]
        self.gex_feed_symbol = (raw_feed_symbol_list or [raw_feed_symbol])[0].upper()
        self.gex_feed_symbols = tuple(
            (getattr(config, "gex_feed_symbols", None) or ())
            or raw_feed_symbol_list
            or (self.gex_feed_symbol,)
        )
        self.gex_feed_channel_map = {
            (k or "").upper(): tuple(v)
            for k, v in (getattr(config, "gex_feed_channel_map", {}) or {}).items()
            if k and v
        }
        # Slightly faster default cadence; allow overrides down to 250ms
        self.gex_feed_update_seconds = max(
            0.25, float(getattr(config, "gex_feed_update_seconds", 0.5) or 0.5)
        )
        self.gex_feed_refresh_minutes = max(
            1, int(getattr(config, "gex_feed_refresh_minutes", 5) or 5)
        )
        self.gex_feed_window_seconds = max(
            15, int(getattr(config, "gex_feed_window_seconds", 60) or 60)
        )
        self.gex_feed_force_window = bool(
            getattr(config, "gex_feed_force_window", False)
        )
        self.gex_feed_aggregation_seconds = max(
            0.25, float(getattr(config, "gex_feed_aggregation_seconds", 0.25) or 0.25)
        )
        metrics_key = (
            getattr(config, "gex_feed_metrics_key", "metrics:gex_feed")
            or "metrics:gex_feed"
        )
        metrics_enabled = bool(getattr(config, "gex_feed_metrics_enabled", False))
        self._gex_feed_metrics = RedisGexFeedMetrics(
            self.redis_client, metrics_key, enabled=metrics_enabled
        )
        self.gex_feed_backoff_base_seconds = max(
            0.05, float(getattr(config, "gex_feed_backoff_base_seconds", 0.25) or 0.25)
        )
        self.gex_feed_backoff_max_seconds = max(
            self.gex_feed_backoff_base_seconds,
            float(getattr(config, "gex_feed_backoff_max_seconds", 1.0) or 1.0),
        )
        # Minimum delay between Discord edits; can be overridden via config/env to 1s
        self.gex_feed_edit_seconds = max(
            0.0, float(getattr(config, "gex_feed_edit_seconds", 0.0) or 0.0)
        )
        self._gex_feed_backoff_seconds = 0.0
        self._gex_feed_block_until: Optional[datetime] = None
        # Disable direct API polling for GEX snapshots when running inside the bot; rely on Redis/DuckDB
        self.gex_api_enabled = os.getenv("GEX_API_ENABLED", "false").lower() == "true"
        self.gexbot_supported_key = getattr(
            config, "gexbot_supported_key", "gexbot:symbols:supported"
        )
        default_symbols = getattr(config, "gexbot_default_symbols", None)
        if not default_symbols:
            default_symbols = ["NQ_NDX", "ES_SPX", "SPY", "QQQ", "SPX", "NDX"]
        if isinstance(default_symbols, str):
            default_symbols = [
                sym.strip() for sym in default_symbols.split(",") if sym.strip()
            ]
        self._gexbot_default_symbols = {sym.upper() for sym in default_symbols}
        # Use a path variable and create short-lived connections per query to avoid file locks
        self.duckdb_path = os.getenv(
            "DUCKDB_PATH", "/home/rwest/projects/data-pipeline/data/gex_data.db"
        )
        # Register commands defined as methods on the subclass so prefix commands work
        try:
            # Use plain functions (closures) as command callbacks so signature checks pass.
            async def _ping_cmd(ctx):
                await ctx.send("Pong!")

            async def _gex_cmd(ctx, *args):
                symbol, show_full = self._resolve_gex_request(args)
                display_symbol = (symbol or "QQQ").upper()
                ticker = self.ticker_aliases.get(display_symbol, display_symbol)

                # verify supported symbol map before continuing
                tickers_map = await self._get_supported_tickers()
                supported_set = set()
                for values in tickers_map.values():
                    if isinstance(values, list):
                        supported_set.update(
                            val.upper() for val in values if isinstance(val, str)
                        )

                if tickers_map and ticker.upper() not in supported_set:
                    msg = self._format_supported_tickers_message(
                        tickers_map, alias_map=self.ticker_aliases
                    )
                    await ctx.send(f"Unsupported symbol '{symbol}'.\n{msg}")
                    return

                # Dynamic symbol enrollment removed; do not touch external keys
                poller = getattr(self, "gex_poller", None)

                if poller:
                    # Verify TastyTrade auth/session is valid before attempting order
                    if self.tastytrade_client:
                        try:
                            tt_status = await asyncio.to_thread(
                                self.tastytrade_client.get_auth_status
                            )
                        except Exception:
                            tt_status = None
                        if tt_status and not tt_status.get("session_valid"):
                            await self._send_dm_or_warn(
                                ctx,
                                "TastyTrade authentication invalid or expired. Update the refresh token or run `!tt auth refresh <token>`.",
                            )
                            return

                    # Preflight: ensure the TastyTrade session is authorized to avoid
                    # attempting a live trade when refresh token is invalid.
                    # Ensure authorized before attempting to fetch positions/close.
                    try:
                        if self.tastytrade_client:
                            await asyncio.to_thread(
                                self.tastytrade_client.ensure_authorized
                            )
                    except TastytradeAuthError as exc:
                        await self._send_dm_or_warn(
                            ctx,
                            f"TastyTrade authentication invalid: {exc}. Update the refresh token or run `!tt auth refresh <token>`.",
                        )
                        return

                    try:
                        if self.tastytrade_client:
                            await asyncio.to_thread(
                                self.tastytrade_client.ensure_authorized
                            )
                    except TastytradeAuthError as exc:
                        await self._send_dm_or_warn(
                            ctx,
                            f"TastyTrade authentication invalid: {exc}. Update the refresh token or run `!tt auth refresh <token>`.",
                        )
                        return

                    try:
                        # No dynamic enrollment; just fetch snapshot from poller if available
                        if hasattr(poller, "fetch_symbol_now"):
                            snap = await poller.fetch_symbol_now(ticker)
                            if snap:
                                snap["_freshness"] = "current"
                                snap["_source"] = "redis-cache"
                                snap["display_symbol"] = display_symbol
                                snap["spot_price"] = snap.get("spot_price") or snap.get(
                                    "spot"
                                )
                                snap["major_pos_vol"] = (
                                    snap.get("major_pos_vol")
                                    if snap.get("major_pos_vol") is not None
                                    else 0
                                )
                                snap["major_neg_vol"] = (
                                    snap.get("major_neg_vol")
                                    if snap.get("major_neg_vol") is not None
                                    else 0
                                )
                                snap["major_pos_oi"] = (
                                    snap.get("major_pos_oi")
                                    if snap.get("major_pos_oi") is not None
                                    else 0
                                )
                                snap["major_neg_oi"] = (
                                    snap.get("major_neg_oi")
                                    if snap.get("major_neg_oi") is not None
                                    else 0
                                )
                                snap["sum_gex_oi"] = (
                                    snap.get("sum_gex_oi")
                                    if snap.get("sum_gex_oi") is not None
                                    else 0
                                )
                                snap["max_priors"] = snap.get("max_priors") or []
                                # Only write to the canonical snapshot key; do NOT create gex:{ticker}:latest
                                await asyncio.to_thread(
                                    self.redis_client.set,
                                    f"{self.redis_snapshot_prefix}{ticker.upper()}",
                                    json.dumps(snap, default=str),
                                )
                                formatter = (
                                    self.format_gex
                                    if show_full
                                    else self.format_gex_small
                                )
                                await ctx.send(formatter(snap))
                                return
                    except Exception:
                        pass
                data = await self.get_gex_data(symbol)
                if data:
                    formatter = self.format_gex if show_full else self.format_gex_small
                    await ctx.send(formatter(data))
                else:
                    await ctx.send("GEX data not available")

            async def _status_cmd(ctx):
                if not await self._ensure_status_channel_access(ctx):
                    return
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get("http://localhost:8877/status")
                        if response.status_code == 200:
                            status_data = response.json()
                            await ctx.send(self.format_status(status_data))
                        else:
                            await ctx.send(
                                f"Failed to fetch status: {response.status_code}"
                            )
                except Exception as e:
                    await ctx.send(f"Error fetching status: {e}")

            async def _tastytrade_cmd(ctx):
                if not await self._ensure_privileged(ctx):
                    return
                await _tt_cmd(ctx)

            async def _market_cmd(ctx):
                snapshot = await self._fetch_market_snapshot()
                if not snapshot:
                    await ctx.send("No market state snapshot available")
                    return
                await ctx.send(self.format_market_snapshot(snapshot))

            async def _uwalerts_cmd(ctx):
                alerts = await self._fetch_option_history(limit=5)
                if not alerts:
                    await ctx.send("No option trade alerts yet")
                    return
                formatted = "\n".join(
                    self.format_option_trade_alert(alert) for alert in alerts
                )
                await ctx.send(formatted)

            async def _tt_cmd(ctx, *args):
                if not await self._ensure_status_channel_access(ctx):
                    return
                if not self.tastytrade_client:
                    await self._send_dm_or_warn(
                        ctx, "TastyTrade client is not configured."
                    )
                    return
                if not args:
                    summary = await self._fetch_tastytrade_summary()
                    if summary is None:
                        await self._send_dm_or_warn(
                            ctx, "Unable to fetch TastyTrade summary."
                        )
                        return
                    message = self.format_tastytrade_summary(summary)
                    await self._send_dm_or_warn(ctx, message)
                    return
                subcommand = args[0].lower()
                if subcommand == "status":
                    overview = await self._fetch_tastytrade_overview()
                    if overview is None:
                        await self._send_dm_or_warn(
                            ctx, "Unable to fetch TastyTrade overview."
                        )
                        return
                    message = self.format_tastytrade_overview(overview)
                    await self._send_dm_or_warn(ctx, message)
                    return
                if subcommand == "account" and len(args) >= 2:
                    target = args[1]
                    success = await asyncio.to_thread(
                        self.tastytrade_client.set_active_account, target
                    )
                    dm_msg = (
                        f"Active TastyTrade account set to {target}."
                        if success
                        else f"Account {target} not found."
                    )
                    await self._send_dm_or_warn(ctx, dm_msg)
                    return
                if subcommand == "help":
                    # Contextual help support: `!tt help cancel` gives targeted info
                    if len(args) >= 2:
                        topic = args[1].lower()
                        if topic == "order" or topic == "orders":
                            await self._send_dm_or_warn(
                                ctx,
                                "`!tt orders` - list open orders; `!tt order <order_id>` - show details; `!tt cancel <order_id>` - cancel the order; `!tt replace <order_id> <price>` - replace order price",
                            )
                            return
                        if topic == "trade" or topic in ("buy", "sell"):
                            await self._send_dm_or_warn(
                                ctx,
                                "Trading: `!tt buy <symbol> <tp_ticks> [qty] [live]` - market+TP; `!tt sell ...`",
                            )
                            return
                        if topic == "auth":
                            await self._send_dm_or_warn(
                                ctx,
                                "Auth: `!tt auth status|refresh <token>|dryrun <true|false>|sandbox <true|false>|default set <account_id>`",
                            )
                            return
                    help_msg = (
                        "**TastyTrade Bot Commands (!tt):**\n\n"
                        "**Account Management:**\n"
                        "• `!tt` - Display account summary (balances and basic info)\n"
                        "• `!tt status` - Show trading status (options level, frozen, margin call, etc.)\n"
                        "• `!tt account <account_id>` - Switch to a different account\n"
                        "• `!tt auth status` - Show TastyTrade auth/session status\n"
                        "• `!tt auth refresh <token>` - Update refresh token and reinitialize session\n"
                        "• `!tt auth dryrun <true|false>` - Toggle dry-run mode for orders\n"
                        "• `!tt auth sandbox <true|false>` - Toggle sandbox (test) environment\n"
                        "• `!tt auth default set <account_id>` - Set default TastyTrade account\n"
                        "• `!tt auth get-refresh` - Instructions to fetch a refresh token\n"
                        "• `!tt accounts` - List all accounts\n"
                        "• `!tt pos` - Show current positions\n"
                        "• `!tt future` / `!tt futures` - List available futures\n"
                        "• `!tt orders` / `!tt list-orders` - Show open orders\n\n"
                        "**Trading Commands:**\n"
                        "• `!tt b/buy <symbol> <tp_ticks> [quantity=1]` - Place market buy order with take profit\n"
                        "• `!tt s/sell <symbol> <tp_ticks> [quantity=1]` - Place market sell order with take profit\n"
                        "  - `<symbol>`: Stock ticker (e.g., AAPL) or future (e.g., /NQ:XCME)\n"
                        "  - `<tp_ticks>`: Take profit distance in ticks (0.25 for stocks, varies for futures)\n"
                        "  - `[quantity]`: Number of shares/contracts (default 1)\n\n"
                        "**Examples:**\n"
                        "• `!tt` - Show my account balances\n"
                        "• `!tt status` - Check if my account is frozen or in margin call\n"
                        "• `!tt account 123456789` - Switch to account 123456789\n"
                        "• `!tt buy AAPL 4` - Buy 1 share of AAPL, TP 1 point above\n"
                        "• `!tt sell /NQ:XCME 20 5` - Sell 5 NQ contracts, TP 5 points below\n\n"
                        "**Order Management:**\n"
                        "• `!tt cancel <order_id>` - Cancel an order by ID\n"
                        "• `!tt replace <order_id> <price>` - Replace existing order price (limit)\n"
                        "• `!tt order <order_id>` - Show order details\n"
                        "• `!tt close <symbol> [quantity]` - Close existing position with market order\n\n"
                        "**Notes:**\n"
                        "• All commands require privileged access\n"
                        "• Trading commands place both market and limit TP orders\n"
                        "• Use in status channel or DM for privileged users"
                    )
                    await self._send_dm_or_warn(ctx, help_msg)
                    return
                # Trading commands
                if subcommand in ["b", "s", "buy", "sell"] and len(args) >= 3:
                    action = "BUY" if subcommand in ["b", "buy"] else "SELL"
                    symbol = args[1].upper()
                    try:
                        tp_ticks = float(args[2])
                    except ValueError:
                        await self._send_dm_or_warn(ctx, "Invalid TP ticks.")
                        return
                    quantity = 1
                    if len(args) >= 4:
                        try:
                            quantity = int(args[3])
                        except ValueError:
                            await self._send_dm_or_warn(ctx, "Invalid quantity.")
                            return
                    # dry-run flag: optional 5th arg 'live' to execute against production.
                    # Default to configured client setting (from .env) to avoid forcing manual flag.
                    dry_run = True
                    if len(args) >= 5:
                        if args[4].lower() in ("live", "execute", "send"):
                            dry_run = False
                        elif args[4].lower() in ("dry", "dryrun", "test"):
                            dry_run = True
                    else:
                        # Default to client-level dry_run if client configured
                        try:
                            dry_run = bool(
                                getattr(self.tastytrade_client, "_dry_run", True)
                            )
                        except Exception:
                            dry_run = True
                    market_price = None
                    # Determine whether to try and fetch a market snapshot for futures
                    # Accept leading slash (/NQZ5), product code (NQ), or explicit contract (NQZ25)
                    try:
                        import re

                        lookup_feed = None
                        s_up = symbol.upper() if symbol else ""
                        # Explicit product mapping
                        if s_up in self.ticker_aliases:
                            lookup_feed = self.ticker_aliases.get(s_up)
                        # Leading slash symbol like '/NQZ5' or '/NQZ25:XCME'
                        elif s_up.startswith("/"):
                            prod_match = re.match(
                                r"^/((?P<prod>NQ|MNQ|ES|MES|RTY|YM))", s_up
                            )
                            if prod_match:
                                lookup_feed = self.ticker_aliases.get(
                                    prod_match.group("prod")
                                )
                        # Explicit contract 'NQZ25' or 'NQZ25:XCME'
                        else:
                            prod_match = re.match(
                                r"^(?P<prod>NQ|MNQ|ES|MES|RTY|YM)", s_up
                            )
                            if prod_match:
                                lookup_feed = self.ticker_aliases.get(
                                    prod_match.group("prod")
                                )

                        if lookup_feed:
                            snap = await self.get_gex_snapshot(lookup_feed)
                            if snap and isinstance(
                                snap.get("spot_price"), (int, float)
                            ):
                                market_price = float(snap.get("spot_price"))
                    except Exception:
                        market_price = None
                    try:
                        # Pass dry_run as keyword to avoid being treated as tick_size
                        result = await asyncio.to_thread(
                            self.tastytrade_client.place_market_order_with_tp,
                            symbol,
                            action,
                            quantity,
                            tp_ticks,
                            dry_run=dry_run,
                            market_price=market_price,
                        )
                        await self._send_dm_or_warn(ctx, result)
                    except TastytradeAuthError as e:
                        await self._send_dm_or_warn(
                            ctx,
                            f"TastyTrade auth invalid: {e}. Update the refresh token or run `!tt auth refresh <token>`.",
                        )
                    except Exception as e:
                        await self._send_dm_or_warn(ctx, f"Order failed: {e}")
                    return
                if subcommand == "cancel" and len(args) >= 2:
                    if not await self._ensure_privileged(ctx):
                        return
                    order_id = args[1]
                    try:
                        resp = await asyncio.to_thread(
                            self.tastytrade_client.cancel_order, order_id
                        )
                        await self._send_dm_or_warn(ctx, f"Cancel successful: {resp}")
                    except TastytradeAuthError as exc:
                        await self._send_dm_or_warn(
                            ctx, f"TastyTrade auth invalid: {exc}"
                        )
                    except Exception as exc:
                        await self._send_dm_or_warn(ctx, f"Failed to cancel: {exc}")
                    return
                if subcommand == "replace" and len(args) >= 3:
                    if not await self._ensure_privileged(ctx):
                        return
                    order_id = args[1]
                    try:
                        price = float(args[2])
                    except Exception:
                        await self._send_dm_or_warn(ctx, "Invalid price for replace")
                        return
                    try:
                        result = await asyncio.to_thread(
                            self.tastytrade_client.replace_order, order_id, price=price
                        )
                        await self._send_dm_or_warn(ctx, f"Replaced order: {result}")
                    except TastytradeAuthError as exc:
                        await self._send_dm_or_warn(
                            ctx, f"TastyTrade auth invalid: {exc}"
                        )
                    except Exception as exc:
                        await self._send_dm_or_warn(
                            ctx, f"Failed to replace order: {exc}"
                        )
                    return
                if subcommand == "order" and len(args) >= 2:
                    order_id = args[1]
                    try:
                        data = await asyncio.to_thread(
                            self.tastytrade_client.get_order, order_id
                        )
                        await self._send_dm_or_warn(ctx, f"Order {order_id}: {data}")
                    except TastytradeAuthError as exc:
                        await self._send_dm_or_warn(
                            ctx, f"TastyTrade auth invalid: {exc}"
                        )
                    except Exception as exc:
                        await self._send_dm_or_warn(
                            ctx, f"Failed to fetch order: {exc}"
                        )
                    return
                if subcommand == "close" and len(args) >= 2:
                    # close a position: close <symbol> [quantity]
                    if not await self._ensure_privileged(ctx):
                        return
                    symbol = args[1]
                    qty = 1
                    if len(args) >= 3:
                        try:
                            qty = int(args[2])
                        except Exception:
                            await self._send_dm_or_warn(ctx, "Invalid quantity")
                            return
                    # Build an order opposite to current pos to close
                    # For simplicity, submit a market order in reverse direction
                    try:
                        # Determine direction based on current positions
                        positions = await asyncio.to_thread(
                            self.tastytrade_client.get_positions
                        )
                        pos_map = {p.get("symbol"): p for p in positions}
                        sym = symbol.upper()
                        pos = pos_map.get(sym) or pos_map.get(sym.lstrip("/"))
                        if not pos:
                            await self._send_dm_or_warn(
                                ctx, f"No position found for {symbol}"
                            )
                            return
                        current_qty = int(pos.get("quantity") or 0)
                        if current_qty == 0:
                            await self._send_dm_or_warn(
                                ctx, f"No position to close for {symbol}"
                            )
                            return
                        # Determine action to close: if position is positive (long), sell to close
                        action = "SELL" if current_qty > 0 else "BUY"
                        try:
                            result = await asyncio.to_thread(
                                self.tastytrade_client.place_market_order_with_tp,
                                symbol,
                                action,
                                qty,
                                0,
                                dry_run=dry_run,
                            )
                            await self._send_dm_or_warn(ctx, result)
                        except TastytradeAuthError as exc:
                            await self._send_dm_or_warn(
                                ctx,
                                f"TastyTrade auth invalid: {exc}. Update the refresh token or run `!tt auth refresh <token>`.",
                            )
                        except Exception as exc:
                            await self._send_dm_or_warn(
                                ctx, f"Failed to close position: {exc}"
                            )
                    except Exception as exc:
                        await self._send_dm_or_warn(
                            ctx, f"Failed to close position: {exc}"
                        )
                    return
                if subcommand == "accounts":
                    accounts = await asyncio.to_thread(
                        self.tastytrade_client.get_accounts
                    )
                    if accounts:
                        msg = "**TastyTrade Accounts:**\n" + "\n".join(
                            [
                                f"• {acc.get('account-number', 'N/A')}: {acc.get('description', 'N/A')}"
                                for acc in accounts
                            ]
                        )
                    else:
                        msg = "No accounts found."
                    await self._send_dm_or_warn(ctx, msg)
                    return
                if subcommand == "pos":
                    positions = await asyncio.to_thread(
                        self.tastytrade_client.get_positions
                    )
                    if positions:
                        msg = "**Positions:**\n" + "\n".join(
                            [
                                f"• {pos.get('symbol', 'N/A')}: {pos.get('quantity', 0)} @ {pos.get('average-price', 'N/A')}"
                                for pos in positions
                            ]
                        )
                    else:
                        msg = "No positions."
                    await self._send_dm_or_warn(ctx, msg)
                    return
                if subcommand == "future" or subcommand == "futures":
                    # Allow optional product codes after the subcommand, e.g., `!tt futures NQ ES`
                    product_codes = []
                    if len(args) >= 2:
                        for token in args[1:]:
                            if isinstance(token, str) and token.strip():
                                t = token.strip().upper()
                                # Accept product code tokens like 'NQ' or 'ES'
                                if t in ("NQ", "MNQ", "ES", "MES", "RTY", "YM"):
                                    product_codes.append(t)
                    try:
                        if product_codes:
                            futures = await asyncio.to_thread(
                                self.tastytrade_client.list_futures, product_codes
                            )
                        else:
                            futures = await asyncio.to_thread(
                                self.tastytrade_client.get_futures_list
                            )
                        if not futures:
                            msg = "No futures found."
                        else:
                            # Format each contract with symbol, streamer_symbol, exp date and tradeable flag
                            lines = []
                            for fut in futures:
                                sym = (
                                    fut.get("symbol")
                                    or fut.get("streamer_symbol")
                                    or "N/A"
                                )
                                sstream = fut.get("streamer_symbol") or ""
                                exp = fut.get("expiration_date") or ""
                                tradeable = fut.get("is_tradeable")
                                trade_txt = (
                                    "tradeable"
                                    if tradeable
                                    else "not tradeable"
                                    if tradeable is not None
                                    else ""
                                )
                                desc = (
                                    fut.get("description")
                                    or fut.get("product_code")
                                    or ""
                                )
                                parts = [f"{sym}"]
                                if sstream:
                                    parts.append(f"({sstream})")
                                if exp:
                                    parts.append(f"exp={exp}")
                                if trade_txt:
                                    parts.append(trade_txt)
                                if desc:
                                    parts.append(f"- {desc}")
                                lines.append(" ".join(parts))
                            msg = "**Futures:**\n" + "\n".join(
                                [f"• {line}" for line in lines]
                            )
                    except Exception as exc:
                        msg = f"Failed to fetch futures list: {exc}"
                    await self._send_dm_or_warn(ctx, msg)
                    return
                if subcommand == "auth" and len(args) >= 2:
                    # auth subcommands: refresh, status, dryrun, default
                    act = args[1].lower()
                    if act == "refresh" and len(args) >= 3:
                        if not await self._ensure_privileged(ctx):
                            return
                        new_token = args[2]
                        try:
                            await asyncio.to_thread(
                                self.tastytrade_client.set_refresh_token, new_token
                            )
                            await self._send_dm_or_warn(
                                ctx,
                                "TastyTrade refresh token updated; session reinitialized.",
                            )
                        except Exception as exc:
                            await self._send_dm_or_warn(
                                ctx, f"Failed to update refresh token: {exc}"
                            )
                        return
                    if act == "status":
                        if not await self._ensure_privileged(ctx):
                            return
                        status = await asyncio.to_thread(
                            self.tastytrade_client.get_auth_status
                        )
                        if not isinstance(status, dict):
                            await self._send_dm_or_warn(
                                ctx, "Unable to fetch auth status"
                            )
                            return
                        def _format_session_exp(exp):
                            if not exp:
                                return "n/a"
                            from datetime import datetime, timezone
                            if isinstance(exp, str):
                                try:
                                    exp_dt = datetime.fromisoformat(exp)
                                except Exception:
                                    return exp
                            else:
                                exp_dt = exp
                            if exp_dt.tzinfo is None:
                                exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                            # Human readable
                            now = datetime.now(timezone.utc)
                            delta = exp_dt - now
                            days = delta.days
                            hours = delta.seconds // 3600
                            minutes = (delta.seconds % 3600) // 60
                            rel = f"in {days}d {hours}h {minutes}m" if delta.total_seconds() > 0 else f"expired {abs(days)}d {abs(hours)}h ago"
                            return f"{exp_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} ({rel})"

                        msg = (
                            f"TastyTrade Auth Status:\n"
                            f"Session Valid: {status.get('session_valid')}\n"
                            f"Active Account: {status.get('active_account')}\n"
                            f"Accounts: {', '.join(status.get('accounts') or [])}\n"
                            f"Use Sandbox: {status.get('use_sandbox')}\n"
                            f"Dry Run: {status.get('dry_run')}\n"
                            f"Session Expiration: {_format_session_exp(status.get('session_expiration'))}\n"
                            f"Refresh Token Hash: {status.get('refresh_token_hash')}\n"
                            f"Needs Reauth: {status.get('needs_reauth')}\n"
                        )
                        err = status.get("error")
                        if err:
                            msg += f"Error: {err}\n"
                        await self._send_dm_or_warn(ctx, msg)
                        return
                    if act == "dryrun" and len(args) >= 3:
                        if not await self._ensure_privileged(ctx):
                            return
                        val = args[2].lower()
                        flag = val in ("true", "1", "yes", "on", "enable", "enabled")
                        await asyncio.to_thread(
                            self.tastytrade_client.set_dry_run, flag
                        )
                        await self._send_dm_or_warn(
                            ctx, f"Set TastyTrade dry_run = {flag}"
                        )
                        return
                    if act == "default" and len(args) >= 4 and args[2].lower() == "set":
                        if not await self._ensure_privileged(ctx):
                            return
                        target = args[3]
                        success = await asyncio.to_thread(
                            self.tastytrade_client.set_active_account, target
                        )
                        msg = (
                            f"Active account set to {target}"
                            if success
                            else f"Failed to set active account to {target}"
                        )
                        await self._send_dm_or_warn(ctx, msg)
                        return
                    if act == "get-refresh":
                        if not await self._ensure_privileged(ctx):
                            return
                        await self._send_dm_or_warn(
                            ctx,
                            "To retrieve a refresh token, run `python scripts/get_tastytrade_refresh_token.py --sandbox` locally and update your .env or paste token here using `!tt auth refresh <token>`.",
                        )
                        return
                    # !tt auth refresh <token>
                    if not await self._ensure_privileged(ctx):
                        return
                    new_token = args[2]
                    try:
                        await asyncio.to_thread(
                            self.tastytrade_client.set_refresh_token, new_token
                        )
                        try:
                            # Keep backend services client in sync with updated token
                            from services.tastytrade_client import (
                                tastytrade_client as svc_tastytrade_client,
                            )

                            svc_tastytrade_client.set_refresh_token(new_token)
                        except Exception as sync_exc:
                            print(
                                f"Warning: failed to sync service tastytrade_client token: {sync_exc}"
                            )
                        await self._send_dm_or_warn(
                            ctx,
                            "TastyTrade refresh token updated; session reinitialized.",
                        )
                    except Exception as exc:
                        await self._send_dm_or_warn(
                            ctx, f"Failed to update refresh token: {exc}"
                        )
                    return
                    if act == "sandbox" and len(args) >= 3:
                        if not await self._ensure_privileged(ctx):
                            return
                        val = args[2].lower()
                        flag = val in ("true", "1", "yes", "on", "enable", "enabled")
                        try:
                            await asyncio.to_thread(
                                self.tastytrade_client.set_use_sandbox, flag
                            )
                            await self._send_dm_or_warn(
                                ctx, f"Set TastyTrade sandbox mode = {flag}"
                            )
                        except Exception as exc:
                            await self._send_dm_or_warn(
                                ctx, f"Failed to set sandbox mode: {exc}"
                            )
                        return
                if subcommand in ("orders", "list-orders", "list_orders", "list"):
                    orders = await asyncio.to_thread(self.tastytrade_client.get_orders)
                    if orders:
                        msg = "**Orders:**\n" + "\n".join(
                            [
                                f"• {ord.get('id', 'N/A')}: {ord.get('action', 'N/A')} {ord.get('quantity', 0)} {ord.get('symbol', 'N/A')} @ {ord.get('price', 'N/A')}"
                                for ord in orders
                            ]
                        )
                    else:
                        msg = "No orders."
                    await self._send_dm_or_warn(ctx, msg)
                    return
                # Default to summary
                summary = await self._fetch_tastytrade_summary()
                if summary is None:
                    await self._send_dm_or_warn(
                        ctx, "Unable to fetch TastyTrade summary."
                    )
                    return
                message = self.format_tastytrade_summary(summary)
                await self._send_dm_or_warn(ctx, message)

            self.add_command(commands.Command(_ping_cmd, name="ping"))
            self.add_command(commands.Command(_gex_cmd, name="gex"))
            self.add_command(commands.Command(_status_cmd, name="status"))
            self.add_command(commands.Command(_tastytrade_cmd, name="tastytrade"))
            self.add_command(commands.Command(_market_cmd, name="market"))
            self.add_command(commands.Command(_uwalerts_cmd, name="uw"))
            self.add_command(commands.Command(_tt_cmd, name="tt"))
            # LEFT IN PLACE FOR BACKWARDS COMPATIBILITY: register class method
            async def _allowlist_cmd(ctx, *args):
                """Admin CLI to manage the allowlist from Discord.

                Usage:
                  !allowlist list users
                  !allowlist list channels
                  !allowlist add user 12345
                  !allowlist remove channel 67890
                """
                # Use class-level command if available
                if hasattr(self, "_allowlist_cmd_impl"):
                    return await self._allowlist_cmd_impl(ctx, *args)
                if not await self._ensure_privileged(ctx):
                    return
                if not args:
                    await self._send_dm_or_warn(ctx, "Usage: !allowlist <list|add|remove> <users|channels> [id]")
                    return
                action = args[0].lower()
                if action == "list" and len(args) >= 2:
                    target = args[1].lower()
                    if target == "users":
                        users = AuthService.list_users_allowlist()
                        await self._send_dm_or_warn(ctx, "Allowlisted users:\n" + "\n".join(users))
                        return
                    if target == "channels":
                        channels = AuthService.list_channels_allowlist()
                        await self._send_dm_or_warn(ctx, "Allowlisted channels:\n" + "\n".join(channels))
                        return
                if action == "add" and len(args) >= 3:
                    kind = args[1].lower()
                    ident = args[2]
                    if kind == "user":
                        if AuthService.add_user_to_allowlist(ident):
                            await self._send_dm_or_warn(ctx, f"Added user {ident} to allowlist")
                        else:
                            await self._send_dm_or_warn(ctx, f"Failed to add user {ident}")
                        return
                    if kind == "channel":
                        if AuthService.add_channel_to_allowlist(ident):
                            await self._send_dm_or_warn(ctx, f"Added channel {ident} to allowlist")
                        else:
                            await self._send_dm_or_warn(ctx, f"Failed to add channel {ident}")
                        return
                if action == "remove" and len(args) >= 3:
                    kind = args[1].lower()
                    ident = args[2]
                    if kind == "user":
                        if AuthService.remove_user_from_allowlist(ident):
                            await self._send_dm_or_warn(ctx, f"Removed user {ident} from allowlist")
                        else:
                            await self._send_dm_or_warn(ctx, f"Failed to remove user {ident}")
                        return
                    if kind == "channel":
                        if AuthService.remove_channel_from_allowlist(ident):
                            await self._send_dm_or_warn(ctx, f"Removed channel {ident} from allowlist")
                        else:
                            await self._send_dm_or_warn(ctx, f"Failed to remove channel {ident}")
                        return
                await self._send_dm_or_warn(ctx, "Invalid command. Usage: !allowlist <list|add|remove> <users|channels> [id]")

            self.add_command(commands.Command(_allowlist_cmd, name="allowlist"))
        except Exception as e:
            # If registration fails (e.g., during import-time tests), print error for debugging
            print(f"Command registration failed: {e}")

    async def on_ready(self):
        print(f"Bot logged in as {self.user}")
        if not self._uw_listener_task:
            self._uw_listener_stop.clear()
            self._uw_listener_task = asyncio.create_task(
                self._listen_option_trade_stream()
            )
        if self._should_run_gex_feed() and not self._gex_feed_task:
            self._gex_feed_stop.clear()
            self._gex_feed_task = asyncio.create_task(self._run_gex_feed_loop())
            print("GEX feed loop started")

    async def on_message(self, message):
        if message.author == self.user:
            return
        if not self._is_allowed_channel(message.channel):
            return

        # Check for alert messages
        # Ignore replies — only first messages trigger alerts
        try:
            is_reply = getattr(message, "reference", None) is not None
        except Exception:
            is_reply = False
        if is_reply:
            return

        if message.content.startswith("Alert"):
            await self._process_alert_message(message)

        await super().on_message(message)

    async def _process_alert_message(self, message):
        """Process an alert message from Discord."""
        print(f"Processing alert message: {message.content}")

        # Check authorization for automated trades
        from services.auth_service import AuthService

        if not AuthService.verify_user_for_automated_trades(str(message.author.id)):
            print(f"User {message.author.id} not authorized for automated trades")
            await message.channel.send(
                "You are not authorized to send automated trade alerts."
            )
            return

        # Import here to avoid circular imports
        from services.automated_options_service import AutomatedOptionsService

        try:
            result = await AutomatedOptionsService(
                self.tastytrade_client
            ).process_alert(
                message.content, str(message.channel.id), str(message.author.id)
            )
        except TastytradeAuthError as exc:
            msg = str(exc)
            print(msg)
            await message.channel.send(msg)
            return

        if not result:
            print("Alert processing returned None")
            await message.channel.send("Alert processing failed or no action taken.")
            return

        print(f"Alert processed successfully: {result}")
        # Result may be a dict with order info: order_id, quantity, entry_price
        if isinstance(result, dict):
            order_id = result.get("order_id")
            qty = result.get("quantity")
            price = result.get("entry_price")
            msg = (
                f"Automated order placed: id={order_id}, qty={qty}, entry_price={price}"
            )
        else:
            msg = f"Automated order placed: {result}"
        await message.channel.send(msg)

    async def get_context(self, origin, *, cls=commands.Context):
        ctx = await super().get_context(origin, cls=cls)
        if ctx and not self._is_allowed_channel(getattr(ctx, "channel", None)):
            ctx.command = None
        return ctx

    async def close(self):
        if self._uw_listener_task:
            self._uw_listener_stop.set()
            try:
                await self._uw_listener_task
            except Exception:
                pass
            self._uw_listener_task = None
        if self._gex_feed_task:
            self._gex_feed_stop.set()
            try:
                await self._gex_feed_task
            except Exception:
                pass
            self._gex_feed_task = None
        await super().close()

    async def ping(self, ctx):
        await ctx.send("Pong!")

    def _should_run_gex_feed(self) -> bool:
        if not self.gex_feed_enabled:
            return False
        if not self.gex_feed_channel_ids and not self.gex_feed_channel_map:
            return False
        return True

    def _reset_feed_backoff(self) -> None:
        self._gex_feed_backoff_seconds = 0.0
        self._gex_feed_block_until = None

    def _record_feed_rate_limit(self, status: Optional[int]) -> None:
        if status not in (427, 429):
            return
        if self._gex_feed_backoff_seconds <= 0:
            self._gex_feed_backoff_seconds = self.gex_feed_backoff_base_seconds
        else:
            self._gex_feed_backoff_seconds = min(
                self.gex_feed_backoff_max_seconds, self._gex_feed_backoff_seconds * 2
            )
        block_for = self._gex_feed_backoff_seconds
        self._gex_feed_block_until = datetime.now(timezone.utc) + timedelta(
            seconds=block_for
        )
        print(
            f"GEX feed rate-limited (HTTP {status}); backing off for {block_for:.1f}s"
        )

    def _is_feed_blocked(self) -> bool:
        if not self._gex_feed_block_until:
            return False
        return datetime.now(timezone.utc) < self._gex_feed_block_until

    def _handle_feed_http_exception(self, exc: discord.HTTPException) -> bool:
        status = getattr(exc, "status", None)
        if status in (427, 429):
            self._record_feed_rate_limit(status)
            return True
        return False

    async def _sleep_between_feed_updates(self) -> None:
        # Use a separate edit delay so we can keep poll cadence fast without flooding Discord
        delay = self.gex_feed_edit_seconds
        if self._is_feed_blocked():
            remaining = (
                self._gex_feed_block_until - datetime.now(timezone.utc)
            ).total_seconds()
            delay = max(delay, remaining)
        elif self._gex_feed_backoff_seconds:
            delay = max(delay, self._gex_feed_backoff_seconds)
        await asyncio.sleep(max(0.05, delay))

    async def _get_supported_tickers(self) -> dict:
        cache_key = "gexbot:tickers:categories"
        try:
            raw = await asyncio.to_thread(self.redis_client.get, cache_key)
            if raw:
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8")
                return json.loads(raw)
        except Exception:
            pass

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("https://api.gexbot.com/tickers")
                data = resp.json()
                if resp.status_code == 200 and isinstance(data, dict):
                    try:
                        await asyncio.to_thread(
                            self.redis_client.setex, cache_key, 86400, json.dumps(data)
                        )
                    except Exception:
                        pass
                    return data
        except Exception:
            pass
        return {}

    def _format_supported_tickers_message(
        self, tickers_map: dict, alias_map: Optional[dict] = None
    ) -> str:
        if not isinstance(tickers_map, dict) or not tickers_map:
            return "Unsupported symbol. Unable to fetch supported symbol list."
        msg = "Unsupported symbol. Supported tickers by category:\n"
        for category, symbols in tickers_map.items():
            if isinstance(symbols, list) and symbols:
                msg += f"**{category.title()}**: {', '.join(sorted(symbols))}\n"
        if alias_map:
            alias_texts = []
            for key, value in alias_map.items():
                alias_texts.append(f"{key}/{key.lower()} -> {value}")
            if alias_texts:
                msg += "\nAliases: " + "; ".join(alias_texts)
        return msg

    # Dynamic enrollment removed from bot; no write path for gexbot:symbols:dynamic

    async def _run_gex_feed_loop(self):
        try:
            while not self._gex_feed_stop.is_set():
                print("GEX feed loop tick")
                if not self._should_run_gex_feed():
                    await asyncio.sleep(60)
                    continue
                now = datetime.now(self.display_zone)
                start = now.replace(hour=9, minute=35, second=0, microsecond=0)
                end = now.replace(hour=16, minute=0, second=0, microsecond=0)
                session_end = end
                if self.gex_feed_force_window:
                    session_end = now + timedelta(hours=8)
                else:
                    if now >= end:
                        await self._sleep_with_stop(
                            (start + timedelta(days=1) - now).total_seconds()
                        )
                        continue
                    if now < start:
                        await self._sleep_with_stop((start - now).total_seconds())
                        continue
                symbols = self.gex_feed_symbols or (self.gex_feed_symbol,)
                tasks = []
                for sym in symbols:
                    sym_norm = (sym or self.gex_feed_symbol).upper()
                    channels = await self._resolve_feed_channels(sym_norm)
                    if not channels:
                        continue
                    tasks.append(
                        asyncio.create_task(
                            self._run_gex_feed_session(sym_norm, channels, session_end)
                        )
                    )
                if not tasks:
                    await asyncio.sleep(120)
                    continue
                await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"GEX feed loop crashed: {exc}")

    async def _sleep_with_stop(self, seconds: float) -> None:
        remaining = max(0.0, seconds)
        while remaining > 0 and not self._gex_feed_stop.is_set():
            chunk = min(30.0, remaining)
            await asyncio.sleep(chunk)
            remaining -= chunk

    async def _resolve_feed_channels(
        self, symbol: Optional[str] = None
    ) -> List[discord.abc.Messageable]:
        channels = []
        ids = ()
        if symbol:
            ids = self.gex_feed_channel_map.get(symbol.upper(), ())
        if not ids:
            ids = self.gex_feed_channel_ids or ()
        for cid in ids:
            channel = self.get_channel(cid)
            if channel is None:
                try:
                    channel = await self.fetch_channel(cid)
                except Exception as exc:
                    print(f"Failed to fetch channel {cid}: {exc}")
                    continue
            if hasattr(channel, "send"):
                channels.append(channel)
        if not channels:
            print(
                f"GEX feed: no channels resolved from IDs {ids} for {symbol or 'default'}"
            )
        return channels

    async def _run_gex_feed_session(
        self,
        symbol: str,
        channels: List[discord.abc.Messageable],
        session_end: datetime,
    ) -> None:
        tracker = RollingWindowTracker(window_seconds=self.gex_feed_window_seconds)
        messages: Dict[int, discord.Message] = {}
        pubsub = await self._create_gex_pubsub()
        if not pubsub:
            print("GEX feed: pubsub unavailable, skipping session")
            return
        while not self._gex_feed_stop.is_set():
            now = datetime.now(self.display_zone)
            if now >= session_end:
                break
            data = await self._next_pubsub_snapshot(pubsub, symbol)
            if not data:
                # No message yet; small sleep to avoid tight loop
                await asyncio.sleep(0.01)
                continue
            delta_map = self._build_feed_delta_map(tracker, data, now)
            content = self.format_gex_short(
                data,
                include_time=True,
                time_format="%I:%M:%S %p %Z",
                delta_block=delta_map,
            )
            if not messages:
                messages = await self._post_feed_messages(channels, content)
                await self._gex_feed_metrics.record_update(
                    now, delta_map, refresh=True
                )
            else:
                await self._edit_feed_messages(messages, content)
                await self._gex_feed_metrics.record_update(
                    now, delta_map, refresh=False
                )
            # Short sleep to avoid hammering Redis/pubsub when messages flood
            await asyncio.sleep(0.01)
        if pubsub:
            try:
                await asyncio.to_thread(pubsub.close)
            except Exception:
                pass

    async def _create_gex_pubsub(self):
        try:
            pubsub = self.redis_client.pubsub(ignore_subscribe_messages=True)
            pubsub.subscribe(self.gex_snapshot_channel)
            return pubsub
        except Exception as exc:
            print(
                f"GEX feed: failed to subscribe to {self.gex_snapshot_channel}: {exc}"
            )
            return None

    async def _next_pubsub_snapshot(self, pubsub, symbol: str):
        if not pubsub:
            return None
        latest = None
        try:
            while True:
                message = await asyncio.to_thread(pubsub.get_message, timeout=0.0)
                if not message or message.get("type") != "message":
                    break
                raw = message.get("data")
                if raw is None:
                    continue
                try:
                    payload = (
                        json.loads(raw.decode())
                        if isinstance(raw, (bytes, bytearray))
                        else json.loads(raw)
                        if isinstance(raw, str)
                        else raw
                    )
                except Exception:
                    continue
                normalized_symbol = (symbol or "NQ_NDX").upper()
                ticker = self.ticker_aliases.get(normalized_symbol, normalized_symbol)
                if (payload.get("symbol") or "").upper() != ticker:
                    continue
                normalized = self._normalize_snapshot_payload(payload, ticker)
                if not normalized:
                    continue
                now = datetime.now(timezone.utc)
                ts = normalized.get("timestamp") or now
                if isinstance(ts, datetime):
                    age = (
                        now - (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc))
                    ).total_seconds()
                else:
                    age = 9999
                normalized["display_symbol"] = normalized_symbol
                normalized["_source"] = "redis-pubsub"
                if age <= 5:
                    normalized["_freshness"] = "current"
                elif age <= 30:
                    normalized["_freshness"] = "stale"
                else:
                    normalized["_freshness"] = "incomplete"
                latest = normalized
        except Exception as exc:
            print(f"GEX feed pubsub error: {exc}")
            return None
        return latest

    async def _post_feed_messages(
        self, channels: List[discord.abc.Messageable], content: str
    ) -> Dict[int, discord.Message]:
        messages: Dict[int, discord.Message] = {}
        if self._is_feed_blocked():
            return messages
        rate_limited = False
        for channel in channels:
            try:
                msg = await channel.send(content)
                print(
                    f"GEX feed: posted new message to channel {getattr(channel, 'id', 'unknown')}"
                )
            except discord.HTTPException as exc:
                if self._handle_feed_http_exception(exc):
                    rate_limited = True
                print(
                    f"Failed to send GEX feed message to {getattr(channel, 'id', 'unknown')}: {exc}"
                )
                continue
            except Exception as exc:
                print(
                    f"Failed to send GEX feed message to {getattr(channel, 'id', 'unknown')}: {exc}"
                )
                continue
            messages[getattr(channel, "id", id(channel))] = msg
        if not rate_limited:
            self._reset_feed_backoff()
        return messages

    async def _edit_feed_messages(
        self, messages: Dict[int, discord.Message], content: str
    ) -> None:
        stale_ids = []
        if self._is_feed_blocked():
            return
        rate_limited = False
        for cid, msg in messages.items():
            try:
                await msg.edit(content=content)
            except discord.HTTPException as exc:
                if exc.status == 404:
                    stale_ids.append(cid)
                else:
                    if self._handle_feed_http_exception(exc):
                        rate_limited = True
                    else:
                        print(f"Failed to edit GEX feed message {cid}: {exc}")
            except Exception as exc:
                print(f"Failed to edit GEX feed message {cid}: {exc}")
        for cid in stale_ids:
            messages.pop(cid, None)
        if not rate_limited:
            self._reset_feed_backoff()

    async def _delete_feed_messages(self, messages: Dict[int, discord.Message]) -> None:
        target = list(messages.items())
        for cid, msg in target:
            try:
                await msg.delete()
            except Exception:
                pass
            finally:
                messages.pop(cid, None)

    def _build_feed_delta_map(
        self, tracker: "RollingWindowTracker", data: dict, now: datetime
    ) -> Dict[str, "MetricSnapshot"]:
        mapping: Dict[str, MetricSnapshot] = {}
        mapping["spot"] = tracker.update("spot", data.get("spot_price"), now)
        mapping["zero_gamma"] = tracker.update(
            "zero_gamma", data.get("zero_gamma"), now
        )
        mapping["call_wall"] = tracker.update(
            "call_wall", data.get("major_pos_vol"), now
        )
        mapping["put_wall"] = tracker.update("put_wall", data.get("major_neg_vol"), now)
        mapping["net_gex"] = tracker.update("net_gex", data.get("net_gex"), now)
        mapping["scaled_gamma"] = tracker.update(
            "scaled_gamma", data.get("sum_gex_oi"), now
        )
        max_entry = self._extract_current_maxchange_entry(data)
        max_delta = (
            max_entry[1]
            if max_entry and isinstance(max_entry[1], (int, float))
            else None
        )
        mapping["maxchange"] = tracker.update("maxchange", max_delta, now)
        return mapping

    async def gex(self, ctx, *args):
        """Get GEX snapshot for a symbol. Uses DuckDB first then GEXBot API fallback."""
        symbol, show_full = self._resolve_gex_request(args)
        data = await self.get_gex_data(symbol)
        if data:
            response = (
                self.format_gex(data) if show_full else self.format_gex_short(data)
            )
            await ctx.send(response)
        else:
            await ctx.send("GEX data not available")

    async def status(self, ctx):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8877/status")
                if response.status_code == 200:
                    status_data = response.json()
                    # Format the status nicely
                    formatted = self.format_status(status_data)
                    await ctx.send(formatted)
                else:
                    await ctx.send(f"Failed to fetch status: {response.status_code}")
        except Exception as e:
            await ctx.send(f"Error fetching status: {e}")

    async def tastytrade(self, ctx):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8877/status")
                if response.status_code == 200:
                    status_data = response.json()
                    tt_status = status_data.get("tastytrade_streamer", {})
                    formatted = "**TastyTrade Status:**\n"
                    formatted += f"Running: {tt_status.get('running', False)}\n"
                    formatted += f"Trade Samples: {tt_status.get('trade_samples', 0)}\n"
                    formatted += (
                        f"Last Trade: {tt_status.get('last_trade_ts', 'N/A')}\n"
                    )
                    formatted += f"Depth Samples: {tt_status.get('depth_samples', 0)}\n"
                    formatted += (
                        f"Last Depth: {tt_status.get('last_depth_ts', 'N/A')}\n"
                    )
                    await ctx.send(formatted)
                else:
                    await ctx.send(f"Failed to fetch status: {response.status_code}")
        except Exception as e:
            await ctx.send(f"Error fetching TastyTrade status: {e}")

    async def get_gex_data(self, symbol: str):
        """Async dual-source retrieval: Redis cache -> DuckDB -> GEXBot API fallback.

        Freshness classification (seconds):
        - current: <=5
        - stale: 5-30
        - incomplete: >30
        """
        display_symbol = (symbol or "QQQ").upper()
        ticker = self.ticker_aliases.get(display_symbol, display_symbol)
        # No transient cache key: use canonical snapshot key only
        snapshot_key = f"{self.redis_snapshot_prefix}{ticker.upper()}"

        async def finalize(data: dict) -> dict:
            data.setdefault("display_symbol", display_symbol)
            # Only use snapshot_key for enrichment; transient cache key removed
            await self._populate_wall_ladders(data, ticker, None, snapshot_key)
            return data

        # Prefer the canonical snapshot key (written by poller/pipeline)
        try:
            snapshot = await asyncio.to_thread(self.redis_client.get, snapshot_key)
            if snapshot:
                normalized = self._normalize_snapshot_payload(snapshot, ticker)
                if normalized:
                    now = datetime.now(timezone.utc)
                    ts = normalized.get("timestamp") or now
                    if isinstance(ts, datetime):
                        age = (
                            now - (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc))
                        ).total_seconds()
                    else:
                        age = 9999
                    normalized["display_symbol"] = display_symbol
                    if age <= 30:
                        normalized["_freshness"] = "current" if age <= 5 else "stale"
                        normalized["_source"] = "redis-snapshot"
                        await asyncio.to_thread(
                            self.redis_client.setex,
                            snapshot_key,
                            300,
                            json.dumps(normalized, default=str),
                        )
                        return await finalize(normalized)
        except Exception as e:
            print(f"Redis snapshot check failed: {e}")

        # NOTE: transient cache key removed; only canonical snapshot key is used for feed and lookups

        # 1b) Try canonical snapshot key populated by the data pipeline
        try:
            snapshot = await asyncio.to_thread(self.redis_client.get, snapshot_key)
            if snapshot:
                normalized = self._normalize_snapshot_payload(snapshot, ticker)
                if normalized:
                    now = datetime.now(timezone.utc)
                    ts = normalized.get("timestamp") or now
                    if isinstance(ts, datetime):
                        age = (
                            now - (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc))
                        ).total_seconds()
                    else:
                        age = 9999
                    normalized["display_symbol"] = display_symbol
                    if age <= 5:
                        normalized["_freshness"] = "current"
                        # snapshot returned from the pipeline; mark as redis snapshot
                        normalized["_source"] = "redis-snapshot"
                        await asyncio.to_thread(
                            self.redis_client.setex,
                            snapshot_key,
                            300,
                            json.dumps(normalized, default=str),
                        )
                        return await finalize(normalized)
                    if age <= 30:
                        normalized["_freshness"] = "stale"
                        normalized["_source"] = "redis-snapshot"
                        await asyncio.to_thread(
                            self.redis_client.setex,
                            snapshot_key,
                            300,
                            json.dumps(normalized, default=str),
                        )
                        asyncio.create_task(
                            self._refresh_gex_from_api(
                                ticker, snapshot_key, display_symbol
                            )
                        )
                        return await finalize(normalized)
                    normalized["_freshness"] = "incomplete"
                    normalized["_source"] = "redis-snapshot"
                    await asyncio.to_thread(
                        self.redis_client.setex,
                        snapshot_key,
                        300,
                        json.dumps(normalized, default=str),
                    )
                    return await finalize(normalized)
        except Exception as e:
            print(f"Snapshot redis check failed for {snapshot_key}: {e}")

        # 2) Query local DuckDB snapshot before resorting to live API
        try:

            def query_db():
                import duckdb

                q = f"""
                SELECT timestamp, ticker, spot_price, zero_gamma, net_gex,
                       sum_gex_vol, delta_risk_reversal, min_dte, sec_min_dte,
                       major_pos_vol, major_neg_vol, major_pos_oi, major_neg_oi, sum_gex_oi, max_priors
                FROM gex_snapshots
                WHERE ticker = '{ticker}'
                ORDER BY timestamp DESC
                LIMIT 1
                """
                conn = None
                try:
                    # Try read-only connection first (may not be supported on older duckdb)
                    try:
                        conn = duckdb.connect(self.duckdb_path, read_only=True)
                    except TypeError:
                        # read_only kwarg not supported; fall back to normal connect
                        conn = duckdb.connect(self.duckdb_path)
                    except Exception:
                        # Other connection errors
                        conn = None

                    if conn:
                        res = conn.execute(q).fetchone()
                    else:
                        res = None
                except Exception:
                    res = None
                finally:
                    try:
                        if conn:
                            conn.close()
                    except Exception:
                        pass
                return res

            result = await asyncio.to_thread(query_db)
            if result:
                columns = [
                    "timestamp",
                    "ticker",
                    "spot_price",
                    "zero_gamma",
                    "net_gex",
                    "sum_gex_vol",
                    "delta_risk_reversal",
                    "min_dte",
                    "sec_min_dte",
                    "major_pos_vol",
                    "major_neg_vol",
                    "major_pos_oi",
                    "major_neg_oi",
                    "sum_gex_oi",
                    "max_priors",
                ]
                data = dict(zip(columns, result))
                data["display_symbol"] = display_symbol
                data["_source"] = "DB"
                # Normalize timestamp
                ts = data.get("timestamp")
                if isinstance(ts, str):
                    try:
                        data["timestamp"] = datetime.fromisoformat(ts)
                    except Exception:
                        data["timestamp"] = datetime.fromtimestamp(
                            float(ts)
                        ).astimezone(timezone.utc)
                elif isinstance(ts, (int, float)):
                    data["timestamp"] = datetime.fromtimestamp(float(ts)).astimezone(
                        timezone.utc
                    )

                # Parse max_priors
                if isinstance(data.get("max_priors"), str):
                    try:
                        data["max_priors"] = json.loads(data["max_priors"])
                    except Exception:
                        data["max_priors"] = []

                # determine freshness
                now = datetime.now(timezone.utc)
                rec_ts = data.get("timestamp") or now
                age = (
                    now
                    - (
                        rec_ts
                        if isinstance(rec_ts, datetime) and rec_ts.tzinfo
                        else rec_ts.replace(tzinfo=timezone.utc)
                    )
                ).total_seconds()
                if age <= 5:
                    data["_freshness"] = "current"
                elif age <= 30:
                    data["_freshness"] = "stale"
                else:
                    data["_freshness"] = "incomplete"

                # cache result for future by writing canonical snapshot key
                await asyncio.to_thread(
                    self.redis_client.setex,
                    snapshot_key,
                    300,
                    json.dumps(data, default=str),
                )
                return await finalize(data)
        except Exception as e:
            print(f"DuckDB query failed: {e}")

        # 3) Finally, poll the live API as a last resort
        try:
            if not self.gex_api_enabled:
                return None
            api_data = await self._poll_gexbot_api(ticker)
            if api_data:
                api_data["_freshness"] = "current"
                api_data["_source"] = "API"
                api_data["display_symbol"] = display_symbol
                try:
                    await asyncio.to_thread(
                        self.redis_client.setex,
                        snapshot_key,
                        300,
                        json.dumps(api_data, default=str),
                    )
                except Exception:
                    pass
                return await finalize(api_data)
        except Exception as e:
            print(f"API fetch failed: {e}")

        return None

    async def _poll_gexbot_api(self, ticker: str):
        """Poll the zero endpoint only, limiting outbound requests."""
        if not self.gex_api_enabled:
            return None
        api_key = os.getenv("GEXBOT_API_KEY", "XXXXXXXXXXXX")
        base = "https://api.gexbot.com"
        zero_url = f"{base}/{ticker}/classic/zero?key={api_key}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            responses = {"zero": None}
            try:
                resp = await client.get(zero_url)
                if resp.status_code == 200:
                    responses["zero"] = resp.json()
            except Exception:
                responses["zero"] = None

        # Normalize into expected schema
        now = datetime.now(timezone.utc)
        data = {
            "timestamp": now,
            "ticker": ticker,
            "spot_price": None,
            "zero_gamma": None,
            "net_gex": None,
            "major_pos_vol": None,
            "major_neg_vol": None,
            "major_pos_oi": None,
            "major_neg_oi": None,
            "sum_gex_oi": None,
            "max_priors": [],
            "strikes": [],
        }

        z = responses.get("zero")
        if z:
            # common fields
            data["spot_price"] = (
                z.get("spot_price") or z.get("price") or data["spot_price"]
            )
            data["zero_gamma"] = (
                z.get("zero_gamma") or z.get("zero_gamma_vol") or data["zero_gamma"]
            )
            data["net_gex"] = z.get("net_gex") or z.get("sum_gex") or data["net_gex"]
            data["major_pos_vol"] = z.get("major_pos_vol") or data["major_pos_vol"]
            data["major_neg_vol"] = z.get("major_neg_vol") or data["major_neg_vol"]
            data["major_pos_oi"] = z.get("major_pos_oi") or data["major_pos_oi"]
            data["major_neg_oi"] = z.get("major_neg_oi") or data["major_neg_oi"]
            data["sum_gex_oi"] = (
                z.get("sum_gex_oi") or z.get("net_gex_oi") or data["sum_gex_oi"]
            )
            strikes = z.get("strikes")
            if strikes:
                data["strikes"] = strikes

        return data

    async def get_gex_snapshot(self, symbol: str):
        """Fetch a GEX snapshot payload directly from the canonical redis snapshot key.

        This returns normalized payload like `get_gex_data` with `_source` set to 'redis-snapshot'.
        It intentionally avoids API or DB fallback to keep a predictable feed cadence.
        """
        if not symbol:
            return None
        display_symbol = (symbol or "QQQ").upper()
        ticker = self.ticker_aliases.get(display_symbol, display_symbol)
        snapshot_key = f"{self.redis_snapshot_prefix}{ticker}"
        try:
            raw = await asyncio.to_thread(self.redis_client.get, snapshot_key)
        except Exception as e:
            print(f"Snapshot redis check failed for {snapshot_key}: {e}")
            return None
        if not raw:
            return None
        normalized = self._normalize_snapshot_payload(raw, ticker)
        if not normalized:
            return None
        now = datetime.now(timezone.utc)
        ts = normalized.get("timestamp") or now
        if isinstance(ts, datetime):
            age = (
                now - (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc))
            ).total_seconds()
        else:
            age = 9999
        normalized["display_symbol"] = display_symbol
        normalized["_source"] = "redis-snapshot"
        if age <= 5:
            normalized["_freshness"] = "current"
        elif age <= 30:
            normalized["_freshness"] = "stale"
        else:
            normalized["_freshness"] = "incomplete"
        try:
            await self._populate_wall_ladders(normalized, ticker, None, snapshot_key)
        except Exception:
            pass
        return normalized

    async def _refresh_gex_from_api(
        self, ticker: str, snapshot_key: str, display_symbol: Optional[str] = None
    ):
        """Background task to refresh cached snapshot from GEXBot API."""
        if not self.gex_api_enabled:
            return
        try:
            api_data = await self._poll_gexbot_api(ticker)
            if api_data:
                api_data["_freshness"] = "current"
                if display_symbol:
                    api_data["display_symbol"] = display_symbol
                await asyncio.to_thread(
                    self.redis_client.setex,
                    snapshot_key,
                    300,
                    json.dumps(api_data, default=str),
                )
        except Exception as e:
            print(f"Background refresh failed for {ticker}: {e}")

    def _resolve_gex_request(self, args: Tuple[str, ...]) -> Tuple[str, bool]:
        symbol = None
        show_full = False
        for arg in args:
            if not isinstance(arg, str):
                continue
            token = arg.strip()
            if not token:
                continue
            if token.lower() == "full":
                show_full = True
                continue
            if symbol is None:
                symbol = token.upper()
        return (symbol or "QQQ"), show_full

    def _normalize_snapshot_payload(self, snapshot_blob, ticker: str):
        try:
            if isinstance(snapshot_blob, (bytes, bytearray)):
                payload = json.loads(snapshot_blob.decode())
            elif isinstance(snapshot_blob, str):
                payload = json.loads(snapshot_blob)
            elif isinstance(snapshot_blob, dict):
                payload = snapshot_blob
            else:
                return None
        except json.JSONDecodeError:
            return None

        ts = payload.get("timestamp") or payload.get("ts")
        if isinstance(ts, str):
            try:
                parsed_ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                parsed_ts = None
        elif isinstance(ts, (int, float)):
            parsed_ts = datetime.fromtimestamp(
                float(ts) / (1000 if ts > 1e12 else 1), tz=timezone.utc
            )
        elif isinstance(ts, datetime):
            parsed_ts = ts
        else:
            parsed_ts = None
        if parsed_ts and parsed_ts.tzinfo is None:
            parsed_ts = parsed_ts.replace(tzinfo=timezone.utc)

        def get_first(*values):
            for value in values:
                if value not in (None, ""):
                    return value
            return None

        data = {
            "timestamp": parsed_ts,
            "ticker": ticker,
            "spot_price": get_first(
                payload.get("spot"), payload.get("spot_price"), payload.get("price")
            ),
            "zero_gamma": get_first(
                payload.get("zero_gamma"), payload.get("zero_gamma_vol")
            ),
            "net_gex": payload.get("net_gex") or payload.get("sum_gex"),
            "major_pos_vol": payload.get("major_pos_vol"),
            "major_neg_vol": payload.get("major_neg_vol"),
            "major_pos_oi": payload.get("major_pos_oi"),
            "major_neg_oi": payload.get("major_neg_oi"),
            "sum_gex_oi": payload.get("sum_gex_oi") or payload.get("net_gex_oi"),
            "max_priors": self._extract_max_priors(payload),
            "maxchange": payload.get("maxchange")
            if isinstance(payload.get("maxchange"), dict)
            else {},
            "strikes": payload.get("strikes"),
        }
        return data

    def _extract_max_priors(self, payload: dict) -> List[list]:
        maxchange = payload.get("maxchange")
        normalized: List[list] = []
        mapping = [
            ("one", 1),
            ("five", 5),
            ("ten", 10),
            ("fifteen", 15),
            ("thirty", 30),
        ]
        if isinstance(maxchange, dict):
            for key, interval in mapping:
                entry = maxchange.get(key)
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    strike, delta = entry[:2]
                    normalized.append([interval, strike, delta])
            if normalized:
                return normalized

        fallback = payload.get("max_priors") or payload.get("priors") or []
        if isinstance(fallback, list):
            for idx, item in enumerate(fallback):
                if isinstance(item, (list, tuple)):
                    if len(item) >= 3:
                        normalized.append([idx + 1, item[1], item[2]])
                    elif len(item) == 2:
                        normalized.append([idx + 1, item[0], item[1]])
        return normalized

    def _has_snapshot_coverage(self, payload: dict) -> bool:
        if not isinstance(payload, dict):
            return False
        required = [
            "spot_price",
            "major_pos_vol",
            "major_neg_vol",
            "major_pos_oi",
            "major_neg_oi",
            "sum_gex_oi",
        ]
        for field in required:
            value = payload.get(field)
            if value is None:
                return False
        max_priors = payload.get("max_priors")
        if not max_priors:
            return False
        return True

    async def _populate_wall_ladders(
        self, data: dict, ticker: str, cache_key: Optional[str], snapshot_key: str
    ) -> None:
        if not isinstance(data, dict) or not ticker:
            return
        if data.get("_wall_ladders_ready"):
            return
        summary = await self._build_wall_ladders(data, ticker, cache_key, snapshot_key)
        if summary:
            data["_wall_ladders"] = summary
        data["_wall_ladders_ready"] = True

    async def _build_wall_ladders(
        self, data: dict, ticker: str, cache_key: Optional[str], snapshot_key: str
    ) -> Optional[dict]:
        strikes = self._normalize_strike_entries(data.get("strikes"))
        source = "payload" if strikes else None
        if not strikes:
            strikes, source = await self._load_strikes_from_cache(
                ticker, cache_key, snapshot_key
            )
        if not strikes:
            ts_ms = self._timestamp_to_epoch_ms(data.get("timestamp"))
            if ts_ms is not None:
                strikes, source = await asyncio.to_thread(
                    self._query_strikes_from_db, ticker, ts_ms
                )
        if not strikes:
            return None
        summary = {
            "call": self._summarize_wall_ladder(
                data.get("major_pos_vol"), strikes, prefer_positive=True
            ),
            "put": self._summarize_wall_ladder(
                data.get("major_neg_vol"), strikes, prefer_positive=False
            ),
        }
        summary = {k: v for k, v in summary.items() if v}
        if not summary:
            return None
        if source:
            summary["source"] = source
        return summary

    async def _load_strikes_from_cache(
        self,
        ticker: str,
        cache_key: Optional[str],
        snapshot_key: str,
    ) -> Tuple[List[Tuple[float, float]], Optional[str]]:
        keys = [k for k in (cache_key, snapshot_key) if k]
        for key in keys:
            try:
                raw = await asyncio.to_thread(self.redis_client.get, key)
            except Exception:
                raw = None
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            strikes = self._normalize_strike_entries(payload.get("strikes"))
            if strikes:
                source = "redis-cache" if key == cache_key else "redis-snapshot"
                return strikes, source
        return [], None

    def _query_strikes_from_db(
        self, ticker: str, epoch_ms: int
    ) -> Tuple[List[Tuple[float, float]], Optional[str]]:
        if not ticker:
            return [], None
        query = """
            SELECT strike, gamma
            FROM gex_strikes
            WHERE ticker = ? AND timestamp = ?
            ORDER BY gamma DESC
            LIMIT 64
        """
        rows: List[Tuple[float, float]] = []
        source = None
        conn = None
        try:
            import duckdb

            try:
                conn = duckdb.connect(self.duckdb_path, read_only=True)
            except TypeError:
                conn = duckdb.connect(self.duckdb_path)
            rows = conn.execute(query, [ticker, epoch_ms]).fetchall()
            source = "duckdb"
        except Exception:
            rows = []
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
        return rows, source

    @staticmethod
    def _normalize_strike_entries(raw: Any) -> List[Tuple[float, float]]:
        normalized: List[Tuple[float, float]] = []
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                return normalized
        if isinstance(raw, list):
            for entry in raw:
                strike = None
                gamma = None
                if isinstance(entry, (list, tuple)):
                    if len(entry) >= 1:
                        try:
                            strike = float(entry[0])
                        except (TypeError, ValueError):
                            strike = None
                    if len(entry) >= 2:
                        try:
                            gamma = float(entry[1])
                        except (TypeError, ValueError):
                            gamma = None
                elif isinstance(entry, dict):
                    strike = entry.get("strike") or entry.get("strike_price")
                    gamma = entry.get("gamma") or entry.get("total_gamma")
                    try:
                        strike = float(strike)
                    except (TypeError, ValueError):
                        strike = None
                    try:
                        gamma = float(gamma)
                    except (TypeError, ValueError):
                        gamma = None
                if strike is None or gamma is None:
                    continue
                normalized.append((strike, gamma))
        return normalized

    @staticmethod
    def _timestamp_to_epoch_ms(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value if value > 1e12 else value * 1000)
        if isinstance(value, datetime):
            ts = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
            return int(ts.timestamp() * 1000)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return int(parsed.timestamp() * 1000)
        return None

    @staticmethod
    def _summarize_wall_ladder(
        major_strike: Any,
        strikes: List[Tuple[float, float]],
        *,
        prefer_positive: bool,
    ) -> Optional[dict]:
        if not strikes:
            return None
        filtered = []
        for strike, gamma in strikes:
            if not isinstance(gamma, (int, float)):
                continue
            if prefer_positive and gamma <= 0:
                continue
            if not prefer_positive and gamma >= 0:
                continue
            filtered.append((strike, gamma))
        if not filtered:
            return None
        filtered.sort(key=lambda pair: abs(pair[1]), reverse=True)
        major_value = None
        tolerance = 0.51
        if isinstance(major_strike, (int, float)):
            for strike, gamma in filtered:
                if abs(strike - major_strike) <= tolerance:
                    major_value = (strike, gamma)
                    break
        if major_value is None:
            major_value = filtered[0]
        major_gamma = major_value[1] or 0
        ladder_entries = []
        for strike, gamma in filtered:
            if abs(strike - major_value[0]) <= tolerance:
                continue
            ratio = (abs(gamma) / abs(major_gamma) * 100) if major_gamma else None
            ladder_entries.append(
                {
                    "strike": strike,
                    "gamma": gamma,
                    "pct_vs_major": ratio,
                }
            )
            if len(ladder_entries) >= 2:
                break
        return {
            "major_strike": major_value[0],
            "major_gamma": major_gamma,
            "entries": ladder_entries,
        }

    def _parse_admin_ids(self) -> set[int]:
        ids: set[int] = set()
        raw = os.getenv("DISCORD_ADMIN_USER_IDS")
        if raw:
            for part in raw.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    ids.add(int(part))
                except ValueError:
                    continue
        owner = os.getenv("DISCORD_OWNER_ID")
        if owner:
            try:
                ids.add(int(owner))
            except ValueError:
                pass
        return ids

    def _parse_admin_names(self) -> set[str]:
        raw = os.getenv("DISCORD_ADMIN_USERNAMES", "skint0552")
        names = {name.strip().lower() for name in raw.split(",") if name.strip()}
        if not names:
            names = {"skint0552"}
        return names

    def _init_tastytrade_client(self) -> Optional[TastyTradeClient]:
        creds = getattr(self.config, "tastytrade_credentials", None)
        if not creds or not creds.client_secret or not creds.refresh_token:
            return None
        default_account = creds.default_account
        use_sandbox = creds.use_sandbox_environment
        dry_run = creds.dry_run
        try:
            return TastyTradeClient(
                client_secret=creds.client_secret,
                refresh_token=creds.refresh_token,
                default_account=default_account,
                use_sandbox=use_sandbox,
                dry_run=dry_run,
            )
        except Exception as exc:
            print(f"Failed to initialize TastyTrade client: {exc}")
            return None

    def _is_privileged_user(self, ctx) -> bool:
        try:
            author_id = int(ctx.author.id)
        except Exception:
            author_id = None
        if author_id is not None and author_id in self.command_admin_ids:
            return True
        author_name = getattr(ctx.author, "name", "") or ""
        if author_name.lower() in self.command_admin_names:
            return True
        return False

    async def _ensure_privileged(self, ctx) -> bool:
        if self._is_privileged_user(ctx):
            return True
        await ctx.send("You are not authorized to use this command.")
        return False

    async def _allowlist_cmd_impl(self, ctx, *args):
        """Implementation backing for allowlist admin command."""
        if not await self._ensure_privileged(ctx):
            return
        if not args:
            await self._send_dm_or_warn(ctx, "Usage: !allowlist <list|add|remove> <users|channels> [id]")
            return
        action = args[0].lower()
        if action == "list" and len(args) >= 2:
            target = args[1].lower()
            if target == "users":
                users = AuthService.list_users_allowlist()
                await self._send_dm_or_warn(ctx, "Allowlisted users:\n" + "\n".join(users))
                return
            if target == "channels":
                channels = AuthService.list_channels_allowlist()
                await self._send_dm_or_warn(ctx, "Allowlisted channels:\n" + "\n".join(channels))
                return
        if action == "add" and len(args) >= 3:
            kind = args[1].lower()
            ident = args[2]
            if kind == "user":
                if AuthService.add_user_to_allowlist(ident):
                    await self._send_dm_or_warn(ctx, f"Added user {ident} to allowlist")
                else:
                    await self._send_dm_or_warn(ctx, f"Failed to add user {ident}")
                return
            if kind == "channel":
                if AuthService.add_channel_to_allowlist(ident):
                    await self._send_dm_or_warn(ctx, f"Added channel {ident} to allowlist")
                else:
                    await self._send_dm_or_warn(ctx, f"Failed to add channel {ident}")
                return
        if action == "remove" and len(args) >= 3:
            kind = args[1].lower()
            ident = args[2]
            if kind == "user":
                if AuthService.remove_user_from_allowlist(ident):
                    await self._send_dm_or_warn(ctx, f"Removed user {ident} from allowlist")
                else:
                    await self._send_dm_or_warn(ctx, f"Failed to remove user {ident}")
                return
            if kind == "channel":
                if AuthService.remove_channel_from_allowlist(ident):
                    await self._send_dm_or_warn(ctx, f"Removed channel {ident} from allowlist")
                else:
                    await self._send_dm_or_warn(ctx, f"Failed to remove channel {ident}")
                return
        await self._send_dm_or_warn(ctx, "Invalid command. Usage: !allowlist <list|add|remove> <users|channels> [id]")

    def _init_alert_channels(self) -> List[int]:
        specific = getattr(self.config, "uw_channel_ids", None) or ()
        if specific:
            return [cid for cid in specific if cid]
        channels: List[int] = []
        if self.status_channel_id:
            channels.append(self.status_channel_id)
        for cid in self.allowed_channel_ids:
            if cid and cid not in channels:
                channels.append(cid)
        return channels

    def _is_allowed_channel(self, channel) -> bool:
        if not self.allowed_channel_ids:
            return True
        if channel is None:
            return True
        channel_id = getattr(channel, "id", None)
        if channel_id is None:
            return True
        # Allow DMs for privileged commands
        if getattr(channel, "type", None) == discord.ChannelType.private:
            return True
        return int(channel_id) in self.allowed_channel_ids

    async def _ensure_status_channel_access(self, ctx) -> bool:
        if not self._is_privileged_user(ctx):
            await ctx.send("You are not authorized to use this command.")
            return False
        if (
            self.status_channel_id
            and getattr(ctx.channel, "id", None) != self.status_channel_id
            and getattr(ctx.channel, "type", None) != discord.ChannelType.private
        ):
            await ctx.send("This command may only be used in the status channel.")
            return False
        author_name = (getattr(ctx.author, "name", "") or "").lower()
        if author_name != self.status_command_user:
            await self._send_dm_or_warn(
                ctx, "Status commands are restricted to authorized users."
            )
            return False
        return True

    async def _listen_option_trade_stream(self):
        pubsub = self.redis_client.pubsub(ignore_subscribe_messages=True)
        try:
            pubsub.subscribe(self.uw_option_stream_channel)
        except Exception as exc:
            print(f"Failed to subscribe to UW stream: {exc}")
            return
        try:
            while not self._uw_listener_stop.is_set():
                message = await asyncio.to_thread(pubsub.get_message, timeout=1.0)
                if not message:
                    await asyncio.sleep(0.1)
                    continue
                if message.get("type") != "message":
                    continue
                data = message.get("data")
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                try:
                    payload = json.loads(data)
                except Exception as exc:
                    print(f"Failed to decode UW payload: {exc}")
                    continue
                try:
                    await self._broadcast_option_trade(payload)
                except Exception as exc:
                    print(f"Failed to broadcast UW payload: {exc}")
        finally:
            try:
                pubsub.close()
            except Exception:
                pass

    async def _broadcast_option_trade(self, payload: dict) -> None:
        if not self.option_alert_channel_ids:
            return
        message = self.format_option_trade_alert(payload)
        for channel_id in self.option_alert_channel_ids:
            if not channel_id:
                continue
            try:
                channel = self.get_channel(channel_id)
                if channel is None:
                    channel = await self.fetch_channel(channel_id)
                if channel:
                    await channel.send(message)
            except Exception as exc:
                print(f"Failed to send UW alert to {channel_id}: {exc}")

    async def _fetch_market_snapshot(self) -> Optional[dict]:
        try:
            raw = await asyncio.to_thread(
                self.redis_client.get, self.uw_market_latest_key
            )
        except Exception as exc:
            print(f"Redis error fetching market snapshot: {exc}")
            return None
        if not raw:
            try:
                raw = await asyncio.to_thread(
                    self.redis_client.lindex, self.uw_market_history_key, 0
                )
            except Exception as exc:
                print(f"Redis error fetching historical market snapshot: {exc}")
                raw = None
            if not raw:
                return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return json.loads(raw)
        except Exception as exc:
            print(f"Failed to decode market snapshot: {exc}")
            return None

    async def _fetch_option_history(self, limit: int = 5) -> List[dict]:
        try:
            entries = await asyncio.to_thread(
                self.redis_client.lrange, self.uw_option_history_key, 0, limit - 1
            )
        except Exception as exc:
            print(f"Redis error fetching option history: {exc}")
            return []
        results: List[dict] = []
        for entry in entries:
            if isinstance(entry, bytes):
                entry = entry.decode("utf-8")
            try:
                results.append(json.loads(entry))
            except Exception:
                continue
        return results

    async def _fetch_tastytrade_summary(self):
        if not self.tastytrade_client:
            return None
        try:
            return await asyncio.to_thread(self.tastytrade_client.get_account_summary)
        except Exception as exc:
            print(f"Failed to fetch TastyTrade summary: {exc}")
            return None

    async def _fetch_tastytrade_overview(self):
        if not self.tastytrade_client:
            return None
        try:
            return await asyncio.to_thread(self.tastytrade_client.get_trading_status)
        except Exception as exc:
            print(f"Failed to fetch TastyTrade overview: {exc}")
            return None

    async def _send_dm(self, user, content: str) -> bool:
        try:
            await user.send(content)
            return True
        except discord.Forbidden:
            return False
        except Exception as exc:
            print(f"Failed to DM user: {exc}")
            return False

    async def _send_dm_or_warn(self, ctx, content: str) -> None:
        if not await self._send_dm(ctx.author, content):
            await ctx.send("Unable to DM you. Check your privacy settings.")

    def _format_wall_value_line(
        self,
        label_text: str,
        volume_value,
        oi_value,
        fmt_price,
        *,
        label_width: int,
        volume_width: int,
        gap: str = "",
    ) -> str:
        vol = fmt_price(volume_value)
        oi = fmt_price(oi_value) if oi_value is not None else ""
        label = f"{label_text:<{label_width}}"
        spacer = gap if gap is not None else ""
        return f"{label}{spacer}{vol:<{volume_width}}{oi}"

    def _format_wall_short_line(
        self,
        label_text: str,
        value,
        fmt_price,
        *,
        label_width: int,
        gap: str = "",
    ) -> str:
        spacer = gap if gap is not None else ""
        return f"{label_text:<{label_width}}{spacer}{fmt_price(value)}"

    def _format_wall_line(
        self,
        data: dict,
        ladder_key: str,
        label_text: str,
        fmt_price,
        *,
        label_width: int,
        default_line: Optional[str] = None,
        gap_override: Optional[str] = None,
    ) -> str:
        ladders = data.get("_wall_ladders")
        summary = ladders.get(ladder_key) if isinstance(ladders, dict) else None
        if not summary or not summary.get("entries"):
            fallback_value = data.get(
                "major_pos_vol" if ladder_key == "call" else "major_neg_vol"
            )
            if default_line:
                return default_line
            gap = gap_override if gap_override is not None else ""
            return f"{label_text:<{label_width}}{gap}{fmt_price(fallback_value)}"
        label = f"{label_text:<{label_width}}"
        major = fmt_price(
            summary.get("major_strike")
            or (
                data.get("major_pos_vol")
                if ladder_key == "call"
                else data.get("major_neg_vol")
            )
        )
        segments = [major]
        for entry in summary.get("entries", [])[:2]:
            strike_txt = fmt_price(entry.get("strike"))
            pct = entry.get("pct_vs_major")
            pct_txt = f"{pct:.0f}%" if isinstance(pct, (int, float)) else "N/A%"
            segments.append(f"{strike_txt} {pct_txt}")
        gap = gap_override if gap_override is not None else "  "
        return f"{label}{gap}{'  '.join(segments)}"

    def _resolve_gex_source_label(self, data: dict) -> str:
        # Primary: explicit source set on the data (set in get_gex_data)
        src = data.get("_source")
        if isinstance(src, str):
            s = src.strip()
            if not s:
                pass
            else:
                lowered = s.lower()
                if lowered == "cache":
                    return "cache"
                if "redis" in lowered:
                    return "redis"
                if lowered == "db" or "duckdb" in lowered:
                    return "DB"
                if lowered == "api" or "payload" in lowered or "api" in lowered:
                    return "API"
                return s

        # Secondary: derived from wall ladder source (redis-snapshot, redis-cache, duckdb, payload)
        ladders = data.get("_wall_ladders")
        if isinstance(ladders, dict):
            raw = ladders.get("source")
            if raw:
                lowered = str(raw).lower()
                if "redis" in lowered:
                    return "redis"
                if "duckdb" in lowered:
                    return "DB"
                if "payload" in lowered or "api" in lowered:
                    return "API"
                return str(raw)

        # Fallback: use freshness or 'local'
        freshness = data.get("_freshness")
        if isinstance(freshness, str):
            return freshness
        return "local"

    def _extract_current_maxchange_entry(self, data: dict) -> Optional[Tuple[Any, Any]]:
        maxchange = (
            data.get("maxchange")
            if isinstance(data.get("maxchange"), dict)
            else data.get("maxchange")
        )
        entry = None
        if isinstance(maxchange, dict):
            entry = maxchange.get("current")
        if not (isinstance(entry, (list, tuple)) and len(entry) >= 2):
            priors = data.get("max_priors") or []
            if priors and isinstance(priors[0], (list, tuple)) and len(priors[0]) >= 3:
                entry = [priors[0][1], priors[0][2]]
        if not (isinstance(entry, (list, tuple)) and len(entry) >= 2):
            return None
        if (
            len(entry) >= 3
            and isinstance(entry[1], (int, float))
            and isinstance(entry[2], (int, float))
        ):
            strike_val = entry[1]
            delta_val = entry[2]
        else:
            strike_val = entry[0]
            delta_val = entry[1]
        return strike_val, delta_val

    def format_gex(self, data):
        dt = data.get("timestamp")
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt)
            except Exception:
                dt = None
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            local_dt = dt.astimezone(self.display_zone)
        else:
            local_dt = None
        formatted_time = (
            local_dt.strftime("%m/%d/%Y  %I:%M:%S %p %Z") if local_dt else "N/A"
        )

        ticker = data.get("display_symbol") or data.get("ticker", "QQQ")

        def fmt_price(x):
            return f"{x:.2f}" if isinstance(x, (int, float)) else "N/A"

        # Keep gamma display as-is (no scaling)

        # Net GEX formatting: positive values show with Bn suffix, negatives show whole number
        def fmt_net_gex(x):
            if not isinstance(x, (int, float)):
                return "N/A"
            # Display magnitude scaled by 1000 with Bn suffix for both signs
            return f"{(abs(x) / 1000):.4f}Bn"

        def fmt_net_abs(x):
            return f"{abs(x):.5f}" if isinstance(x, (int, float)) else "N/A"

        ansi = {
            "reset": "\u001b[0m",
            "dim_white": "\u001b[2;37m",
            "yellow": "\u001b[2;33m",
            "green": "\u001b[2;32m",
            "red": "\u001b[2;31m",
        }

        def colorize(code, text):
            if not code:
                return text
            return f"{code}{text}{ansi['reset']}"

        def color_for_value(value, neutral="yellow"):
            if not isinstance(value, (int, float)):
                return neutral
            if value > 0:
                return "green"
            if value < 0:
                return "red"
            return neutral

        label_width = 19
        volume_width = 22

        def fmt_pair(
            label: str,
            volume_value,
            oi_value=None,
            *,
            volume_color=None,
            oi_color=None,
            formatter=None,
        ):
            formatter = formatter or fmt_price
            vol = formatter(volume_value)
            oi = formatter(oi_value) if oi_value is not None else ""
            if volume_color:
                vol = colorize(ansi.get(volume_color), vol)
            if oi_color:
                oi = colorize(ansi.get(oi_color), oi)
            return f"{label:<{label_width}}{vol:<{volume_width}}{oi}"

        source_label = self._resolve_gex_source_label(data)
        header = (
            f"GEX: {colorize(ansi['dim_white'], ticker)} {formatted_time}  "
            f"{colorize(ansi['dim_white'], fmt_price(data.get('spot_price')))}  "
            f"{colorize(ansi['dim_white'], source_label)}"
        )

        wall_label_width = max(label_width - 4, 1)
        call_gap = " " * 5
        put_gap = " " * 3
        call_wall_line = self._format_wall_line(
            data,
            "call",
            "call wall",
            fmt_price,
            label_width=wall_label_width,
            gap_override=call_gap,
            default_line=self._format_wall_value_line(
                "call wall",
                data.get("major_pos_vol"),
                data.get("major_pos_oi"),
                fmt_price,
                label_width=wall_label_width,
                volume_width=volume_width,
                gap=call_gap,
            ),
        )
        put_wall_line = self._format_wall_line(
            data,
            "put",
            "put wall",
            fmt_price,
            label_width=wall_label_width,
            gap_override=put_gap,
            default_line=self._format_wall_value_line(
                "put wall",
                data.get("major_neg_vol"),
                data.get("major_neg_oi"),
                fmt_price,
                label_width=wall_label_width,
                volume_width=volume_width,
                gap=put_gap,
            ),
        )

        table_lines = [
            "volume                                   oi",
            fmt_pair(
                "zero gamma",
                data.get("zero_gamma"),
                volume_color="yellow",
                formatter=fmt_price,
            ),
            call_wall_line,
            put_wall_line,
            fmt_pair(
                "net gex",
                data.get("net_gex"),
                data.get("sum_gex_oi"),
                volume_color=color_for_value(data.get("net_gex")),
                oi_color=color_for_value(data.get("sum_gex_oi")),
                formatter=fmt_net_gex,
            ),
        ]

        maxchange_lines = ["", "max change gex"]
        current_entry = self._extract_current_maxchange_entry(data)

        def fmt_delta_entry(label: str, entry):
            if not (isinstance(entry, (list, tuple)) and len(entry) >= 2):
                return f"{label:<18}N/A  N/ABn"
            if (
                len(entry) >= 3
                and isinstance(entry[1], (int, float))
                and isinstance(entry[2], (int, float))
            ):
                strike_val = entry[1]
                delta = entry[2]
            else:
                strike_val = entry[0]
                delta = entry[1]
            strike = (
                fmt_price(strike_val)
                if isinstance(strike_val, (int, float))
                else str(strike_val)
            )
            delta_color = ansi.get(color_for_value(delta))
            delta_text = (
                fmt_net_gex(delta) if isinstance(delta, (int, float)) else str(delta)
            )
            return f"{label:<18}{strike:<8} {colorize(delta_color, delta_text)}"

        maxchange_lines.append(fmt_delta_entry("current", current_entry))

        max_priors = data.get("max_priors", []) or []
        intervals = [1, 5, 10, 15, 30]
        for i, interval in enumerate(intervals):
            entry = max_priors[i] if i < len(max_priors) else None
            label = f"{interval} min"
            maxchange_lines.append(fmt_delta_entry(label, entry))

        body = "\n".join([header, "", *table_lines, "", *maxchange_lines])
        return f"```ansi\n{body}\n```"

    def format_gex_small(self, data):
        dt = data.get("timestamp")
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt)
            except Exception:
                dt = None
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            local_dt = dt.astimezone(self.display_zone)
        else:
            local_dt = None

        formatted_time = None
        if local_dt:
            formatted_time = local_dt.strftime("%m/%d/%Y  %I:%M:%S %p %Z")

        ticker = data.get("display_symbol") or data.get("ticker", "QQQ")

        def fmt_price(x):
            return f"{x:.2f}" if isinstance(x, (int, float)) else "N/A"

        def fmt_net_gex(x):
            if not isinstance(x, (int, float)):
                return "N/A"
            return f"{(abs(x) / 1000):.4f}Bn"

        ansi = {
            "reset": "\u001b[0m",
            "dim_white": "\u001b[2;37m",
            "yellow": "\u001b[2;33m",
            "green": "\u001b[2;32m",
            "red": "\u001b[2;31m",
        }

        def colorize(code, text):
            if not code:
                return text
            return f"{code}{text}{ansi['reset']}"

        def color_for_value(value, neutral="yellow"):
            if not isinstance(value, (int, float)):
                return neutral
            if value > 0:
                return "green"
            if value < 0:
                return "red"
            return neutral

        header_parts = [f"{ansi['dim_white']}GEX: {ticker}{ansi['reset']}"]
        if formatted_time:
            header_parts.append(formatted_time)
        price_text = fmt_price(data.get("spot_price"))
        header_parts.append(f"{ansi['dim_white']}{price_text}{ansi['reset']}")
        header_parts.append(
            f"{ansi['dim_white']}{self._resolve_gex_source_label(data)}{ansi['reset']}"
        )
        header = "  ".join(header_parts)

        classic_label_width = 19
        wall_label_width = max(classic_label_width - 4, 1)
        call_gap = "    "
        put_gap = "    "

        zero_gamma_line = f"{'zero gamma':<{classic_label_width}}{colorize(ansi['yellow'], fmt_price(data.get('zero_gamma')))}"
        call_wall_line = (
            self._format_wall_line(
                data,
                "call",
                "call wall",
                fmt_price,
                label_width=wall_label_width,
                gap_override=call_gap,
                default_line=self._format_wall_short_line(
                    "call wall",
                    data.get("major_pos_vol"),
                    fmt_price,
                    label_width=wall_label_width,
                    gap=call_gap,
                ),
            )
            + "  "
        )
        put_wall_line = (
            self._format_wall_line(
                data,
                "put",
                "put wall",
                fmt_price,
                label_width=wall_label_width,
                gap_override=put_gap,
                default_line=self._format_wall_short_line(
                    "put wall",
                    data.get("major_neg_vol"),
                    fmt_price,
                    label_width=wall_label_width,
                    gap=put_gap,
                ),
            )
            + "  "
        )

        net_color_code = ansi.get(color_for_value(data.get("net_gex")))
        net_value = colorize(net_color_code, fmt_net_gex(data.get("net_gex")))
        net_line = f"{'net gex':<{classic_label_width}}{net_value}"

        current_entry = self._extract_current_maxchange_entry(data)

        def fmt_current_entry(entry):
            label = f"{'current':<{classic_label_width}}"
            if not (isinstance(entry, (list, tuple)) and len(entry) >= 2):
                return f"{label}N/A"
            if (
                len(entry) >= 3
                and isinstance(entry[1], (int, float))
                and isinstance(entry[2], (int, float))
            ):
                strike_val = entry[1]
                delta_val = entry[2]
            else:
                strike_val = entry[0]
                delta_val = entry[1]
            strike_txt = (
                fmt_price(strike_val)
                if isinstance(strike_val, (int, float))
                else str(strike_val)
            )
            delta_txt = (
                fmt_net_gex(delta_val)
                if isinstance(delta_val, (int, float))
                else str(delta_val)
            )
            delta_color = ansi.get(color_for_value(delta_val))
            return f"{label}{strike_txt:<12}{colorize(delta_color, delta_txt)}"

        current_line = fmt_current_entry(current_entry)

        lines = [
            header,
            "",
            zero_gamma_line,
            call_wall_line,
            put_wall_line,
            net_line,
            current_line,
        ]
        body = "\n".join(lines)
        return f"```ansi\n{body}\n```"

    def format_gex_short(
        self,
        data,
        *,
        include_time: bool = True,
        time_format: Optional[str] = None,
        delta_block: Optional[Dict[str, "MetricSnapshot"]] = None,
    ):
        dt = data.get("timestamp")
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt)
            except Exception:
                dt = None
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            local_dt = dt.astimezone(self.display_zone)
        else:
            local_dt = None
        formatted_time = "N/A"
        if local_dt:
            fmt = time_format or "%m/%d/%Y  %I:%M:%S %p %Z"
            formatted_time = local_dt.strftime(fmt)

        ticker = data.get("display_symbol") or data.get("ticker", "QQQ")

        def fmt_price(x):
            return f"{x:.2f}" if isinstance(x, (int, float)) else "N/A"

        # Net GEX formatting as above
        def fmt_net_gex(x):
            if not isinstance(x, (int, float)):
                return "N/A"
            return f"{(abs(x) / 1000):.4f}Bn"

        ansi = {
            "reset": "\u001b[0m",
            "dim_white": "\u001b[2;37m",
            "yellow": "\u001b[2;33m",
            "green": "\u001b[2;32m",
            "red": "\u001b[2;31m",
        }

        def colorize(code, text):
            if not code:
                return text
            return f"{code}{text}{ansi['reset']}"

        def color_for_value(value, neutral="yellow"):
            if not isinstance(value, (int, float)):
                return neutral
            if value > 0:
                return "green"
            if value < 0:
                return "red"
            return neutral

        wall_label_width = 15
        call_gap = " " * 4
        put_gap = " " * 3
        classic_label_width = 18
        zero_gamma_line = f"{'zero gamma':<{classic_label_width}}{colorize(ansi['yellow'], fmt_price(data.get('zero_gamma')))}"
        major_pos_line = self._format_wall_line(
            data,
            "call",
            "call wall",
            fmt_price,
            label_width=wall_label_width,
            gap_override=call_gap,
            default_line=self._format_wall_short_line(
                "call wall",
                data.get("major_pos_vol"),
                fmt_price,
                label_width=wall_label_width,
                gap=call_gap,
            ),
        )
        major_neg_line = self._format_wall_line(
            data,
            "put",
            "put wall",
            fmt_price,
            label_width=wall_label_width,
            gap_override=put_gap,
            default_line=self._format_wall_short_line(
                "put wall",
                data.get("major_neg_vol"),
                fmt_price,
                label_width=wall_label_width,
                gap=put_gap,
            ),
        )

        net_color_code = ansi.get(color_for_value(data.get("net_gex")))
        net_value = colorize(net_color_code, fmt_net_gex(data.get("net_gex")))

        sum_gex_oi_value = fmt_price(data.get("sum_gex_oi"))
        sum_gex_oi_color = ansi.get(color_for_value(data.get("sum_gex_oi")))
        sum_oi_line = f"{'sum gex oi':<{classic_label_width}}{colorize(sum_gex_oi_color, sum_gex_oi_value)}"

        delta_rr_value = data.get("delta_risk_reversal")
        delta_rr_color = ansi.get(color_for_value(delta_rr_value))
        delta_rr_line = f"{'delta risk rev':<{classic_label_width}}{colorize(delta_rr_color, fmt_price(delta_rr_value))}"

        current_entry = self._extract_current_maxchange_entry(data)

        def snapshot(key: str) -> "MetricSnapshot":
            snap = delta_block.get(key) if delta_block else None
            if isinstance(snap, MetricSnapshot):
                return snap
            return MetricSnapshot(
                value=None, delta=None, percent=None, previous=None, baseline=None
            )

        def fmt_abs(value, digits: int = 2) -> str:
            if not isinstance(value, (int, float)):
                return "N/A"
            return f"{abs(value):.{digits}f}"

        def fmt_percent(value) -> str:
            if not isinstance(value, (int, float)):
                return "n/a%"
            return f"{value:+.2f}%"

        def color_for_delta(delta) -> str:
            if not isinstance(delta, (int, float)):
                return "yellow"
            return "green" if delta >= 0 else "red"

        header_parts = [f"{ansi['dim_white']}GEX: {ticker}{ansi['reset']}"]
        if include_time:
            header_parts.append(formatted_time)

        price_text = fmt_price(data.get("spot_price"))
        if delta_block:
            spot_snap = snapshot("spot")
            delta_text = fmt_abs(spot_snap.delta)
            percent_text = fmt_percent(spot_snap.percent)
            delta_color = ansi.get(color_for_delta(spot_snap.delta))
            delta_display = colorize(delta_color, f"Δ{delta_text} {percent_text}")
            price_text = f"{price_text}  {delta_display}"
        header_parts.append(f"{ansi['dim_white']}{price_text}{ansi['reset']}")
        header_parts.append(
            f"{ansi['dim_white']}{self._resolve_gex_source_label(data)}{ansi['reset']}"
        )
        header = "  ".join(header_parts)

        if delta_block:
            lines = self._format_feed_short_lines(
                header,
                data,
                delta_block,
                ansi,
                colorize,
                color_for_value,
                fmt_price,
                fmt_net_gex,
            )
        else:

            def fmt_delta_entry(label: str, entry):
                if not (isinstance(entry, (list, tuple)) and len(entry) >= 2):
                    return f"{label:<18}N/A  N/ABn"
                if (
                    len(entry) >= 3
                    and isinstance(entry[1], (int, float))
                    and isinstance(entry[2], (int, float))
                ):
                    strike_val = entry[1]
                    delta = entry[2]
                else:
                    strike_val = entry[0]
                    delta = entry[1]
                strike_txt = (
                    fmt_price(strike_val)
                    if isinstance(strike_val, (int, float))
                    else str(strike_val)
                )
                delta_txt = (
                    fmt_net_gex(delta)
                    if isinstance(delta, (int, float))
                    else str(delta)
                )
                delta_color = ansi.get(color_for_value(delta))
                return f"{label:<18}{strike_txt:<8} {colorize(delta_color, delta_txt)}"

            maxchange_lines = ["", "max change gex"]
            maxchange_lines.append(fmt_delta_entry("current", current_entry))

            max_priors = data.get("max_priors", []) or []
            intervals = [1, 5, 10, 15, 30]
            for i, interval in enumerate(intervals):
                entry = max_priors[i] if i < len(max_priors) else None
                label = f"{interval} min"
                maxchange_lines.append(fmt_delta_entry(label, entry))

            lines = [
                header,
                "",
                zero_gamma_line,
                major_pos_line,
                major_neg_line,
                f"net gex           {net_value}",
                sum_oi_line,
                delta_rr_line,
                *maxchange_lines,
            ]
        body = "\n".join(lines)
        return f"```ansi\n{body}\n```"

    def _format_feed_short_lines(
        self,
        header: str,
        data: dict,
        delta_block: Dict[str, "MetricSnapshot"],
        ansi: Dict[str, str],
        colorize,
        color_for_value,
        fmt_price,
        fmt_net_gex,
    ) -> List[str]:
        def fmt_abs(value, digits: int = 2) -> str:
            if not isinstance(value, (int, float)):
                return "N/A"
            return f"{abs(value):.{digits}f}"

        def fmt_percent(value) -> str:
            if not isinstance(value, (int, float)):
                return "n/a%"
            return f"{abs(value):.2f}%"

        def color_for_delta(delta) -> str:
            if not isinstance(delta, (int, float)):
                return "yellow"
            return "green" if delta >= 0 else "red"

        def snapshot(key: str) -> "MetricSnapshot":
            snap = delta_block.get(key)
            if isinstance(snap, MetricSnapshot):
                return snap
            return MetricSnapshot(
                value=None, delta=None, percent=None, previous=None, baseline=None
            )

        def fmt_prev(value) -> str:
            if not isinstance(value, (int, float)):
                return "n/a"
            return fmt_abs(value)

        lines = [header, ""]

        def append_prev_line(label: str, snap_key: str):
            snap = snapshot(snap_key)
            delta = None
            if isinstance(snap.value, (int, float)) and isinstance(
                snap.previous, (int, float)
            ):
                delta = snap.value - snap.previous
            color = color_for_delta(delta)
            current_txt = colorize(ansi.get(color), fmt_abs(snap.value))
            prev_txt = fmt_prev(snap.previous)
            lines.append(f"{label:<17}{current_txt:<10} ({prev_txt})")

        append_prev_line("zero gamma", "zero_gamma")
        classic_label_width = 19
        wall_label_width = max(classic_label_width - 4, 1)
        call_gap = "  "
        put_gap = "  "

        call_wall_line = (
            self._format_wall_line(
                data,
                "call",
                "call wall",
                fmt_price,
                label_width=wall_label_width,
                gap_override=call_gap,
                default_line=self._format_wall_short_line(
                    "call wall",
                    data.get("major_pos_vol"),
                    fmt_price,
                    label_width=wall_label_width,
                    gap=call_gap,
                ),
            )
            + "  "
        )
        put_wall_line = (
            self._format_wall_line(
                data,
                "put",
                "put wall",
                fmt_price,
                label_width=wall_label_width,
                gap_override=put_gap,
                default_line=self._format_wall_short_line(
                    "put wall",
                    data.get("major_neg_vol"),
                    fmt_price,
                    label_width=wall_label_width,
                    gap=put_gap,
                ),
            )
            + "  "
        )
        lines.append(call_wall_line)
        lines.append(put_wall_line)

        def fmt_net_change(delta) -> str:
            if not isinstance(delta, (int, float)):
                return "N/ABn"
            return f"{(abs(delta) / 1000):.4f}Bn"

        net_snap = snapshot("net_gex")
        net_value = colorize(
            ansi.get(color_for_value(data.get("net_gex"))),
            fmt_net_gex(data.get("net_gex")),
        )
        net_change_color = ansi.get(color_for_delta(net_snap.delta))
        net_change_txt = colorize(net_change_color, fmt_net_change(net_snap.delta))
        lines.append(f"net gex           {net_value:<10} Δ{net_change_txt}")

        scaled_snap = snapshot("scaled_gamma")
        scaled_value = fmt_net_gex(data.get("sum_gex_oi"))
        scaled_change_txt = colorize(
            ansi.get(color_for_delta(scaled_snap.delta)),
            fmt_net_change(scaled_snap.delta),
        )
        lines.append(f"scaled gamma      {scaled_value:<10} Δ{scaled_change_txt}")

        current_entry = self._extract_current_maxchange_entry(data)
        if current_entry:
            strike_val, delta_val = current_entry
            current_price = fmt_price(strike_val)
            delta_text = (
                fmt_net_gex(delta_val)
                if isinstance(delta_val, (int, float))
                else str(delta_val)
            )
            delta_color = ansi.get(color_for_value(delta_val))
            current_delta = colorize(delta_color, delta_text)
        else:
            current_price = "N/A"
            current_delta = colorize(ansi["yellow"], "N/ABn")
        max_snap = snapshot("maxchange")
        max_change_txt = colorize(
            ansi.get(color_for_delta(max_snap.delta)), fmt_net_change(max_snap.delta)
        )
        lines.append(
            f"current strike    {current_price:<8}   {current_delta}  Δ{max_change_txt}"
        )
        return lines

    def format_option_trade_alert(self, payload: dict) -> str:
        data = payload.get("data") or payload
        timestamp = data.get("timestamp") or payload.get("received_at") or "N/A"
        if isinstance(timestamp, str):
            timestamp = timestamp.replace("T", " ").replace("Z", " UTC")
        transaction_types = (
            data.get("transaction_type") or data.get("transaction_types") or []
        )
        if isinstance(transaction_types, str):
            transaction_types = [transaction_types]
        ticker = data.get("ticker") or data.get("symbol") or "UNKNOWN"
        side = data.get("side") or data.get("direction") or "N/A"
        call_put = data.get("call_put") or data.get("option_type") or ""
        strike = data.get("strike") or data.get("strike_price") or "N/A"
        contract = data.get("contract") or "n/a"
        dte = data.get("dte") or data.get("days_to_expiration") or "N/A"
        stock_spot = data.get("stock_spot") or data.get("underlying_price") or "N/A"
        bid_range = data.get("bid_ask_range") or data.get("bid_range") or "N/A"
        option_spot = data.get("option_spot") or data.get("option_price") or "N/A"
        size = data.get("size") or data.get("contracts") or "N/A"
        premium = data.get("premium") or data.get("notional") or "N/A"
        volume = data.get("volume") or "N/A"
        oi = data.get("oi") or data.get("open_interest") or "N/A"
        chain_bid = data.get("chain_bid") or data.get("bid") or "N/A"
        chain_ask = data.get("chain_ask") or data.get("ask") or "N/A"
        legs = data.get("legs") or []
        code = data.get("code") or "N/A"
        flags = data.get("flags") or []
        tags = data.get("tags") or []
        uw_info = data.get("unusual_whales") or {}
        uw_id = uw_info.get("alert_id") or "n/a"
        uw_score = uw_info.get("score")

        legs_text = (
            ", ".join(
                f"{leg.get('ratio', 1)}x{leg.get('strike')} {leg.get('type')}"
                for leg in legs
                if isinstance(leg, dict)
            )
            or "n/a"
        )
        flags_text = ", ".join(flags) if flags else "n/a"
        tags_text = ", ".join(tags) if tags else "n/a"
        tx_text = ", ".join(transaction_types) if transaction_types else "unknown"

        uw_line = f"UW alert {uw_id}"
        if uw_score is not None:
            uw_line += f" score {uw_score}"

        lines = [
            f"UW option alert  {timestamp}",
            f"ticker          {ticker}",
            f"types           {tx_text}",
            f"contract        {contract}",
            f"side/strike     {side} {strike} {call_put}  dte {dte}",
            f"stock spot      {stock_spot}  bid-ask {bid_range}",
            f"option spot     {option_spot}  size {size}",
            f"premium         {premium}  volume {volume}  oi {oi}",
            f"chain bid/ask   {chain_bid} / {chain_ask}",
            f"legs            {legs_text}",
            f"code            {code}",
            f"flags           {flags_text}",
            f"tags            {tags_text}",
            uw_line,
        ]
        return "```ansi\n" + "\n".join(lines) + "\n```"

    def format_market_snapshot(self, payload: dict) -> str:
        data = payload.get("data") or payload
        timestamp = data.get("timestamp") or payload.get("received_at") or "N/A"
        ticker = data.get("ticker") or data.get("symbol") or "INDEX"
        stock_spot = data.get("stock_spot") or data.get("price") or "N/A"
        bid = data.get("bid") or "N/A"
        ask = data.get("ask") or "N/A"
        volume = data.get("volume") or "N/A"
        advancers = data.get("advancers") or "N/A"
        decliners = data.get("decliners") or "N/A"
        net_flow = data.get("net_flow") or data.get("net") or "N/A"
        session = data.get("session") or data.get("market_session") or "N/A"

        lines = [
            f"Market state | {ticker}",
            f"timestamp       {timestamp}",
            f"spot            {stock_spot}",
            f"bid / ask       {bid} / {ask}",
            f"volume          {volume}",
            f"adv / dec       {advancers} / {decliners}",
            f"net flow        {net_flow}",
            f"session         {session}",
        ]
        return "```ansi\n" + "\n".join(lines) + "\n```"

    def format_status(self, status_data):
        formatted = "**Data Pipeline Status:**\n\n"
        for key, value in status_data.items():
            if isinstance(value, dict):
                formatted += f"**{key}:**\n"
                for subkey, subvalue in value.items():
                    formatted += f"  {subkey}: {subvalue}\n"
                formatted += "\n"
            else:
                formatted += f"**{key}:** {value}\n\n"
        return formatted

    def format_tastytrade_summary(self, summary) -> str:
        lines = [
            "TastyTrade account summary",
            f"account        {summary.account_number}",
            f"nickname       {summary.nickname or 'n/a'}",
            f"type           {summary.account_type}",
            f"buying power   {summary.buying_power:,.2f}",
            f"net liq        {summary.net_liq:,.2f}",
            f"cash balance   {summary.cash_balance:,.2f}",
        ]
        return "```ansi\n" + "\n".join(lines) + "\n```"

    def format_tastytrade_overview(self, overview: dict) -> str:
        def fmt_bool(value):
            return "Yes" if value else "No"

        lines = [
            f"TastyTrade trading status   account {overview.get('account-number', 'n/a')}",
            f"options level      {overview.get('options-level', 'n/a')}",
            f"frozen             {fmt_bool(overview.get('is-frozen'))}",
            f"closing only       {fmt_bool(overview.get('is-closing-only'))}",
            f"margin call        {fmt_bool(overview.get('is-in-margin-call'))}",
            f"pattern day trader {fmt_bool(overview.get('is-pattern-day-trader'))}",
            f"futures enabled    {fmt_bool(overview.get('is-futures-enabled'))}",
            f"crypto enabled     {fmt_bool(overview.get('is-cryptocurrency-enabled'))}",
            f"equity offering    {fmt_bool(overview.get('is-equity-offering-enabled'))}",
            f"day trade count    {overview.get('day-trade-count', 'n/a')}",
            f"fee schedule       {overview.get('fee-schedule-name', 'n/a')}",
            f"updated at         {overview.get('updated-at', 'n/a')}",
        ]
        return "```ansi\n" + "\n".join(lines) + "\n```"


@dataclass
class MetricSnapshot:
    value: Optional[float]
    delta: Optional[float]
    percent: Optional[float]
    previous: Optional[float]
    baseline: Optional[float]


class RollingWindowTracker:
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self._samples: Dict[str, Deque[Tuple[datetime, float]]] = defaultdict(deque)
        self._last_value: Dict[str, Optional[float]] = {}
        self._last_delta: Dict[str, Optional[float]] = {}
        self._last_percent: Dict[str, Optional[float]] = {}

    def update(self, key: str, value: Any, now: datetime) -> MetricSnapshot:
        previous_value = self._last_value.get(key)
        if not isinstance(value, (int, float)):
            return MetricSnapshot(
                value=None,
                delta=self._last_delta.get(key),
                percent=self._last_percent.get(key),
                previous=previous_value,
                baseline=None,
            )
        self._last_value[key] = value
        deque_ref = self._samples.setdefault(key, deque())
        deque_ref.append((now, value))
        cutoff = now - timedelta(seconds=self.window_seconds)
        while len(deque_ref) > 1 and deque_ref[0][0] < cutoff:
            deque_ref.popleft()
        baseline = deque_ref[0][1] if deque_ref else value
        delta = value - baseline if deque_ref else None
        if delta is None:
            delta = self._last_delta.get(key)
        else:
            self._last_delta[key] = delta
        percent = None
        if delta is not None and baseline not in (None, 0):
            percent = (delta / baseline) * 100
        if percent is None:
            percent = self._last_percent.get(key)
        else:
            self._last_percent[key] = percent
        return MetricSnapshot(
            value=value,
            delta=delta,
            percent=percent,
            previous=previous_value,
            baseline=baseline,
        )


class RedisGexFeedMetrics:
    def __init__(self, redis_client, key: str, *, enabled: bool = False):
        self.redis_client = redis_client
        self.key = key
        self.enabled = enabled
        self._update_count = 0
        self._error_count = 0

    async def record_update(
        self, now: datetime, delta_map: Dict[str, MetricSnapshot], *, refresh: bool
    ) -> None:
        if not self.enabled:
            return
        self._update_count += 1
        mapping = {
            "last_update_ts": now.isoformat(),
            "update_count": str(self._update_count),
        }
        if refresh:
            mapping["last_refresh_ts"] = now.isoformat()
        mapping.update(self._flatten_delta_map(delta_map))
        await asyncio.to_thread(self.redis_client.hset, self.key, mapping=mapping)

    async def record_error(self, now: datetime, *, reason: str) -> None:
        if not self.enabled:
            return
        self._error_count += 1
        mapping = {
            "last_error_ts": now.isoformat(),
            "error_count": str(self._error_count),
            "last_error_reason": reason,
        }
        await asyncio.to_thread(self.redis_client.hset, self.key, mapping=mapping)

    def _flatten_delta_map(
        self, delta_map: Dict[str, MetricSnapshot]
    ) -> Dict[str, str]:
        flattened: Dict[str, str] = {}
        for key, snapshot in (delta_map or {}).items():
            if not isinstance(snapshot, MetricSnapshot):
                continue
            flattened[f"{key}_value"] = self._stringify(snapshot.value)
            flattened[f"{key}_delta"] = self._stringify(snapshot.delta)
            flattened[f"{key}_percent"] = self._stringify(snapshot.percent)
        return flattened

    @staticmethod
    def _stringify(value: Optional[float]) -> str:
        if value is None:
            return ""
        return f"{value}"
