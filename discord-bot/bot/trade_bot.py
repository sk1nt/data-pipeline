import asyncio
import json
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import discord
import httpx
import redis
from discord.ext import commands
from zoneinfo import ZoneInfo

from .tastytrade_client import TastyTradeClient

class TradeBot(commands.Bot):
    def __init__(self, config):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        self.config = config
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            db=int(os.getenv('REDIS_DB', '0')),
            password=os.getenv('REDIS_PASSWORD')
        )
        self.command_admin_ids = self._parse_admin_ids()
        self.command_admin_names = self._parse_admin_names()
        self.tastytrade_client = self._init_tastytrade_client()
        self.uw_option_latest_key = os.getenv('UW_OPTION_LATEST_KEY', 'uw:options-trade:latest')
        self.uw_option_history_key = os.getenv('UW_OPTION_HISTORY_KEY', 'uw:options-trade:history')
        self.uw_market_latest_key = os.getenv('UW_MARKET_LATEST_KEY', 'uw:market-state:latest')
        self.uw_market_history_key = os.getenv('UW_MARKET_HISTORY_KEY', 'uw:market-state:history')
        self.uw_option_stream_channel = os.getenv('UW_OPTION_STREAM_CHANNEL', 'uw:options-trade:stream')
        # Canonical tickers for GEXBot futures endpoints so cache/API reuse shared contracts
        self.ticker_aliases = {
            'NQ': 'NQ_NDX',
            'MNQ': 'NQ_NDX',
            'ES': 'ES_SPX',
            'MES': 'ES_SPX',
        }
        self.display_zone = ZoneInfo(os.getenv('DISPLAY_TIMEZONE', 'America/New_York'))
        self.redis_snapshot_prefix = os.getenv('GEX_SNAPSHOT_PREFIX', 'gex:snapshot:')
        self.allowed_channel_ids = tuple(getattr(config, 'allowed_channel_ids', ()) or ())
        self.status_channel_id = getattr(config, 'status_channel_id', None)
        self.status_command_user = os.getenv('DISCORD_STATUS_USER', 'skint0552').lower()
        self.option_alert_channel_ids = self._init_alert_channels()
        self._uw_listener_task: Optional[asyncio.Task] = None
        self._uw_listener_stop = asyncio.Event()
        self._gex_feed_task: Optional[asyncio.Task] = None
        self._gex_feed_stop = asyncio.Event()
        self.gex_feed_enabled = bool(getattr(config, 'gex_feed_enabled', False))
        self.gex_feed_channel_ids = getattr(config, 'gex_feed_channel_ids', None)
        self.gex_feed_symbol = (getattr(config, 'gex_feed_symbol', 'NQ_NDX') or 'NQ_NDX').upper()
        self.gex_feed_update_seconds = max(0.5, float(getattr(config, 'gex_feed_update_seconds', 1.0) or 1.0))
        self.gex_feed_refresh_minutes = max(1, int(getattr(config, 'gex_feed_refresh_minutes', 5) or 5))
        self.gex_feed_window_seconds = max(15, int(getattr(config, 'gex_feed_window_seconds', 60) or 60))
        self.gex_feed_force_window = bool(getattr(config, 'gex_feed_force_window', False))
        metrics_key = getattr(config, 'gex_feed_metrics_key', 'metrics:gex_feed') or 'metrics:gex_feed'
        metrics_enabled = bool(getattr(config, 'gex_feed_metrics_enabled', False))
        self._gex_feed_metrics = RedisGexFeedMetrics(self.redis_client, metrics_key, enabled=metrics_enabled)
        self.gex_feed_backoff_base_seconds = max(self.gex_feed_update_seconds, float(getattr(config, 'gex_feed_backoff_base_seconds', self.gex_feed_update_seconds)))
        self.gex_feed_backoff_max_seconds = max(self.gex_feed_backoff_base_seconds, float(getattr(config, 'gex_feed_backoff_max_seconds', 300.0)))
        self._gex_feed_backoff_seconds = 0.0
        self.gexbot_dynamic_key = getattr(config, 'gexbot_dynamic_key', 'gexbot:symbols:dynamic')
        self.gexbot_supported_key = getattr(config, 'gexbot_supported_key', 'gexbot:symbols:supported')
        default_symbols = getattr(config, 'gexbot_default_symbols', None)
        if not default_symbols:
            default_symbols = ['NQ_NDX', 'ES_SPX', 'SPY', 'QQQ', 'SPX', 'NDX']
        if isinstance(default_symbols, str):
            default_symbols = [sym.strip() for sym in default_symbols.split(',') if sym.strip()]
        self._gexbot_default_symbols = {sym.upper() for sym in default_symbols}
        # Use a path variable and create short-lived connections per query to avoid file locks
        self.duckdb_path = os.getenv('DUCKDB_PATH', '/home/rwest/projects/data-pipeline/data/gex_data.db')
        # Register commands defined as methods on the subclass so prefix commands work
        try:
            # Use plain functions (closures) as command callbacks so signature checks pass.
            async def _ping_cmd(ctx):
                await ctx.send('Pong!')

            async def _gex_cmd(ctx, *args):
                symbol, show_full = self._resolve_gex_request(args)
                display_symbol = (symbol or 'QQQ').upper()
                ticker = self.ticker_aliases.get(display_symbol, display_symbol)

                # verify supported symbol map before continuing
                tickers_map = await self._get_supported_tickers()
                supported_set = set()
                for values in tickers_map.values():
                    if isinstance(values, list):
                        supported_set.update(val.upper() for val in values if isinstance(val, str))

                if tickers_map and ticker.upper() not in supported_set:
                    msg = self._format_supported_tickers_message(tickers_map, alias_map=self.ticker_aliases)
                    await ctx.send(f"Unsupported symbol '{symbol}'.\n{msg}")
                    return

                await self._maybe_enroll_dynamic_symbol(ticker)
                poller = getattr(self, 'gex_poller', None)

                if poller:
                    try:
                        base_symbols = getattr(poller, '_base_symbols', set()) or set()
                        if ticker.upper() not in base_symbols and hasattr(poller, 'add_symbol_for_day'):
                            try:
                                poller.add_symbol_for_day(ticker)
                            except Exception:
                                pass
                        if hasattr(poller, 'fetch_symbol_now'):
                            snap = await poller.fetch_symbol_now(ticker)
                            if snap:
                                snap['_freshness'] = 'current'
                                snap['_source'] = 'redis-cache'
                                snap['display_symbol'] = display_symbol
                                snap['spot_price'] = snap.get('spot_price') or snap.get('spot')
                                snap['major_pos_vol'] = snap.get('major_pos_vol') if snap.get('major_pos_vol') is not None else 0
                                snap['major_neg_vol'] = snap.get('major_neg_vol') if snap.get('major_neg_vol') is not None else 0
                                snap['major_pos_oi'] = snap.get('major_pos_oi') if snap.get('major_pos_oi') is not None else 0
                                snap['major_neg_oi'] = snap.get('major_neg_oi') if snap.get('major_neg_oi') is not None else 0
                                snap['sum_gex_oi'] = snap.get('sum_gex_oi') if snap.get('sum_gex_oi') is not None else 0
                                snap['max_priors'] = snap.get('max_priors') or []
                                await asyncio.to_thread(self.redis_client.setex, f"gex:{ticker.lower()}:latest", 300, json.dumps(snap, default=str))
                                await asyncio.to_thread(self.redis_client.set, f"{self.redis_snapshot_prefix}{ticker.upper()}", json.dumps(snap, default=str))
                                formatter = self.format_gex if show_full else self.format_gex_short
                                await ctx.send(formatter(snap))
                                return
                    except Exception:
                        pass
                data = await self.get_gex_data(symbol)
                if data:
                    formatter = self.format_gex if show_full else self.format_gex_short
                    await ctx.send(formatter(data))
                else:
                    await ctx.send('GEX data not available')

            async def _status_cmd(ctx):
                if not await self._ensure_status_channel_access(ctx):
                    return
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get('http://localhost:8877/status')
                        if response.status_code == 200:
                            status_data = response.json()
                            await ctx.send(self.format_status(status_data))
                        else:
                            await ctx.send(f'Failed to fetch status: {response.status_code}')
                except Exception as e:
                    await ctx.send(f'Error fetching status: {e}')

            async def _tastytrade_cmd(ctx):
                if not await self._ensure_privileged(ctx):
                    return
                await _tt_cmd(ctx)

            async def _market_cmd(ctx):
                snapshot = await self._fetch_market_snapshot()
                if not snapshot:
                    await ctx.send('No market state snapshot available')
                    return
                await ctx.send(self.format_market_snapshot(snapshot))

            async def _uwalerts_cmd(ctx):
                alerts = await self._fetch_option_history(limit=5)
                if not alerts:
                    await ctx.send('No option trade alerts yet')
                    return
                formatted = "\n".join(self.format_option_trade_alert(alert) for alert in alerts)
                await ctx.send(formatted)

            async def _tt_cmd(ctx, *args):
                if not await self._ensure_status_channel_access(ctx):
                    return
                if not self.tastytrade_client:
                    await self._send_dm_or_warn(ctx, 'TastyTrade client is not configured.')
                    return
                subcommand = args[0].lower() if args else 'summary'
                if subcommand == 'status':
                    overview = await self._fetch_tastytrade_overview()
                    if overview is None:
                        await self._send_dm_or_warn(ctx, 'Unable to fetch TastyTrade overview.')
                        return
                    message = self.format_tastytrade_overview(overview)
                    await self._send_dm_or_warn(ctx, message)
                    return
                if subcommand == 'account' and len(args) >= 2:
                    target = args[1]
                    success = await asyncio.to_thread(self.tastytrade_client.set_active_account, target)
                    dm_msg = (
                        f"Active TastyTrade account set to {target}."
                        if success else f"Account {target} not found."
                    )
                    await self._send_dm_or_warn(ctx, dm_msg)
                    return
                summary = await self._fetch_tastytrade_summary()
                if summary is None:
                    await self._send_dm_or_warn(ctx, 'Unable to fetch TastyTrade summary.')
                    return
                message = self.format_tastytrade_summary(summary)
                await self._send_dm_or_warn(ctx, message)

            self.add_command(commands.Command(_ping_cmd, name='ping'))
            self.add_command(commands.Command(_gex_cmd, name='gex'))
            self.add_command(commands.Command(_status_cmd, name='status'))
            self.add_command(commands.Command(_tastytrade_cmd, name='tastytrade'))
            self.add_command(commands.Command(_market_cmd, name='market'))
            self.add_command(commands.Command(_uwalerts_cmd, name='uw'))
            self.add_command(commands.Command(_tt_cmd, name='tt'))
        except Exception as e:
            # If registration fails (e.g., during import-time tests), print error for debugging
            print(f"Command registration failed: {e}")

    async def on_ready(self):
        print(f'Bot logged in as {self.user}')
        if not self._uw_listener_task:
            self._uw_listener_stop.clear()
            self._uw_listener_task = asyncio.create_task(self._listen_option_trade_stream())
        if self._should_run_gex_feed() and not self._gex_feed_task:
            self._gex_feed_stop.clear()
            self._gex_feed_task = asyncio.create_task(self._run_gex_feed_loop())

    async def on_message(self, message):
        if message.author == self.user:
            return
        if not self._is_allowed_channel(message.channel):
            return
        await super().on_message(message)

    async def get_context(self, origin, *, cls=commands.Context):
        ctx = await super().get_context(origin, cls=cls)
        if ctx and not self._is_allowed_channel(getattr(ctx, 'channel', None)):
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
        await ctx.send('Pong!')

    def _should_run_gex_feed(self) -> bool:
        if not self.gex_feed_enabled:
            return False
        if not self.gex_feed_channel_ids:
            return False
        return True

    def _reset_feed_backoff(self) -> None:
        self._gex_feed_backoff_seconds = 0.0

    def _record_feed_rate_limit(self, status: Optional[int]) -> None:
        if status not in (427, 429):
            return
        if self._gex_feed_backoff_seconds <= 0:
            self._gex_feed_backoff_seconds = self.gex_feed_backoff_base_seconds
        else:
            self._gex_feed_backoff_seconds = min(self.gex_feed_backoff_max_seconds, self._gex_feed_backoff_seconds * 2)
        print(f"GEX feed rate-limited (HTTP {status}); backing off for {self._gex_feed_backoff_seconds:.1f}s")

    def _handle_feed_http_exception(self, exc: discord.HTTPException) -> bool:
        status = getattr(exc, 'status', None)
        if status in (427, 429):
            self._record_feed_rate_limit(status)
            return True
        return False

    async def _sleep_between_feed_updates(self) -> None:
        delay = self.gex_feed_update_seconds
        if self._gex_feed_backoff_seconds:
            delay = max(delay, self._gex_feed_backoff_seconds)
        await asyncio.sleep(delay)

    async def _get_supported_tickers(self) -> dict:
        cache_key = 'gexbot:tickers:categories'
        try:
            raw = await asyncio.to_thread(self.redis_client.get, cache_key)
            if raw:
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode('utf-8')
                return json.loads(raw)
        except Exception:
            pass

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get('https://api.gexbot.com/tickers')
                data = resp.json()
                if resp.status_code == 200 and isinstance(data, dict):
                    try:
                        await asyncio.to_thread(self.redis_client.setex, cache_key, 86400, json.dumps(data))
                    except Exception:
                        pass
                    return data
        except Exception:
            pass
        return {}

    def _format_supported_tickers_message(self, tickers_map: dict, alias_map: Optional[dict] = None) -> str:
        if not isinstance(tickers_map, dict) or not tickers_map:
            return 'Unsupported symbol. Unable to fetch supported symbol list.'
        msg = 'Unsupported symbol. Supported tickers by category:\n'
        for category, symbols in tickers_map.items():
            if isinstance(symbols, list) and symbols:
                msg += f"**{category.title()}**: {', '.join(sorted(symbols))}\n"
        if alias_map:
            alias_texts = []
            for key, value in alias_map.items():
                alias_texts.append(f"{key}/{key.lower()} -> {value}")
            if alias_texts:
                msg += '\nAliases: ' + '; '.join(alias_texts)
        return msg

    async def _maybe_enroll_dynamic_symbol(self, ticker: str) -> None:
        normalized = (ticker or '').upper().strip()
        if not normalized or normalized in self._gexbot_default_symbols:
            return
        entries = await self._load_dynamic_entries()
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        expires_at = (now + timedelta(hours=24)).astimezone(timezone.utc).isoformat()
        entries[normalized] = expires_at
        await self._write_dynamic_entries(entries)

    async def _load_dynamic_entries(self) -> Dict[str, str]:
        try:
            raw = await asyncio.to_thread(self.redis_client.get, self.gexbot_dynamic_key)
        except Exception:
            raw = None
        return self._parse_dynamic_entries(raw)

    def _parse_dynamic_entries(self, raw: Any) -> Dict[str, str]:
        entries: Dict[str, str] = {}
        if not raw:
            return entries
        if isinstance(raw, (bytes, bytearray)):
            try:
                raw = raw.decode('utf-8')
            except Exception:
                return entries
        try:
            decoded = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            decoded = raw
        if not isinstance(decoded, list):
            return entries
        for item in decoded:
            normalized = self._normalize_dynamic_entry(item)
            if normalized:
                entries[normalized['symbol']] = normalized['expires_at']
        return entries

    def _normalize_dynamic_entry(self, item: Any) -> Optional[Dict[str, str]]:
        symbol: Optional[str] = None
        expires_at: Optional[str] = None
        if isinstance(item, str):
            symbol = item.upper()
        elif isinstance(item, dict):
            raw_symbol = item.get('symbol')
            if isinstance(raw_symbol, str) and raw_symbol.strip():
                symbol = raw_symbol.upper()
            elif raw_symbol is not None:
                symbol = str(raw_symbol).upper()
            expiry_raw = item.get('expires_at') or item.get('expiry')
            if expiry_raw:
                expires_at = self._normalize_datetime(expiry_raw)
        if not symbol:
            return None
        if not expires_at:
            expires_at = (datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(hours=24)).isoformat()
        return {'symbol': symbol, 'expires_at': expires_at}

    def _normalize_datetime(self, value: Any) -> Optional[str]:
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
            except Exception:
                return None
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()
        return None

    async def _write_dynamic_entries(self, entries: Dict[str, str]) -> None:
        if not entries:
            try:
                await asyncio.to_thread(self.redis_client.delete, self.gexbot_dynamic_key)
            except Exception:
                pass
            return
        payload = [
            {
                'symbol': symbol,
                'expires_at': expires,
            }
            for symbol, expires in sorted(entries.items())
        ]
        value = json.dumps(payload)
        ttl = 86400
        try:
            await asyncio.to_thread(self.redis_client.setex, self.gexbot_dynamic_key, ttl, value)
        except Exception:
            try:
                await asyncio.to_thread(self.redis_client.set, self.gexbot_dynamic_key, value)
            except Exception:
                pass

    async def _run_gex_feed_loop(self):
        try:
            while not self._gex_feed_stop.is_set():
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
                        await self._sleep_with_stop((start + timedelta(days=1) - now).total_seconds())
                        continue
                    if now < start:
                        await self._sleep_with_stop((start - now).total_seconds())
                        continue
                channels = await self._resolve_feed_channels()
                if not channels:
                    await asyncio.sleep(120)
                    continue
                await self._run_gex_feed_session(channels, session_end)
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

    async def _resolve_feed_channels(self) -> List[discord.abc.Messageable]:
        channels = []
        ids = self.gex_feed_channel_ids or ()
        for cid in ids:
            channel = self.get_channel(cid)
            if channel is None:
                try:
                    channel = await self.fetch_channel(cid)
                except Exception as exc:
                    print(f"Failed to fetch channel {cid}: {exc}")
                    continue
            if hasattr(channel, 'send'):
                channels.append(channel)
        return channels

    async def _run_gex_feed_session(self, channels: List[discord.abc.Messageable], session_end: datetime) -> None:
        tracker = RollingWindowTracker(window_seconds=self.gex_feed_window_seconds)
        messages: Dict[int, discord.Message] = {}
        next_refresh = self._next_refresh_boundary(datetime.now(self.display_zone))
        while not self._gex_feed_stop.is_set():
            now = datetime.now(self.display_zone)
            if now >= session_end:
                break
            data = await self.get_gex_data(self.gex_feed_symbol)
            if not data:
                await self._gex_feed_metrics.record_error(now, reason='no_data')
                await self._sleep_between_feed_updates()
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
                await self._gex_feed_metrics.record_update(now, delta_map, refresh=True)
            else:
                await self._edit_feed_messages(messages, content)
                await self._gex_feed_metrics.record_update(now, delta_map, refresh=False)
            if now >= next_refresh and now < session_end:
                await self._delete_feed_messages(messages)
                messages = await self._post_feed_messages(channels, content)
                await self._gex_feed_metrics.record_update(now, delta_map, refresh=True)
                next_refresh = self._next_refresh_boundary(now)
            await self._sleep_between_feed_updates()

    def _next_refresh_boundary(self, current: datetime) -> datetime:
        minute_block = (current.minute // self.gex_feed_refresh_minutes) * self.gex_feed_refresh_minutes
        boundary = current.replace(minute=minute_block, second=0, microsecond=0) + timedelta(minutes=self.gex_feed_refresh_minutes)
        if boundary <= current:
            boundary += timedelta(minutes=self.gex_feed_refresh_minutes)
        return boundary

    async def _post_feed_messages(self, channels: List[discord.abc.Messageable], content: str) -> Dict[int, discord.Message]:
        messages: Dict[int, discord.Message] = {}
        rate_limited = False
        for channel in channels:
            try:
                msg = await channel.send(content)
            except discord.HTTPException as exc:
                if self._handle_feed_http_exception(exc):
                    rate_limited = True
                print(f"Failed to send GEX feed message to {getattr(channel, 'id', 'unknown')}: {exc}")
                continue
            except Exception as exc:
                print(f"Failed to send GEX feed message to {getattr(channel, 'id', 'unknown')}: {exc}")
                continue
            messages[getattr(channel, 'id', id(channel))] = msg
        if not rate_limited:
            self._reset_feed_backoff()
        return messages

    async def _edit_feed_messages(self, messages: Dict[int, discord.Message], content: str) -> None:
        stale_ids = []
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

    def _build_feed_delta_map(self, tracker: 'RollingWindowTracker', data: dict, now: datetime) -> Dict[str, 'MetricSnapshot']:
        mapping: Dict[str, MetricSnapshot] = {}
        mapping['spot'] = tracker.update('spot', data.get('spot_price'), now)
        mapping['zero_gamma'] = tracker.update('zero_gamma', data.get('zero_gamma'), now)
        mapping['call_wall'] = tracker.update('call_wall', data.get('major_pos_vol'), now)
        mapping['put_wall'] = tracker.update('put_wall', data.get('major_neg_vol'), now)
        mapping['net_gex'] = tracker.update('net_gex', data.get('net_gex'), now)
        mapping['scaled_gamma'] = tracker.update('scaled_gamma', data.get('sum_gex_oi'), now)
        max_entry = self._extract_current_maxchange_entry(data)
        max_delta = max_entry[1] if max_entry and isinstance(max_entry[1], (int, float)) else None
        mapping['maxchange'] = tracker.update('maxchange', max_delta, now)
        return mapping

    async def gex(self, ctx, *args):
        """Get GEX snapshot for a symbol. Uses DuckDB first then GEXBot API fallback."""
        symbol, show_full = self._resolve_gex_request(args)
        data = await self.get_gex_data(symbol)
        if data:
            response = self.format_gex(data) if show_full else self.format_gex_short(data)
            await ctx.send(response)
        else:
            await ctx.send('GEX data not available')

    async def status(self, ctx):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get('http://localhost:8877/status')
                if response.status_code == 200:
                    status_data = response.json()
                    # Format the status nicely
                    formatted = self.format_status(status_data)
                    await ctx.send(formatted)
                else:
                    await ctx.send(f'Failed to fetch status: {response.status_code}')
        except Exception as e:
            await ctx.send(f'Error fetching status: {e}')

    async def tastytrade(self, ctx):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get('http://localhost:8877/status')
                if response.status_code == 200:
                    status_data = response.json()
                    tt_status = status_data.get('tastytrade_streamer', {})
                    formatted = "**TastyTrade Status:**\n"
                    formatted += f"Running: {tt_status.get('running', False)}\n"
                    formatted += f"Trade Samples: {tt_status.get('trade_samples', 0)}\n"
                    formatted += f"Last Trade: {tt_status.get('last_trade_ts', 'N/A')}\n"
                    formatted += f"Depth Samples: {tt_status.get('depth_samples', 0)}\n"
                    formatted += f"Last Depth: {tt_status.get('last_depth_ts', 'N/A')}\n"
                    await ctx.send(formatted)
                else:
                    await ctx.send(f'Failed to fetch status: {response.status_code}')
        except Exception as e:
            await ctx.send(f'Error fetching TastyTrade status: {e}')

    async def get_gex_data(self, symbol: str):
        """Async dual-source retrieval: Redis cache -> DuckDB -> GEXBot API fallback.

        Freshness classification (seconds):
        - current: <=5
        - stale: 5-30
        - incomplete: >30
        """
        display_symbol = (symbol or 'QQQ').upper()
        ticker = self.ticker_aliases.get(display_symbol, display_symbol)
        cache_key = f"gex:{ticker.lower()}:latest"
        snapshot_key = f"{self.redis_snapshot_prefix}{ticker.upper()}"
        async def finalize(data: dict) -> dict:
            data.setdefault('display_symbol', display_symbol)
            await self._populate_wall_ladders(data, ticker, cache_key, snapshot_key)
            return data

        # 1) Try Redis cache (fast)
        try:
            cached = await asyncio.to_thread(self.redis_client.get, cache_key)
            if cached:
                data = json.loads(cached)
                # normalize timestamp to datetime
                ts = data.get('timestamp')
                if isinstance(ts, str):
                    try:
                        data['timestamp'] = datetime.fromisoformat(ts)
                    except Exception:
                        data['timestamp'] = datetime.fromtimestamp(float(ts)).astimezone(timezone.utc)

                # check freshness
                now = datetime.now(timezone.utc)
                rec_ts = data.get('timestamp') or now
                if isinstance(rec_ts, datetime):
                    age = (now - (rec_ts if rec_ts.tzinfo else rec_ts.replace(tzinfo=timezone.utc))).total_seconds()
                else:
                    age = 9999

                complete = self._has_snapshot_coverage(data)

                if age <= 5 and complete:
                    data['_freshness'] = 'current'
                    # mark as redis cache-origin
                    data['_source'] = 'redis-cache'
                    return await finalize(data)
                if age <= 30 and complete:
                    # accept stale but attempt background refresh via API
                    data['_freshness'] = 'stale'
                    data['_source'] = 'redis-cache'
                    asyncio.create_task(self._refresh_gex_from_api(ticker, cache_key, display_symbol))
                    return await finalize(data)
                # cached entry exists but lacks full snapshot; force refresh path
                await asyncio.to_thread(self.redis_client.delete, cache_key)
        except Exception as e:
            print(f"Redis check failed: {e}")

        # 1b) Try canonical snapshot key populated by the data pipeline
        try:
            snapshot = await asyncio.to_thread(self.redis_client.get, snapshot_key)
            if snapshot:
                normalized = self._normalize_snapshot_payload(snapshot, ticker)
                if normalized:
                    now = datetime.now(timezone.utc)
                    ts = normalized.get('timestamp') or now
                    if isinstance(ts, datetime):
                        age = (now - (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc))).total_seconds()
                    else:
                        age = 9999
                    normalized['display_symbol'] = display_symbol
                    if age <= 5:
                        normalized['_freshness'] = 'current'
                        # snapshot returned from the pipeline; mark as redis snapshot
                        normalized['_source'] = 'redis-snapshot'
                        await asyncio.to_thread(self.redis_client.setex, cache_key, 300, json.dumps(normalized, default=str))
                        return await finalize(normalized)
                    if age <= 30:
                        normalized['_freshness'] = 'stale'
                        normalized['_source'] = 'redis-snapshot'
                        await asyncio.to_thread(self.redis_client.setex, cache_key, 300, json.dumps(normalized, default=str))
                        asyncio.create_task(self._refresh_gex_from_api(ticker, cache_key, display_symbol))
                        return await finalize(normalized)
                    normalized['_freshness'] = 'incomplete'
                    normalized['_source'] = 'redis-snapshot'
                    await asyncio.to_thread(self.redis_client.setex, cache_key, 300, json.dumps(normalized, default=str))
                    return await finalize(normalized)
        except Exception as e:
            print(f"Snapshot redis check failed for {snapshot_key}: {e}")

        # 2) Query local DuckDB snapshot before resorting to live API
        try:
            def query_db():
                import duckdb
                q = f"""
                SELECT timestamp, ticker, spot_price, zero_gamma, net_gex,
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
                columns = ['timestamp', 'ticker', 'spot_price', 'zero_gamma', 'net_gex',
                          'major_pos_vol', 'major_neg_vol', 'major_pos_oi', 'major_neg_oi', 'sum_gex_oi', 'max_priors']
                data = dict(zip(columns, result))
                data['display_symbol'] = display_symbol
                data['_source'] = 'DB'
                # Normalize timestamp
                ts = data.get('timestamp')
                if isinstance(ts, str):
                    try:
                        data['timestamp'] = datetime.fromisoformat(ts)
                    except Exception:
                        data['timestamp'] = datetime.fromtimestamp(float(ts)).astimezone(timezone.utc)
                elif isinstance(ts, (int, float)):
                    data['timestamp'] = datetime.fromtimestamp(float(ts)).astimezone(timezone.utc)

                # Parse max_priors
                if isinstance(data.get('max_priors'), str):
                    try:
                        data['max_priors'] = json.loads(data['max_priors'])
                    except Exception:
                        data['max_priors'] = []

                # determine freshness
                now = datetime.now(timezone.utc)
                rec_ts = data.get('timestamp') or now
                age = (now - (rec_ts if isinstance(rec_ts, datetime) and rec_ts.tzinfo else rec_ts.replace(tzinfo=timezone.utc))).total_seconds()
                if age <= 5:
                    data['_freshness'] = 'current'
                elif age <= 30:
                    data['_freshness'] = 'stale'
                else:
                    data['_freshness'] = 'incomplete'

                # cache result for future
                await asyncio.to_thread(self.redis_client.setex, cache_key, 300, json.dumps(data, default=str))
                return await finalize(data)
        except Exception as e:
            print(f"DuckDB query failed: {e}")

        # 3) Finally, poll the live API as a last resort
        try:
            api_data = await self._poll_gexbot_api(ticker)
            if api_data:
                api_data['_freshness'] = 'current'
                api_data['_source'] = 'API'
                api_data['display_symbol'] = display_symbol
                await asyncio.to_thread(self.redis_client.setex, cache_key, 300, json.dumps(api_data, default=str))
                try:
                    await asyncio.to_thread(self.redis_client.set, snapshot_key, json.dumps(api_data, default=str))
                except Exception:
                    pass
                return await finalize(api_data)
        except Exception as e:
            print(f"API fetch failed: {e}")

        return None

    async def _poll_gexbot_api(self, ticker: str):
        """Poll the zero endpoint only, limiting outbound requests."""
        api_key = os.getenv('GEXBOT_API_KEY', 'XXXXXXXXXXXX')
        base = 'https://api.gexbot.com'
        zero_url = f"{base}/{ticker}/classic/zero?key={api_key}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            responses = {'zero': None}
            try:
                resp = await client.get(zero_url)
                if resp.status_code == 200:
                    responses['zero'] = resp.json()
            except Exception:
                responses['zero'] = None

        # Normalize into expected schema
        now = datetime.now(timezone.utc)
        data = {
            'timestamp': now,
            'ticker': ticker,
            'spot_price': None,
            'zero_gamma': None,
            'net_gex': None,
            'major_pos_vol': None,
            'major_neg_vol': None,
            'major_pos_oi': None,
            'major_neg_oi': None,
            'sum_gex_oi': None,
            'max_priors': [],
            'strikes': [],
        }

        z = responses.get('zero')
        if z:
            # common fields
            data['spot_price'] = z.get('spot_price') or z.get('price') or data['spot_price']
            data['zero_gamma'] = z.get('zero_gamma') or z.get('zero_gamma_vol') or data['zero_gamma']
            data['net_gex'] = z.get('net_gex') or z.get('sum_gex') or data['net_gex']
            data['major_pos_vol'] = z.get('major_pos_vol') or data['major_pos_vol']
            data['major_neg_vol'] = z.get('major_neg_vol') or data['major_neg_vol']
            data['major_pos_oi'] = z.get('major_pos_oi') or data['major_pos_oi']
            data['major_neg_oi'] = z.get('major_neg_oi') or data['major_neg_oi']
            data['sum_gex_oi'] = z.get('sum_gex_oi') or z.get('net_gex_oi') or data['sum_gex_oi']
            strikes = z.get('strikes')
            if strikes:
                data['strikes'] = strikes

        return data

    async def _refresh_gex_from_api(self, ticker: str, cache_key: str, display_symbol: Optional[str] = None):
        """Background task to refresh cached snapshot from GEXBot API."""
        try:
            api_data = await self._poll_gexbot_api(ticker)
            if api_data:
                api_data['_freshness'] = 'current'
                if display_symbol:
                    api_data['display_symbol'] = display_symbol
                await asyncio.to_thread(self.redis_client.setex, cache_key, 300, json.dumps(api_data, default=str))
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
            if token.lower() == 'full':
                show_full = True
                continue
            if symbol is None:
                symbol = token.upper()
        return (symbol or 'QQQ'), show_full

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

        ts = payload.get('timestamp') or payload.get('ts')
        if isinstance(ts, str):
            try:
                parsed_ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except ValueError:
                parsed_ts = None
        elif isinstance(ts, (int, float)):
            parsed_ts = datetime.fromtimestamp(float(ts) / (1000 if ts > 1e12 else 1), tz=timezone.utc)
        elif isinstance(ts, datetime):
            parsed_ts = ts
        else:
            parsed_ts = None
        if parsed_ts and parsed_ts.tzinfo is None:
            parsed_ts = parsed_ts.replace(tzinfo=timezone.utc)

        def get_first(*values):
            for value in values:
                if value not in (None, ''):
                    return value
            return None

        data = {
            'timestamp': parsed_ts,
            'ticker': ticker,
            'spot_price': get_first(payload.get('spot'), payload.get('spot_price'), payload.get('price')),
            'zero_gamma': get_first(payload.get('zero_gamma'), payload.get('zero_gamma_vol')),
            'net_gex': payload.get('net_gex') or payload.get('sum_gex'),
            'major_pos_vol': payload.get('major_pos_vol'),
            'major_neg_vol': payload.get('major_neg_vol'),
            'major_pos_oi': payload.get('major_pos_oi'),
            'major_neg_oi': payload.get('major_neg_oi'),
            'sum_gex_oi': payload.get('sum_gex_oi') or payload.get('net_gex_oi'),
            'max_priors': self._extract_max_priors(payload),
            'maxchange': payload.get('maxchange') if isinstance(payload.get('maxchange'), dict) else {},
            'strikes': payload.get('strikes'),
        }
        return data

    def _extract_max_priors(self, payload: dict) -> List[list]:
        maxchange = payload.get('maxchange')
        normalized: List[list] = []
        mapping = [
            ('one', 1),
            ('five', 5),
            ('ten', 10),
            ('fifteen', 15),
            ('thirty', 30),
        ]
        if isinstance(maxchange, dict):
            for key, interval in mapping:
                entry = maxchange.get(key)
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    strike, delta = entry[:2]
                    normalized.append([interval, strike, delta])
            if normalized:
                return normalized

        fallback = payload.get('max_priors') or payload.get('priors') or []
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
            'spot_price',
            'major_pos_vol',
            'major_neg_vol',
            'major_pos_oi',
            'major_neg_oi',
            'sum_gex_oi',
        ]
        for field in required:
            value = payload.get(field)
            if value is None:
                return False
        max_priors = payload.get('max_priors')
        if not max_priors:
            return False
        return True

    async def _populate_wall_ladders(self, data: dict, ticker: str, cache_key: str, snapshot_key: str) -> None:
        if not isinstance(data, dict) or not ticker:
            return
        if data.get('_wall_ladders_ready'):
            return
        summary = await self._build_wall_ladders(data, ticker, cache_key, snapshot_key)
        if summary:
            data['_wall_ladders'] = summary
        data['_wall_ladders_ready'] = True

    async def _build_wall_ladders(self, data: dict, ticker: str, cache_key: str, snapshot_key: str) -> Optional[dict]:
        strikes = self._normalize_strike_entries(data.get('strikes'))
        source = 'payload' if strikes else None
        if not strikes:
            strikes, source = await self._load_strikes_from_cache(ticker, cache_key, snapshot_key)
        if not strikes:
            ts_ms = self._timestamp_to_epoch_ms(data.get('timestamp'))
            if ts_ms is not None:
                strikes, source = await asyncio.to_thread(self._query_strikes_from_db, ticker, ts_ms)
        if not strikes:
            return None
        summary = {
            'call': self._summarize_wall_ladder(data.get('major_pos_vol'), strikes, prefer_positive=True),
            'put': self._summarize_wall_ladder(data.get('major_neg_vol'), strikes, prefer_positive=False),
        }
        summary = {k: v for k, v in summary.items() if v}
        if not summary:
            return None
        if source:
            summary['source'] = source
        return summary

    async def _load_strikes_from_cache(
        self,
        ticker: str,
        cache_key: str,
        snapshot_key: str,
    ) -> Tuple[List[Tuple[float, float]], Optional[str]]:
        keys = [cache_key, snapshot_key]
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
            strikes = self._normalize_strike_entries(payload.get('strikes'))
            if strikes:
                source = 'redis-cache' if key == cache_key else 'redis-snapshot'
                return strikes, source
        return [], None

    def _query_strikes_from_db(self, ticker: str, epoch_ms: int) -> Tuple[List[Tuple[float, float]], Optional[str]]:
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
            source = 'duckdb'
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
                    strike = entry.get('strike') or entry.get('strike_price')
                    gamma = entry.get('gamma') or entry.get('total_gamma')
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
        raw = os.getenv('DISCORD_ADMIN_USER_IDS')
        if raw:
            for part in raw.split(','):
                part = part.strip()
                if not part:
                    continue
                try:
                    ids.add(int(part))
                except ValueError:
                    continue
        owner = os.getenv('DISCORD_OWNER_ID')
        if owner:
            try:
                ids.add(int(owner))
            except ValueError:
                pass
        return ids

    def _parse_admin_names(self) -> set[str]:
        raw = os.getenv('DISCORD_ADMIN_USERNAMES', 'skint0552')
        names = {name.strip().lower() for name in raw.split(',') if name.strip()}
        if not names:
            names = {'skint0552'}
        return names

    def _init_tastytrade_client(self) -> Optional[TastyTradeClient]:
        client_secret = os.getenv('TASTYTRADE_CLIENT_SECRET')
        refresh_token = os.getenv('TASTYTRADE_REFRESH_TOKEN')
        if not client_secret or not refresh_token:
            return None
        default_account = os.getenv('TASTYTRADE_ACCOUNT')
        use_sandbox = os.getenv('TASTYTRADE_USE_SANDBOX', 'false').lower() == 'true'
        try:
            return TastyTradeClient(
                client_secret=client_secret,
                refresh_token=refresh_token,
                default_account=default_account,
                use_sandbox=use_sandbox,
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
        author_name = getattr(ctx.author, 'name', '') or ''
        if author_name.lower() in self.command_admin_names:
            return True
        return False

    async def _ensure_privileged(self, ctx) -> bool:
        if self._is_privileged_user(ctx):
            return True
        await ctx.send('You are not authorized to use this command.')
        return False

    def _init_alert_channels(self) -> List[int]:
        specific = getattr(self.config, 'uw_channel_ids', None) or ()
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
        channel_id = getattr(channel, 'id', None)
        if channel_id is None:
            return True
        return int(channel_id) in self.allowed_channel_ids

    async def _ensure_status_channel_access(self, ctx) -> bool:
        if not self._is_privileged_user(ctx):
            await ctx.send('You are not authorized to use this command.')
            return False
        if self.status_channel_id and getattr(ctx.channel, 'id', None) != self.status_channel_id:
            await ctx.send('This command may only be used in the status channel.')
            return False
        author_name = (getattr(ctx.author, 'name', '') or '').lower()
        if author_name != self.status_command_user:
            await self._send_dm_or_warn(ctx, 'Status commands are restricted to authorized users.')
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
                if message.get('type') != 'message':
                    continue
                data = message.get('data')
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
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
            raw = await asyncio.to_thread(self.redis_client.get, self.uw_market_latest_key)
        except Exception as exc:
            print(f"Redis error fetching market snapshot: {exc}")
            return None
        if not raw:
            try:
                raw = await asyncio.to_thread(self.redis_client.lindex, self.uw_market_history_key, 0)
            except Exception as exc:
                print(f"Redis error fetching historical market snapshot: {exc}")
                raw = None
            if not raw:
                return None
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        try:
            return json.loads(raw)
        except Exception as exc:
            print(f"Failed to decode market snapshot: {exc}")
            return None

    async def _fetch_option_history(self, limit: int = 5) -> List[dict]:
        try:
            entries = await asyncio.to_thread(self.redis_client.lrange, self.uw_option_history_key, 0, limit - 1)
        except Exception as exc:
            print(f"Redis error fetching option history: {exc}")
            return []
        results: List[dict] = []
        for entry in entries:
            if isinstance(entry, bytes):
                entry = entry.decode('utf-8')
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
            return await asyncio.to_thread(self.tastytrade_client.get_account_overview)
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
            await ctx.send('Unable to DM you. Check your privacy settings.')

    def _format_wall_value_line(
        self,
        label_text: str,
        volume_value,
        oi_value,
        fmt_price,
        *,
        label_width: int,
        volume_width: int,
        gap: str = '',
    ) -> str:
        vol = fmt_price(volume_value)
        oi = fmt_price(oi_value) if oi_value is not None else ''
        label = f"{label_text:<{label_width}}"
        spacer = gap if gap is not None else ''
        return f"{label}{spacer}{vol:<{volume_width}}{oi}"

    def _format_wall_short_line(
        self,
        label_text: str,
        value,
        fmt_price,
        *,
        label_width: int,
        gap: str = '',
    ) -> str:
        spacer = gap if gap is not None else ''
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
        ladders = data.get('_wall_ladders')
        summary = ladders.get(ladder_key) if isinstance(ladders, dict) else None
        if not summary or not summary.get('entries'):
            fallback_value = data.get('major_pos_vol' if ladder_key == 'call' else 'major_neg_vol')
            if default_line:
                return default_line
            gap = gap_override if gap_override is not None else ''
            return f"{label_text:<{label_width}}{gap}{fmt_price(fallback_value)}"
        label = f"{label_text:<{label_width}}"
        major = fmt_price(summary.get('major_strike') or (data.get('major_pos_vol') if ladder_key == 'call' else data.get('major_neg_vol')))
        segments = [major]
        for entry in summary.get('entries', [])[:2]:
            strike_txt = fmt_price(entry.get('strike'))
            pct = entry.get('pct_vs_major')
            pct_txt = f"{pct:.0f}%" if isinstance(pct, (int, float)) else "N/A%"
            segments.append(f"{strike_txt} {pct_txt}")
        gap = gap_override if gap_override is not None else '  '
        return f"{label}{gap}{'  '.join(segments)}"

    def _resolve_gex_source_label(self, data: dict) -> str:
        # Primary: explicit source set on the data (set in get_gex_data)
        src = data.get('_source')
        if isinstance(src, str):
            s = src.strip()
            if not s:
                pass
            else:
                lowered = s.lower()
                if lowered == 'cache':
                    return 'cache'
                if 'redis' in lowered:
                    return 'redis'
                if lowered == 'db' or 'duckdb' in lowered:
                    return 'DB'
                if lowered == 'api' or 'payload' in lowered or 'api' in lowered:
                    return 'API'
                return s

        # Secondary: derived from wall ladder source (redis-snapshot, redis-cache, duckdb, payload)
        ladders = data.get('_wall_ladders')
        if isinstance(ladders, dict):
            raw = ladders.get('source')
            if raw:
                lowered = str(raw).lower()
                if 'redis' in lowered:
                    return 'redis'
                if 'duckdb' in lowered:
                    return 'DB'
                if 'payload' in lowered or 'api' in lowered:
                    return 'API'
                return str(raw)

        # Fallback: use freshness or 'local'
        freshness = data.get('_freshness')
        if isinstance(freshness, str):
            return freshness
        return 'local'

    def _extract_current_maxchange_entry(self, data: dict) -> Optional[Tuple[Any, Any]]:
        maxchange = data.get('maxchange') if isinstance(data.get('maxchange'), dict) else data.get('maxchange')
        entry = None
        if isinstance(maxchange, dict):
            entry = maxchange.get('current')
        if not (isinstance(entry, (list, tuple)) and len(entry) >= 2):
            priors = data.get('max_priors') or []
            if priors and isinstance(priors[0], (list, tuple)) and len(priors[0]) >= 3:
                entry = [priors[0][1], priors[0][2]]
        if not (isinstance(entry, (list, tuple)) and len(entry) >= 2):
            return None
        if len(entry) >= 3 and isinstance(entry[1], (int, float)) and isinstance(entry[2], (int, float)):
            strike_val = entry[1]
            delta_val = entry[2]
        else:
            strike_val = entry[0]
            delta_val = entry[1]
        return strike_val, delta_val

    def format_gex(self, data):
        dt = data.get('timestamp')
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
        formatted_time = local_dt.strftime("%m/%d/%Y  %I:%M:%S %p %Z") if local_dt else "N/A"

        ticker = data.get('display_symbol') or data.get('ticker', 'QQQ')

        def fmt_price(x):
            return f"{x:.2f}" if isinstance(x, (int, float)) else "N/A"

        # Keep gamma display as-is (no scaling)

        # Net GEX formatting: positive values show with Bn suffix, negatives show whole number
        def fmt_net_gex(x):
            if not isinstance(x, (int, float)):
                return "N/A"
            # Display magnitude scaled by 1000 with Bn suffix for both signs
            return f"{(abs(x)/1000):.4f}Bn"

        def fmt_net_abs(x):
            return f"{abs(x):.5f}" if isinstance(x, (int, float)) else "N/A"

        ansi = {
            'reset': "\u001b[0m",
            'dim_white': "\u001b[2;37m",
            'yellow': "\u001b[2;33m",
            'green': "\u001b[2;32m",
            'red': "\u001b[2;31m",
        }

        def colorize(code, text):
            if not code:
                return text
            return f"{code}{text}{ansi['reset']}"

        def color_for_value(value, neutral='yellow'):
            if not isinstance(value, (int, float)):
                return neutral
            if value > 0:
                return 'green'
            if value < 0:
                return 'red'
            return neutral

        label_width = 19
        volume_width = 22

        def fmt_pair(label: str, volume_value, oi_value=None, *, volume_color=None, oi_color=None, formatter=None):
            formatter = formatter or fmt_price
            vol = formatter(volume_value)
            oi = formatter(oi_value) if oi_value is not None else ''
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
        call_gap = ' ' * 3
        put_gap = ' ' * 3
        call_wall_line = self._format_wall_line(
            data,
            'call',
            'call wall',
            fmt_price,
            label_width=wall_label_width,
            gap_override=call_gap,
            default_line=self._format_wall_value_line(
                'call wall',
                data.get('major_pos_vol'),
                data.get('major_pos_oi'),
                fmt_price,
                label_width=wall_label_width,
                volume_width=volume_width,
                gap=call_gap,
            ),
        )
        put_wall_line = self._format_wall_line(
            data,
            'put',
            'put wall',
            fmt_price,
            label_width=wall_label_width,
            gap_override=put_gap,
            default_line=self._format_wall_value_line(
                'put wall',
                data.get('major_neg_vol'),
                data.get('major_neg_oi'),
                fmt_price,
                label_width=wall_label_width,
                volume_width=volume_width,
                gap=put_gap,
            ),
        )

        table_lines = [
            "volume                                   oi",
            fmt_pair('zero gamma', data.get('zero_gamma'), volume_color='yellow', formatter=fmt_price),
            call_wall_line,
            put_wall_line,
            fmt_pair(
                'net gex',
                data.get('net_gex'),
                data.get('sum_gex_oi'),
                volume_color=color_for_value(data.get('net_gex')),
                oi_color=color_for_value(data.get('sum_gex_oi')),
                formatter=fmt_net_gex,
            ),
        ]

        maxchange_lines = ["", "max change gex"]
        current_entry = self._extract_current_maxchange_entry(data)
        def fmt_delta_entry(label: str, entry):
            if not (isinstance(entry, (list, tuple)) and len(entry) >= 2):
                return f"{label:<18}N/A  N/ABn"
            if len(entry) >= 3 and isinstance(entry[1], (int, float)) and isinstance(entry[2], (int, float)):
                strike_val = entry[1]
                delta = entry[2]
            else:
                strike_val = entry[0]
                delta = entry[1]
            strike = fmt_price(strike_val) if isinstance(strike_val, (int, float)) else str(strike_val)
            delta_color = ansi.get(color_for_value(delta))
            delta_text = fmt_net_gex(delta) if isinstance(delta, (int, float)) else str(delta)
            return f"{label:<18}{strike:<8} {colorize(delta_color, delta_text)}"

        maxchange_lines.append(fmt_delta_entry('current', current_entry))

        max_priors = data.get('max_priors', []) or []
        intervals = [1, 5, 10, 15, 30]
        for i, interval in enumerate(intervals):
            entry = max_priors[i] if i < len(max_priors) else None
            label = f"{interval} min"
            maxchange_lines.append(fmt_delta_entry(label, entry))

        body = "\n".join([header, "", *table_lines, "", *maxchange_lines])
        return f"```ansi\n{body}\n```"

    def format_gex_short(
        self,
        data,
        *,
        include_time: bool = True,
        time_format: Optional[str] = None,
        delta_block: Optional[Dict[str, 'MetricSnapshot']] = None,
    ):
        dt = data.get('timestamp')
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

        ticker = data.get('display_symbol') or data.get('ticker', 'QQQ')

        def fmt_price(x):
            return f"{x:.2f}" if isinstance(x, (int, float)) else "N/A"

        # Keep gamma display as-is (no scaling)

        # Net GEX formatting as above
        def fmt_net_gex(x):
            if not isinstance(x, (int, float)):
                return "N/A"
            # Display magnitude scaled by 1000 with Bn suffix for both signs
            return f"{(abs(x)/1000):.4f}Bn"

        def fmt_net_abs(x):
            return f"{abs(x):.5f}" if isinstance(x, (int, float)) else "N/A"

        ansi = {
            'reset': "\u001b[0m",
            'dim_white': "\u001b[2;37m",
            'yellow': "\u001b[2;33m",
            'green': "\u001b[2;32m",
            'red': "\u001b[2;31m",
        }

        def colorize(code, text):
            if not code:
                return text
            return f"{code}{text}{ansi['reset']}"

        def color_for_value(value, neutral='yellow'):
            if not isinstance(value, (int, float)):
                return neutral
            if value > 0:
                return 'green'
            if value < 0:
                return 'red'
            return neutral

        zero_gamma_line = f"zero gamma        {colorize(ansi['yellow'], fmt_price(data.get('zero_gamma')))}"
        wall_label_width = 15
        call_gap = ' ' * 3
        put_gap = ' ' * 3
        major_pos_line = self._format_wall_line(
            data,
            'call',
            'call wall',
            fmt_price,
            label_width=wall_label_width,
            gap_override=call_gap,
            default_line=self._format_wall_short_line(
                'call wall',
                data.get('major_pos_vol'),
                fmt_price,
                label_width=wall_label_width,
                gap=call_gap,
            ),
        )
        major_neg_line = self._format_wall_line(
            data,
            'put',
            'put wall',
            fmt_price,
            label_width=wall_label_width,
            gap_override=put_gap,
            default_line=self._format_wall_short_line(
                'put wall',
                data.get('major_neg_vol'),
                fmt_price,
                label_width=wall_label_width,
                gap=put_gap,
            ),
        )
        net_color_code = ansi.get(color_for_value(data.get('net_gex')))
        net_value = colorize(net_color_code, fmt_net_gex(data.get('net_gex')))

        current_entry = self._extract_current_maxchange_entry(data)
        if isinstance(current_entry, tuple):
            price_val, delta_val = current_entry
            current_price = fmt_price(price_val)
            delta_text = fmt_net_gex(delta_val) if isinstance(delta_val, (int, float)) else str(delta_val)
            delta_color = ansi.get(color_for_value(delta_val))
            current_delta = colorize(delta_color, delta_text)
        else:
            fallback = ansi['yellow']
            current_price = 'N/A'
            current_delta = colorize(fallback, 'N/ABn')

        source_label = self._resolve_gex_source_label(data)
        def snapshot(key: str) -> 'MetricSnapshot':
            snap = delta_block.get(key) if delta_block else None
            if isinstance(snap, MetricSnapshot):
                return snap
            return MetricSnapshot(value=None, delta=None, percent=None, previous=None, baseline=None)

        def fmt_abs(value, digits: int = 2) -> str:
            if not isinstance(value, (int, float)):
                return 'N/A'
            return f"{abs(value):.{digits}f}"

        def fmt_percent(value) -> str:
            if not isinstance(value, (int, float)):
                return 'n/a%'
            return f"{value:+.2f}%"

        def color_for_delta(delta) -> str:
            if not isinstance(delta, (int, float)):
                return 'yellow'
            return 'green' if delta >= 0 else 'red'

        header_parts = [f"{ansi['dim_white']}GEX: {ticker}{ansi['reset']}"]
        if include_time:
            header_parts.append(formatted_time)

        price_text = fmt_price(data.get('spot_price'))
        if delta_block:
            spot_snap = snapshot('spot')
            delta_text = fmt_abs(spot_snap.delta)
            percent_text = fmt_percent(spot_snap.percent)
            delta_color = ansi.get(color_for_delta(spot_snap.delta))
            delta_display = colorize(delta_color, f"Δ{delta_text} {percent_text}")
            price_text = f"{price_text}  {delta_display}"
        header_parts.append(f"{ansi['dim_white']}{price_text}{ansi['reset']}")
        header_parts.append(f"{ansi['dim_white']}{source_label}{ansi['reset']}")
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
            lines = [
                header,
                "",
                zero_gamma_line,
                major_pos_line,
                major_neg_line,
                f"net gex           {net_value}",
                f"current           {current_price:<8}   {current_delta}",
            ]
        body = "\n".join(lines)
        return f"```ansi\n{body}\n```"

    def _format_feed_short_lines(
        self,
        header: str,
        data: dict,
        delta_block: Dict[str, 'MetricSnapshot'],
        ansi: Dict[str, str],
        colorize,
        color_for_value,
        fmt_price,
        fmt_net_gex,
    ) -> List[str]:
        def fmt_abs(value, digits: int = 2) -> str:
            if not isinstance(value, (int, float)):
                return 'N/A'
            return f"{abs(value):.{digits}f}"

        def fmt_percent(value) -> str:
            if not isinstance(value, (int, float)):
                return 'n/a%'
            return f"{abs(value):.2f}%"

        def color_for_delta(delta) -> str:
            if not isinstance(delta, (int, float)):
                return 'yellow'
            return 'green' if delta >= 0 else 'red'

        def snapshot(key: str) -> 'MetricSnapshot':
            snap = delta_block.get(key)
            if isinstance(snap, MetricSnapshot):
                return snap
            return MetricSnapshot(value=None, delta=None, percent=None, previous=None, baseline=None)

        def fmt_prev(value) -> str:
            if not isinstance(value, (int, float)):
                return 'n/a'
            return fmt_abs(value)

        lines = [header, ""]

        def append_prev_line(label: str, snap_key: str):
            snap = snapshot(snap_key)
            delta = None
            if isinstance(snap.value, (int, float)) and isinstance(snap.previous, (int, float)):
                delta = snap.value - snap.previous
            color = color_for_delta(delta)
            current_txt = colorize(ansi.get(color), fmt_abs(snap.value))
            prev_txt = fmt_prev(snap.previous)
            lines.append(f"{label:<17}{current_txt:<10} ({prev_txt})")

        append_prev_line('zero gamma', 'zero_gamma')
        append_prev_line('call wall', 'call_wall')
        append_prev_line('put wall', 'put_wall')

        def fmt_net_change(delta) -> str:
            if not isinstance(delta, (int, float)):
                return 'N/ABn'
            return f"{(abs(delta)/1000):.4f}Bn"

        net_snap = snapshot('net_gex')
        net_value = colorize(ansi.get(color_for_value(data.get('net_gex'))), fmt_net_gex(data.get('net_gex')))
        net_change_color = ansi.get(color_for_delta(net_snap.delta))
        net_change_txt = colorize(net_change_color, fmt_net_change(net_snap.delta))
        lines.append(f"net gex           {net_value:<10} Δ{net_change_txt}")

        scaled_snap = snapshot('scaled_gamma')
        scaled_value = fmt_net_gex(data.get('sum_gex_oi'))
        scaled_change_txt = colorize(ansi.get(color_for_delta(scaled_snap.delta)), fmt_net_change(scaled_snap.delta))
        lines.append(f"scaled gamma      {scaled_value:<10} Δ{scaled_change_txt}")

        current_entry = self._extract_current_maxchange_entry(data)
        if current_entry:
            strike_val, delta_val = current_entry
            current_price = fmt_price(strike_val)
            delta_text = fmt_net_gex(delta_val) if isinstance(delta_val, (int, float)) else str(delta_val)
            delta_color = ansi.get(color_for_value(delta_val))
            current_delta = colorize(delta_color, delta_text)
        else:
            current_price = 'N/A'
            current_delta = colorize(ansi['yellow'], 'N/ABn')
        max_snap = snapshot('maxchange')
        max_change_txt = colorize(ansi.get(color_for_delta(max_snap.delta)), fmt_net_change(max_snap.delta))
        lines.append(f"current strike    {current_price:<8}   {current_delta}  Δ{max_change_txt}")
        return lines

    def format_option_trade_alert(self, payload: dict) -> str:
        data = payload.get('data') or payload
        timestamp = data.get('timestamp') or payload.get('received_at') or 'N/A'
        if isinstance(timestamp, str):
            timestamp = timestamp.replace('T', ' ').replace('Z', ' UTC')
        transaction_types = data.get('transaction_type') or data.get('transaction_types') or []
        if isinstance(transaction_types, str):
            transaction_types = [transaction_types]
        ticker = data.get('ticker') or data.get('symbol') or 'UNKNOWN'
        side = data.get('side') or data.get('direction') or 'N/A'
        call_put = data.get('call_put') or data.get('option_type') or ''
        strike = data.get('strike') or data.get('strike_price') or 'N/A'
        contract = data.get('contract') or 'n/a'
        dte = data.get('dte') or data.get('days_to_expiration') or 'N/A'
        stock_spot = data.get('stock_spot') or data.get('underlying_price') or 'N/A'
        bid_range = data.get('bid_ask_range') or data.get('bid_range') or 'N/A'
        option_spot = data.get('option_spot') or data.get('option_price') or 'N/A'
        size = data.get('size') or data.get('contracts') or 'N/A'
        premium = data.get('premium') or data.get('notional') or 'N/A'
        volume = data.get('volume') or 'N/A'
        oi = data.get('oi') or data.get('open_interest') or 'N/A'
        chain_bid = data.get('chain_bid') or data.get('bid') or 'N/A'
        chain_ask = data.get('chain_ask') or data.get('ask') or 'N/A'
        legs = data.get('legs') or []
        code = data.get('code') or 'N/A'
        flags = data.get('flags') or []
        tags = data.get('tags') or []
        uw_info = data.get('unusual_whales') or {}
        uw_id = uw_info.get('alert_id') or 'n/a'
        uw_score = uw_info.get('score')

        legs_text = ", ".join(
            f"{leg.get('ratio', 1)}x{leg.get('strike')} {leg.get('type')}"
            for leg in legs if isinstance(leg, dict)
        ) or "n/a"
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
        data = payload.get('data') or payload
        timestamp = data.get('timestamp') or payload.get('received_at') or 'N/A'
        ticker = data.get('ticker') or data.get('symbol') or 'INDEX'
        stock_spot = data.get('stock_spot') or data.get('price') or 'N/A'
        bid = data.get('bid') or 'N/A'
        ask = data.get('ask') or 'N/A'
        volume = data.get('volume') or 'N/A'
        advancers = data.get('advancers') or 'N/A'
        decliners = data.get('decliners') or 'N/A'
        net_flow = data.get('net_flow') or data.get('net') or 'N/A'
        session = data.get('session') or data.get('market_session') or 'N/A'

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
        def fmt(value):
            try:
                return f"{float(value):,.2f}"
            except (TypeError, ValueError):
                return str(value)

        lines = [
            f"TastyTrade status   account {overview.get('account_number', 'n/a')}",
            f"available funds    {fmt(overview.get('available_trading_funds'))}",
            f"equity BP          {fmt(overview.get('equity_buying_power'))}",
            f"derivative BP      {fmt(overview.get('derivative_buying_power'))}",
            f"day trade BP       {fmt(overview.get('day_trading_buying_power'))}",
            f"net liq            {fmt(overview.get('net_liquidating_value'))}",
            f"cash balance       {fmt(overview.get('cash_balance'))}",
            f"margin equity      {fmt(overview.get('margin_equity'))}",
            f"maintenance req    {fmt(overview.get('maintenance_requirement'))}",
            f"day trade excess   {fmt(overview.get('day_trade_excess'))}",
            f"pending cash       {fmt(overview.get('pending_cash'))}",
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
        return MetricSnapshot(value=value, delta=delta, percent=percent, previous=previous_value, baseline=baseline)


class RedisGexFeedMetrics:
    def __init__(self, redis_client, key: str, *, enabled: bool = False):
        self.redis_client = redis_client
        self.key = key
        self.enabled = enabled
        self._update_count = 0
        self._error_count = 0

    async def record_update(self, now: datetime, delta_map: Dict[str, MetricSnapshot], *, refresh: bool) -> None:
        if not self.enabled:
            return
        self._update_count += 1
        mapping = {
            'last_update_ts': now.isoformat(),
            'update_count': str(self._update_count),
        }
        if refresh:
            mapping['last_refresh_ts'] = now.isoformat()
        mapping.update(self._flatten_delta_map(delta_map))
        await asyncio.to_thread(self.redis_client.hset, self.key, mapping=mapping)

    async def record_error(self, now: datetime, *, reason: str) -> None:
        if not self.enabled:
            return
        self._error_count += 1
        mapping = {
            'last_error_ts': now.isoformat(),
            'error_count': str(self._error_count),
            'last_error_reason': reason,
        }
        await asyncio.to_thread(self.redis_client.hset, self.key, mapping=mapping)

    def _flatten_delta_map(self, delta_map: Dict[str, MetricSnapshot]) -> Dict[str, str]:
        flattened: Dict[str, str] = {}
        for key, snapshot in (delta_map or {}).items():
            if not isinstance(snapshot, MetricSnapshot):
                continue
            flattened[f'{key}_value'] = self._stringify(snapshot.value)
            flattened[f'{key}_delta'] = self._stringify(snapshot.delta)
            flattened[f'{key}_percent'] = self._stringify(snapshot.percent)
        return flattened

    @staticmethod
    def _stringify(value: Optional[float]) -> str:
        if value is None:
            return ''
        return f"{value}"
