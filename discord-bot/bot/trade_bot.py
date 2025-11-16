import discord
from discord.ext import commands
import redis
import json
import os
import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo
import httpx

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
        self.uw_option_latest_key = os.getenv('UW_OPTION_LATEST_KEY', 'uw:option_trades_super_algo:latest')
        self.uw_option_history_key = os.getenv('UW_OPTION_HISTORY_KEY', 'uw:option_trades_super_algo:history')
        self.uw_market_latest_key = os.getenv('UW_MARKET_LATEST_KEY', 'uw:market_agg_socket:latest')
        self.uw_market_history_key = os.getenv('UW_MARKET_HISTORY_KEY', 'uw:market_agg_socket:history')
        self.uw_option_stream_channel = os.getenv('UW_OPTION_STREAM_CHANNEL', 'uw:option_trades_super_algo:stream')
        # Canonical tickers for GEXBot futures endpoints so cache/API reuse shared contracts
        self.ticker_aliases = {
            'NQ': 'NQ_NDX',
            'MNQ': 'NQ_NDX',
            'ES': 'ES_SPX',
            'MES': 'ES_SPX',
        }
        self.display_zone = ZoneInfo(os.getenv('DISPLAY_TIMEZONE', 'America/New_York'))
        self.redis_snapshot_prefix = os.getenv('GEX_SNAPSHOT_PREFIX', 'gex:snapshot:')
        self.option_alert_channel_ids = self._init_alert_channels()
        self._uw_listener_task: Optional[asyncio.Task] = None
        self._uw_listener_stop = asyncio.Event()
        # Use a path variable and create short-lived connections per query to avoid file locks
        self.duckdb_path = os.getenv('DUCKDB_PATH', '/home/rwest/projects/data-pipeline/data/gex_data.db')
        # Register commands defined as methods on the subclass so prefix commands work
        try:
            # Use plain functions (closures) as command callbacks so signature checks pass.
            async def _ping_cmd(ctx):
                await ctx.send('Pong!')

            async def _gex_cmd(ctx, *args):
                symbol, show_full = self._resolve_gex_request(args)
                data = await self.get_gex_data(symbol)
                if data:
                    formatter = self.format_gex if show_full else self.format_gex_short
                    await ctx.send(formatter(data))
                else:
                    await ctx.send('GEX data not available')

            async def _status_cmd(ctx):
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

            async def _market_cmd(ctx):
                snapshot = await self._fetch_market_snapshot()
                if not snapshot:
                    await ctx.send('No market aggregate snapshot available')
                    return
                await ctx.send(self.format_market_snapshot(snapshot))

            async def _uwalerts_cmd(ctx):
                alerts = await self._fetch_option_history(limit=5)
                if not alerts:
                    await ctx.send('No option trade alerts yet')
                    return
                formatted = "\n".join(self.format_option_trade_alert(alert) for alert in alerts)
                await ctx.send(formatted)

            self.add_command(commands.Command(_ping_cmd, name='ping'))
            self.add_command(commands.Command(_gex_cmd, name='gex'))
            self.add_command(commands.Command(_status_cmd, name='status'))
            self.add_command(commands.Command(_tastytrade_cmd, name='tastytrade'))
            self.add_command(commands.Command(_market_cmd, name='market'))
            self.add_command(commands.Command(_uwalerts_cmd, name='uw'))
        except Exception as e:
            # If registration fails (e.g., during import-time tests), print error for debugging
            print(f"Command registration failed: {e}")

    async def on_ready(self):
        print(f'Bot logged in as {self.user}')
        if not self._uw_listener_task:
            self._uw_listener_stop.clear()
            self._uw_listener_task = asyncio.create_task(self._listen_option_trade_stream())

    async def on_message(self, message):
        if message.author == self.user:
            return
        await super().on_message(message)

    async def close(self):
        if self._uw_listener_task:
            self._uw_listener_stop.set()
            try:
                await self._uw_listener_task
            except Exception:
                pass
            self._uw_listener_task = None
        await super().close()

    async def ping(self, ctx):
        await ctx.send('Pong!')

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
                    data.setdefault('display_symbol', display_symbol)
                    return data
                if age <= 30 and complete:
                    # accept stale but attempt background refresh via API
                    data['_freshness'] = 'stale'
                    data.setdefault('display_symbol', display_symbol)
                    # spawn a background refresh
                    asyncio.create_task(self._refresh_gex_from_api(ticker, cache_key, display_symbol))
                    return data
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
                        await asyncio.to_thread(self.redis_client.setex, cache_key, 300, json.dumps(normalized, default=str))
                        return normalized
                    if age <= 30:
                        normalized['_freshness'] = 'stale'
                        await asyncio.to_thread(self.redis_client.setex, cache_key, 300, json.dumps(normalized, default=str))
                        asyncio.create_task(self._refresh_gex_from_api(ticker, cache_key, display_symbol))
                        return normalized
                    normalized['_freshness'] = 'incomplete'
                    await asyncio.to_thread(self.redis_client.setex, cache_key, 300, json.dumps(normalized, default=str))
                    return normalized
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
                return data
        except Exception as e:
            print(f"DuckDB query failed: {e}")

        # 3) Finally, poll the live API as a last resort
        try:
            api_data = await self._poll_gexbot_api(ticker)
            if api_data:
                api_data['_freshness'] = 'current'
                api_data['display_symbol'] = display_symbol
                await asyncio.to_thread(self.redis_client.setex, cache_key, 300, json.dumps(api_data, default=str))
                return api_data
        except Exception as e:
            print(f"API fetch failed: {e}")

        return None

    async def _poll_gexbot_api(self, ticker: str):
        """Poll the zero endpoint only, limiting outbound requests."""
        api_key = os.getenv('GEXBOT_API_KEY', 'mW0CQI9onfOX')
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
        }

        z = responses.get('zero')
        if z:
            # common fields
            data['spot_price'] = z.get('spot_price') or z.get('price') or data['spot_price']
            data['zero_gamma'] = z.get('zero_gamma') or z.get('zero_gamma_vol') or data['zero_gamma']
            data['net_gex'] = z.get('net_gex') or z.get('sum_gex') or data['net_gex']

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

    def _init_alert_channels(self) -> List[int]:
        channels: List[int] = []
        for cid in (
            getattr(self.config, 'execution_channel_id', None),
            getattr(self.config, 'status_channel_id', None),
            getattr(self.config, 'alert_channel_id', None),
        ):
            if cid and cid not in channels:
                channels.append(cid)
        allowed = getattr(self.config, 'allowed_channel_ids', None) or ()
        for cid in allowed:
            if cid and cid not in channels:
                channels.append(cid)
        return channels

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

        def fmt_net(x):
            return f"{x:.5f}" if isinstance(x, (int, float)) else "N/A"

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

        header = (
            f"GEX: {colorize(ansi['dim_white'], ticker)} {formatted_time}  "
            f"{colorize(ansi['dim_white'], fmt_price(data.get('spot_price')))}"
        )

        table_lines = [
            "volume                                   oi",
            fmt_pair('zero gamma', data.get('zero_gamma'), volume_color='yellow'),
            fmt_pair('call wall', data.get('major_pos_vol'), data.get('major_pos_oi')),
            fmt_pair('put wall', data.get('major_neg_vol'), data.get('major_neg_oi')),
            fmt_pair(
                'net gex',
                data.get('net_gex'),
                data.get('sum_gex_oi'),
                volume_color=color_for_value(data.get('net_gex')),
                oi_color=color_for_value(data.get('sum_gex_oi')),
                formatter=fmt_net_abs,
            ),
        ]

        maxchange_lines = ["", "max change gex"]
        maxchange = data.get('maxchange') if isinstance(data.get('maxchange'), dict) else {}
        current_entry = maxchange.get('current') if isinstance(maxchange, dict) else None
        if not (isinstance(current_entry, (list, tuple)) and len(current_entry) >= 2):
            priors = data.get('max_priors') or []
            if priors and isinstance(priors[0], (list, tuple)) and len(priors[0]) >= 3:
                current_entry = [priors[0][1], priors[0][2]]
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
            delta_text = f"{delta:.4f}Bn" if isinstance(delta, (int, float)) else str(delta)
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

    def format_gex_short(self, data):
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

        zero_gamma_line = f"zero gamma          {colorize(ansi['yellow'], fmt_price(data.get('zero_gamma')))}"
        major_pos_line = f"call wall           {fmt_price(data.get('major_pos_vol'))}"
        major_neg_line = f"put wall            {fmt_price(data.get('major_neg_vol'))}"
        net_color_code = ansi.get(color_for_value(data.get('net_gex')))
        net_value = colorize(net_color_code, fmt_net_abs(data.get('net_gex')))

        maxchange = data.get('maxchange') if isinstance(data.get('maxchange'), dict) else {}
        current_entry = maxchange.get('current') if isinstance(maxchange, dict) else None
        if not (isinstance(current_entry, (list, tuple)) and len(current_entry) >= 2):
            priors = data.get('max_priors') or []
            if priors and isinstance(priors[0], (list, tuple)) and len(priors[0]) >= 3:
                current_entry = [priors[0][1], priors[0][2]]
            else:
                current_entry = None

        if isinstance(current_entry, (list, tuple)) and len(current_entry) >= 2:
            price_val = current_entry[0]
            delta_val = current_entry[1]
            current_price = fmt_price(price_val)
            delta_text = f"{delta_val:.4f}Bn" if isinstance(delta_val, (int, float)) else str(delta_val)
            delta_color = ansi.get(color_for_value(delta_val))
            current_delta = colorize(delta_color, delta_text)
        else:
            fallback = ansi['yellow']
            current_price = 'N/A'
            current_delta = colorize(fallback, 'N/ABn')

        header = (
            f"{ansi['dim_white']}GEX: {ticker}{ansi['reset']}  "
            f"{formatted_time}  "
            f"{ansi['dim_white']}{fmt_price(data.get('spot_price'))}{ansi['reset']}"
        )

        lines = [
            header,
            "",
            zero_gamma_line,
            major_pos_line,
            major_neg_line,
            f"net gex             {net_value}",
            f"current             {current_price:<8}   {current_delta}",
        ]
        body = "\n".join(lines)
        return f"```ansi\n{body}\n```"

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
            f"Market aggregate | {ticker}",
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
