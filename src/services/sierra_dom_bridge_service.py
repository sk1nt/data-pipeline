"""sierra_dom_bridge_service.py -- DTC TCP client bridge: Sierra Chart -> Redis.

Connects to Sierra Chart's built-in DTC Protocol Server, subscribes to
Level 2 market depth and trade prints for the configured futures symbol,
and publishes computed DOM / CVD payloads to Redis every POLL_INTERVAL_MS.

No ACSIL study required -- uses Sierra Chart's native DTC server directly.

DTC server prerequisites (Sierra Chart -> Global Settings -> DTC Protocol Server)
    Enable DTC Protocol Server : Yes
    Listening Port             : 11099  (or set SC_DTC_PORT)
    Require Authentication     : No
    Encoding (List)            : JSON Compact

Redis channels published
------------------------
    market:dom:{symbol}
        ts_ms, symbol, price, bids, asks, ofi_1s,
        bid_depth_1/5/10/20, ask_depth_1/5/10/20, imbalance_ratio

    market:cvd:{symbol}
        ts_ms, symbol, cvd_1min, cvd_5min, cvd_15min, cvd_running_day,
        buy_vol_1min, sell_vol_1min

Redis channel consumed
----------------------
    sweep:danger:{symbol}
        Writes danger_trigger.json to SC_DATA_DIR so a position-bridge
        ACSIL study can call sc.FlattenAndCancelAllOrders().

Environment variables
---------------------
    SC_DTC_HOST          Windows host IP from WSL2.
                          Default: auto-detected from /etc/resolv.conf.
                          Override with LAN IP if running remotely.
    SC_DTC_PORT          DTC server TCP port (default: 11099)
    SC_DTC_SYMBOL        Full Sierra Chart symbol (default: MNQM26_FUT_CME)
    SC_DOM_SYMBOL        Short name used in Redis keys (default: MNQ)
    SC_DATA_DIR          WSL path for danger_trigger.json writes
                          (default: /mnt/c/SierraChart/Data)
    SC_DANGER_TRIGGER_PATH  Override full path for danger_trigger.json
    REDIS_URL            (default: redis://localhost:6379/0)
    SC_BRIDGE_POLL_MS    Redis publish interval in ms (default: 100)
    SC_BRIDGE_STALE_MS   Max DOM age before suppressing publish (default: 2000)
    MNQ_TICK_VALUE       Price units per tick (default: 0.25)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import redis.asyncio as aioredis

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DTC protocol constants
# ---------------------------------------------------------------------------
_DTC_PROTOCOL_VERSION          = 7
_JSON_COMPACT_ENCODING         = 4

_T_LOGON_REQUEST               = 1
_T_LOGON_RESPONSE              = 2
_T_HEARTBEAT                   = 3
_T_ENCODING_REQUEST            = 6
_T_ENCODING_RESPONSE           = 7
_T_MARKET_DATA_REQUEST         = 101
_T_MARKET_DATA_REJECT          = 103
_T_MARKET_DATA_SNAPSHOT        = 104
_T_MARKET_DATA_UPDATE_TRADE    = 107
_T_MARKET_DEPTH_REQUEST        = 102
_T_MARKET_DEPTH_REJECT         = 121
_T_MARKET_DEPTH_SNAPSHOT_L2    = 122
_T_MARKET_DEPTH_UPDATE_L2      = 123
_T_MARKET_DATA_UPDATE_TRADE_COMPACT = 112

_SIDE_ASK      = 0
_SIDE_BID      = 1
_UPDATE_ADD    = 1
_UPDATE_DELETE = 2

_AT_BID        = 1   # sell-side aggressor
_AT_ASK        = 2   # buy-side aggressor

_SYMBOL_ID     = 1   # arbitrary SymbolID for this session

# 16-byte binary ENCODING_REQUEST (always sent as binary before JSON negotiation)
_ENCODING_REQUEST_BYTES = struct.pack(
    "<HHii4s",
    16,                        # Size (total message bytes)
    _T_ENCODING_REQUEST,       # Type = 6
    _DTC_PROTOCOL_VERSION,     # ProtocolVersion = 7
    _JSON_COMPACT_ENCODING,    # Encoding = 4
    b"DTC\x00",                # ProtocolType
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SC_DATA_DIR        = os.getenv("SC_DATA_DIR",            "/mnt/c/SierraChart/Data")
SC_DANGER_PATH_ENV = os.getenv("SC_DANGER_TRIGGER_PATH", "")

SC_DTC_PORT        = int(os.getenv("SC_DTC_PORT",    "11099"))
SC_DTC_SYMBOL      = os.getenv("SC_DTC_SYMBOL",      "MNQM26_FUT_CME")
DOM_SYMBOL         = os.getenv("SC_DOM_SYMBOL",      "MNQ")

REDIS_URL          = os.getenv("REDIS_URL",           "redis://localhost:6379/0")
POLL_INTERVAL_MS   = int(os.getenv("SC_BRIDGE_POLL_MS",  "100"))
STALE_THRESHOLD_MS = int(os.getenv("SC_BRIDGE_STALE_MS", "2000"))
MNQ_TICK_SIZE      = float(os.getenv("MNQ_TICK_VALUE",    "0.25"))

DOM_LEVELS       = 20
HEARTBEAT_S      = 10
RECONNECT_BASE_S = 2.0
RECONNECT_MAX_S  = 60.0


def _resolve_danger_path() -> Path:
    if SC_DANGER_PATH_ENV:
        return Path(SC_DANGER_PATH_ENV)
    return Path(SC_DATA_DIR) / "danger_trigger.json"


def _detect_windows_host() -> str:
    """In WSL2, the Windows host IP is the nameserver in /etc/resolv.conf."""
    try:
        with open("/etc/resolv.conf") as fh:
            for line in fh:
                if line.startswith("nameserver"):
                    ip = line.split()[1].strip()
                    if ip and ip != "127.0.0.1":
                        return ip
    except OSError:
        pass
    return "127.0.0.1"


def _get_dtc_host() -> str:
    host = os.getenv("SC_DTC_HOST", "").strip()
    return host if host else _detect_windows_host()


# ---------------------------------------------------------------------------
# DTC wire helpers
# ---------------------------------------------------------------------------

def _pack_json(msg: Dict[str, Any]) -> bytes:
    """Encode one DTC JSON message: 4-byte LE size prefix + JSON bytes."""
    body = json.dumps(msg, separators=(",", ":")).encode()
    return struct.pack("<I", len(body)) + body


async def _read_message(reader: asyncio.StreamReader) -> Optional[Dict[str, Any]]:
    """Read one DTC JSON message. Returns None on clean EOF."""
    header = await reader.readexactly(4)
    size = struct.unpack("<I", header)[0]
    if size == 0:
        return {}
    body = await reader.readexactly(size)
    return json.loads(body)


# ---------------------------------------------------------------------------
# DOM book
# ---------------------------------------------------------------------------

class _DOMBook:
    """Maintains a live bid/ask book from DTC Level 2 messages."""

    def __init__(self, tick_size: float = 0.25) -> None:
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.tick_size           = tick_size
        self.last_price          = 0.0
        self._last_update_ms     = 0
        self._prev_bid_best_size = 0.0
        self._prev_ask_best_size = 0.0
        self._ofi_window: Deque[Tuple[int, float]] = deque()

    def clear(self) -> None:
        self.bids.clear()
        self.asks.clear()

    def apply(self, price: float, size: float, side: int, update_type: int) -> None:
        book = self.bids if side == _SIDE_BID else self.asks
        if update_type == _UPDATE_DELETE or size <= 0:
            book.pop(price, None)
        else:
            book[price] = size
        self._last_update_ms = int(time.time() * 1000)

    def _best_bid(self) -> Optional[float]:
        return max(self.bids) if self.bids else None

    def _best_ask(self) -> Optional[float]:
        return min(self.asks) if self.asks else None

    def _depth_n(
        self, book: Dict[float, float], best: float, ticks: int, ascending: bool
    ) -> float:
        limit = ticks * self.tick_size
        return sum(
            size for price, size in book.items()
            if 0 <= (price - best if ascending else best - price) <= limit
        )

    def _update_ofi(self) -> float:
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - 1_000
        while self._ofi_window and self._ofi_window[0][0] < cutoff:
            self._ofi_window.popleft()

        bb = self._best_bid()
        ba = self._best_ask()
        bid_sz = self.bids.get(bb, 0.0) if bb is not None else 0.0
        ask_sz = self.asks.get(ba, 0.0) if ba is not None else 0.0

        delta = (bid_sz - self._prev_bid_best_size) - (ask_sz - self._prev_ask_best_size)
        self._prev_bid_best_size = bid_sz
        self._prev_ask_best_size = ask_sz
        if delta != 0.0:
            self._ofi_window.append((now_ms, delta))
        return sum(d for _, d in self._ofi_window)

    def to_dom_payload(self, symbol: str) -> Optional[Dict[str, Any]]:
        bb = self._best_bid()
        ba = self._best_ask()
        if bb is None or ba is None:
            return None

        ofi       = self._update_ofi()
        bd5       = self._depth_n(self.bids, bb, 5, ascending=False)
        ad5       = self._depth_n(self.asks, ba, 5, ascending=True)
        imbalance = bd5 / max(bd5 + ad5, 1e-9)

        bids_out: List[Dict[str, float]] = [
            {"price": p, "size": s}
            for p, s in sorted(self.bids.items(), reverse=True)[:DOM_LEVELS]
        ]
        asks_out: List[Dict[str, float]] = [
            {"price": p, "size": s}
            for p, s in sorted(self.asks.items())[:DOM_LEVELS]
        ]

        return {
            "ts_ms":           self._last_update_ms,
            "symbol":          symbol,
            "price":           self.last_price or bb,
            "bids":            bids_out,
            "asks":            asks_out,
            "ofi_1s":          round(ofi, 2),
            "bid_depth_1":     round(self._depth_n(self.bids, bb,  1, False), 1),
            "bid_depth_5":     round(bd5,                                     1),
            "bid_depth_10":    round(self._depth_n(self.bids, bb, 10, False), 1),
            "bid_depth_20":    round(self._depth_n(self.bids, bb, 20, False), 1),
            "ask_depth_1":     round(self._depth_n(self.asks, ba,  1, True),  1),
            "ask_depth_5":     round(ad5,                                     1),
            "ask_depth_10":    round(self._depth_n(self.asks, ba, 10, True),  1),
            "ask_depth_20":    round(self._depth_n(self.asks, ba, 20, True),  1),
            "imbalance_ratio": round(imbalance, 4),
        }


# ---------------------------------------------------------------------------
# CVD accumulator
# ---------------------------------------------------------------------------

class _CVDAccumulator:
    """Rolling CVD over 1/5/15-minute windows plus running day total."""

    def __init__(self) -> None:
        self._trades: Deque[Tuple[int, float]] = deque()
        self._day_cvd        = 0.0
        self._last_update_ms = 0

    def add_trade(self, volume: float, at_bid_or_ask: int) -> None:
        now_ms = int(time.time() * 1000)
        signed = volume if at_bid_or_ask == _AT_ASK else -volume
        self._trades.append((now_ms, signed))
        self._day_cvd        += signed
        self._last_update_ms  = now_ms
        cutoff = now_ms - 900_000
        while self._trades and self._trades[0][0] < cutoff:
            self._trades.popleft()

    def _window_cvd(self, window_ms: int) -> float:
        cutoff = int(time.time() * 1000) - window_ms
        return sum(v for ts, v in self._trades if ts >= cutoff)

    def _window_vols(self, window_ms: int) -> Tuple[float, float]:
        cutoff = int(time.time() * 1000) - window_ms
        buy  = sum( v for ts, v in self._trades if ts >= cutoff and v > 0)
        sell = sum(-v for ts, v in self._trades if ts >= cutoff and v < 0)
        return buy, sell

    def to_cvd_payload(self, symbol: str) -> Dict[str, Any]:
        buy_1, sell_1 = self._window_vols(60_000)
        return {
            "ts_ms":           self._last_update_ms or int(time.time() * 1000),
            "symbol":          symbol,
            "cvd_1min":        round(self._window_cvd(60_000),  1),
            "cvd_5min":        round(self._window_cvd(300_000), 1),
            "cvd_15min":       round(self._window_cvd(900_000), 1),
            "cvd_running_day": round(self._day_cvd, 1),
            "buy_vol_1min":    round(buy_1,  1),
            "sell_vol_1min":   round(sell_1, 1),
        }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class SierraDOMBridgeService:
    """Asyncio service: DTC TCP client <-> Redis bridge for SC DOM/CVD data."""

    def __init__(self, redis_client: Optional[aioredis.Redis] = None) -> None:
        self._redis        = redis_client
        self._symbol       = DOM_SYMBOL
        self._dtc_host     = _get_dtc_host()
        self._dtc_port     = SC_DTC_PORT
        self._dtc_symbol   = SC_DTC_SYMBOL
        self._danger_path  = _resolve_danger_path()
        self._book         = _DOMBook(tick_size=MNQ_TICK_SIZE)
        self._cvd          = _CVDAccumulator()
        self._running      = False
        self._snap_done    = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._redis is None:
            self._redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
        self._running = True
        LOGGER.info(
            "SierraDOMBridgeService starting -- DTC %s:%d  symbol=%s  publish every %d ms",
            self._dtc_host, self._dtc_port, self._dtc_symbol, POLL_INTERVAL_MS,
        )
        await asyncio.gather(
            self._dtc_loop(),
            self._danger_subscriber(),
        )

    async def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # DTC reconnect wrapper
    # ------------------------------------------------------------------

    async def _dtc_loop(self) -> None:
        backoff = RECONNECT_BASE_S
        while self._running:
            try:
                await self._run_session()
                backoff = RECONNECT_BASE_S
            except (ConnectionRefusedError, OSError) as exc:
                LOGGER.warning("DTC connection failed (%s) -- retry in %.0fs", exc, backoff)
            except asyncio.CancelledError:
                break
            except Exception:
                LOGGER.exception("Unexpected DTC error -- retry in %.0fs", backoff)
            if self._running:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_MAX_S)

    # ------------------------------------------------------------------
    # Single DTC session
    # ------------------------------------------------------------------

    async def _run_session(self) -> None:
        LOGGER.info("DTC connecting to %s:%d", self._dtc_host, self._dtc_port)
        reader, writer = await asyncio.open_connection(self._dtc_host, self._dtc_port)
        self._snap_done = False
        self._book.clear()

        try:
            # 1. Encoding negotiation (binary, always first)
            writer.write(_ENCODING_REQUEST_BYTES)
            await writer.drain()
            enc_resp = await asyncio.wait_for(reader.readexactly(16), timeout=5.0)
            enc_type     = struct.unpack_from("<H", enc_resp, 2)[0]
            enc_encoding = struct.unpack_from("<i", enc_resp, 8)[0]
            if enc_type != _T_ENCODING_RESPONSE or enc_encoding != _JSON_COMPACT_ENCODING:
                raise ValueError(
                    f"Unexpected encoding response: type={enc_type} enc={enc_encoding}"
                )
            LOGGER.debug("DTC encoding negotiated: JSON Compact")

            # 2. Logon
            writer.write(_pack_json({
                "Type":                       _T_LOGON_REQUEST,
                "ProtocolVersion":            _DTC_PROTOCOL_VERSION,
                "Username":                   "",
                "Password":                   "",
                "HeartbeatIntervalInSeconds": HEARTBEAT_S,
                "ClientName":                 "SweepRunner",
            }))

            # 3. Market depth subscription
            writer.write(_pack_json({
                "Type":          _T_MARKET_DEPTH_REQUEST,
                "RequestAction": 1,
                "SymbolID":      _SYMBOL_ID,
                "Symbol":        self._dtc_symbol,
                "Exchange":      "",
                "NumLevels":     DOM_LEVELS,
            }))

            # 4. Trade data subscription (CVD)
            writer.write(_pack_json({
                "Type":          _T_MARKET_DATA_REQUEST,
                "RequestAction": 1,
                "SymbolID":      _SYMBOL_ID,
                "Symbol":        self._dtc_symbol,
                "Exchange":      "",
            }))
            await writer.drain()

            LOGGER.info("DTC subscribed: depth + trades for %s", self._dtc_symbol)

            await asyncio.gather(
                self._message_loop(reader, writer),
                self._publish_loop(),
            )
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Message loop
    # ------------------------------------------------------------------

    async def _message_loop(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        while self._running:
            try:
                msg = await asyncio.wait_for(
                    _read_message(reader), timeout=HEARTBEAT_S * 2
                )
            except asyncio.TimeoutError:
                LOGGER.warning("DTC heartbeat timeout -- reconnecting")
                raise OSError("DTC heartbeat timeout")

            if msg is None:
                raise OSError("DTC connection closed by server")

            msg_type = msg.get("Type", 0)

            if msg_type == _T_HEARTBEAT:
                writer.write(_pack_json({"Type": _T_HEARTBEAT}))
                await writer.drain()

            elif msg_type == _T_LOGON_RESPONSE:
                if msg.get("Result", 0) != 1:
                    raise ValueError(f"DTC logon rejected: {msg.get('ResultText', '')}")
                LOGGER.info("DTC logon OK  server=%s", msg.get("ServerName", "SC"))

            elif msg_type == _T_MARKET_DATA_SNAPSHOT:
                price = float(msg.get("LastTradePrice", 0))
                if price > 0:
                    self._book.last_price = price

            elif msg_type == _T_MARKET_DEPTH_SNAPSHOT_L2:
                self._book.apply(
                    price=float(msg.get("Price", 0)),
                    size=float(msg.get("Quantity", 0)),
                    side=int(msg.get("Side", _SIDE_BID)),
                    update_type=_UPDATE_ADD,
                )
                if msg.get("IsLastMessageInBatch", 0):
                    self._snap_done = True
                    LOGGER.info(
                        "DTC DOM snapshot complete -- %d bids  %d asks",
                        len(self._book.bids), len(self._book.asks),
                    )

            elif msg_type == _T_MARKET_DEPTH_UPDATE_L2:
                if self._snap_done:
                    self._book.apply(
                        price=float(msg.get("Price", 0)),
                        size=float(msg.get("Quantity", 0)),
                        side=int(msg.get("Side", _SIDE_BID)),
                        update_type=int(msg.get("UpdateType", _UPDATE_ADD)),
                    )

            elif msg_type in (_T_MARKET_DATA_UPDATE_TRADE,
                               _T_MARKET_DATA_UPDATE_TRADE_COMPACT):
                price  = float(msg.get("Price", 0))
                volume = float(msg.get("Volume", msg.get("Quantity", 0)))
                side   = int(msg.get("AtBidOrAsk", _AT_ASK))
                if price > 0:
                    self._book.last_price = price
                    self._cvd.add_trade(volume, side)

            elif msg_type in (_T_MARKET_DATA_REJECT, _T_MARKET_DEPTH_REJECT):
                LOGGER.error("DTC request rejected: %s", msg.get("RejectText", msg))

    # ------------------------------------------------------------------
    # Publish loop
    # ------------------------------------------------------------------

    async def _publish_loop(self) -> None:
        interval = POLL_INTERVAL_MS / 1000.0
        while self._running:
            await asyncio.sleep(interval)
            if not self._snap_done:
                continue
            now_ms = int(time.time() * 1000)
            await asyncio.gather(
                self._publish_dom(now_ms),
                self._publish_cvd(),
            )

    async def _publish_dom(self, now_ms: int) -> None:
        payload = self._book.to_dom_payload(self._symbol)
        if payload is None:
            return
        if (now_ms - payload["ts_ms"]) > STALE_THRESHOLD_MS:
            return
        try:
            await self._redis.publish(
                f"market:dom:{self._symbol}", json.dumps(payload)
            )
        except Exception:
            LOGGER.debug("Redis DOM publish failed", exc_info=True)

    async def _publish_cvd(self) -> None:
        payload = self._cvd.to_cvd_payload(self._symbol)
        try:
            await self._redis.publish(
                f"market:cvd:{self._symbol}", json.dumps(payload)
            )
        except Exception:
            LOGGER.debug("Redis CVD publish failed", exc_info=True)

    # ------------------------------------------------------------------
    # Danger trigger write-back
    # ------------------------------------------------------------------

    async def _danger_subscriber(self) -> None:
        sub = self._redis.pubsub()
        await sub.subscribe(f"sweep:danger:{self._symbol}")
        LOGGER.info("SierraDOMBridgeService: subscribed to sweep:danger:%s", self._symbol)
        async for message in sub.listen():
            if not self._running:
                break
            if message["type"] != "message":
                continue
            try:
                data: Dict[str, Any] = json.loads(message["data"])
            except (json.JSONDecodeError, TypeError):
                continue
            await self._write_danger_trigger(data)

    async def _write_danger_trigger(self, data: Dict[str, Any]) -> None:
        payload = {
            "ts_ms":    int(time.time() * 1000),
            "action":   data.get("action", "flatten"),
            "reason":   data.get("reason", ""),
            "severity": data.get("severity", "critical"),
            "consumed": False,
        }
        try:
            tmp = self._danger_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(self._danger_path)
            LOGGER.warning(
                "DOMBridge: wrote danger_trigger.json  severity=%s  reason=%s",
                payload["severity"], payload["reason"],
            )
        except Exception:
            LOGGER.exception("Failed to write danger_trigger.json")


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------

async def _main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s -- %(message)s",
    )
    svc = SierraDOMBridgeService()
    await svc.start()


if __name__ == "__main__":
    asyncio.run(_main())
