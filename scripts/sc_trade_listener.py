"""sc_trade_listener.py -- Sierra Chart trade executor for the trading computer.

Runs on the Windows trading computer (or any host with LAN access to Redis AND
Sierra Chart's DTC server).  Subscribes to sc:trade:signal, executes orders
via Sierra Chart's DTC Protocol Server (localhost), then publishes fill
acknowledgements to sc:trade:ack so the Discord bot can DM the trader.

Prerequisites
─────────────
Sierra Chart → Global Settings → DTC Protocol Server:
    Enable DTC Protocol Server : Yes
    Listening Port             : 11099  (or set SC_DTC_PORT)
    Require Authentication     : No
    Encoding (List)            : JSON Compact
    Allow Trading              : Yes        ← required for order submission

Environment variables
─────────────────────
    REDIS_HOST              Redis server IP reachable from this machine
                            Default: 192.168.1.151
    REDIS_PORT              Default: 6379
    REDIS_PASSWORD          Optional
    REDIS_DB                Default: 0
    SC_DTC_HOST             Sierra Chart DTC host (default: 127.0.0.1)
    SC_DTC_PORT             DTC port (default: 11099)
    SC_DTC_SYMBOL           Full DTC symbol (default: MNQM26_FUT_CME)
    SC_DOM_SYMBOL           Short name for logging (default: MNQ)
    SC_TRADE_ACCOUNT        Trade account name in SC (default: Sim1)
    MNQ_TICK_VALUE          Price units per tick (default: 0.25)
    SC_TRADE_DRY_RUN        Override dry-run for all signals (default: respect signal)
    SC_TRADE_SIGNAL_CHANNEL Redis channel to subscribe (default: sc:trade:signal)
    SC_TRADE_ACK_CHANNEL    Redis channel to publish acks (default: sc:trade:ack)

Usage
─────
    python scripts/sc_trade_listener.py
    python scripts/sc_trade_listener.py --dry-run   # force dry-run regardless of signal

Flow for buy/sell signal
────────────────────────
    1. Connect to SC DTC (JSON compact over TCP)
    2. Logon + subscribe to order updates
    3. SubmitNewSingleOrder  (Type 208, Market)
    4. Watch OrderUpdate     (Type 301) for OrderStatus == 8 (Filled)
    5. Calculate TP price: fill ± (tp_ticks × tick_size)
    6. SubmitNewSingleOrder  (Type 208, Limit, opposite side)
    7. Publish ack to sc:trade:ack

Flow for flatten signal
───────────────────────
    1. Cancel all locally tracked open TP orders (CancelOrder Type 203)
    2. Submit opposite-side market order for abs(tracked_position_qty)
    3. Reset tracked position
    4. Publish ack

DTC message type constants used
────────────────────────────────
    1   LogonRequest
    2   LogonResponse
    3   Heartbeat
    6   EncodingRequest
    208 SubmitNewSingleOrder
    203 CancelOrder
    301 OrderUpdate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

import redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sc_trade_listener")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REDIS_HOST    = os.getenv("REDIS_HOST",              "192.168.168.151")
REDIS_PORT    = int(os.getenv("REDIS_PORT",          "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_DB      = int(os.getenv("REDIS_DB",            "0"))

SC_DTC_HOST   = os.getenv("SC_DTC_HOST",             "127.0.0.1")
SC_DTC_PORT   = int(os.getenv("SC_DTC_PORT",         "11099"))
SC_DTC_SYMBOL = os.getenv("SC_DTC_SYMBOL",           "MNQM26_FUT_CME")
SC_DISPLAY    = os.getenv("SC_DOM_SYMBOL",            "MNQ")
SC_ACCOUNT    = os.getenv("SC_TRADE_ACCOUNT",         "Sim1")
TICK_SIZE     = float(os.getenv("MNQ_TICK_VALUE",    "0.25"))

SIGNAL_CHANNEL = os.getenv("SC_TRADE_SIGNAL_CHANNEL", "sc:trade:signal")
ACK_CHANNEL    = os.getenv("SC_TRADE_ACK_CHANNEL",    "sc:trade:ack")

# Force dry-run for all signals when set via env
ENV_DRY_RUN   = os.getenv("SC_TRADE_DRY_RUN", "").lower() in ("1", "true", "yes")

# DTC protocol constants
_DTC_PROTOCOL_VERSION     = 8
_JSON_ENCODING            = 2

_T_LOGON_REQUEST          = 1
_T_LOGON_RESPONSE         = 2
_T_HEARTBEAT              = 3
_T_ENCODING_REQUEST       = 6
_T_SUBMIT_NEW_SINGLE_ORDER = 208
_T_CANCEL_ORDER           = 203
_T_ORDER_UPDATE           = 301

_ORDER_STATUS_OPEN        = 4
_ORDER_STATUS_CANCELED    = 7
_ORDER_STATUS_FILLED      = 8
_ORDER_STATUS_PARTIAL     = 9

_ORDER_TYPE_MARKET        = 1
_ORDER_TYPE_LIMIT         = 2
_TIF_DAY                  = 1
_TIF_GTC                  = 2

_BUY                      = 1
_SELL                     = 2

HEARTBEAT_S               = 10
FILL_TIMEOUT_S            = 15   # seconds to wait for entry fill before aborting
RECONNECT_BASE_S          = 2.0
RECONNECT_MAX_S           = 60.0

# 16-byte binary ENCODING_REQUEST (must be sent as raw binary before JSON negotiation)
_ENCODING_REQUEST_BYTES = struct.pack(
    "<HHii4s",
    16,
    _T_ENCODING_REQUEST,
    _DTC_PROTOCOL_VERSION,
    _JSON_ENCODING,
    b"DTC\x00",
)


# ---------------------------------------------------------------------------
# DTC wire helpers
# ---------------------------------------------------------------------------

def _pack_json(msg: Dict[str, Any]) -> bytes:
    return json.dumps(msg, separators=(",", ":")).encode() + b"\x00"


async def _read_dtc(reader: asyncio.StreamReader) -> Optional[Dict[str, Any]]:
    try:
        raw = await reader.readuntil(b"\x00")
        return json.loads(raw.rstrip(b"\x00"))
    except asyncio.IncompleteReadError:
        return None


# ---------------------------------------------------------------------------
# Order state tracker
# ---------------------------------------------------------------------------

class _OrderTracker:
    """Minimal in-memory position and open-TP tracker."""

    def __init__(self) -> None:
        # client_order_id → {server_order_id, qty, side, signal}
        self._pending_entry: Dict[str, Dict[str, Any]] = {}
        # tp_client_id → {server_order_id, entry_client_id, qty, side, tp_price}
        self._open_tps: Dict[str, Dict[str, Any]] = {}
        # server_order_id → client_order_id (reverse lookup)
        self._server_to_client: Dict[str, str] = {}
        # net tracked position: positive = long, negative = short
        self.position_qty: int = 0

    def track_entry(self, client_id: str, qty: int, side: int, signal: Dict) -> None:
        self._pending_entry[client_id] = {
            "server_order_id": "",
            "qty": qty,
            "side": side,
            "signal": signal,
        }

    def track_tp(self, tp_client_id: str, entry_client_id: str,
                 qty: int, side: int, tp_price: float) -> None:
        self._open_tps[tp_client_id] = {
            "server_order_id": "",
            "entry_client_id": entry_client_id,
            "qty": qty,
            "side": side,
            "tp_price": tp_price,
        }

    def update_server_id(self, client_id: str, server_id: str) -> None:
        if client_id in self._pending_entry:
            self._pending_entry[client_id]["server_order_id"] = server_id
        elif client_id in self._open_tps:
            self._open_tps[client_id]["server_order_id"] = server_id
        if server_id:
            self._server_to_client[server_id] = client_id

    def on_entry_filled(self, client_id: str) -> Optional[Dict]:
        rec = self._pending_entry.pop(client_id, None)
        if rec:
            qty = rec["qty"]
            if rec["side"] == _BUY:
                self.position_qty += qty
            else:
                self.position_qty -= qty
        return rec

    def on_tp_filled(self, tp_client_id: str) -> Optional[Dict]:
        rec = self._open_tps.pop(tp_client_id, None)
        if rec:
            qty = rec["qty"]
            if rec["side"] == _BUY:   # TP for a buy entry is a sell
                self.position_qty += qty
            else:
                self.position_qty -= qty
        return rec

    def all_open_tp_server_ids(self) -> List[str]:
        return [r["server_order_id"] for r in self._open_tps.values() if r["server_order_id"]]

    def clear_tps(self) -> None:
        self._open_tps.clear()

    def resolve_client_id(self, server_id: str) -> Optional[str]:
        return self._server_to_client.get(server_id)


# ---------------------------------------------------------------------------
# DTC session
# ---------------------------------------------------------------------------

class _DTCSession:
    """Single DTC TCP session: logon + order dispatch + order update routing."""

    def __init__(self, tracker: _OrderTracker) -> None:
        self._tracker = tracker
        self._writer: Optional[asyncio.StreamWriter] = None
        self._logon_ok = asyncio.Event()
        # fill_event keyed by client_order_id → (event, fill_price slot)
        self._fill_events: Dict[str, Tuple[asyncio.Event, List[float]]] = {}

    async def connect(self) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        log.info("DTC connecting → %s:%d", SC_DTC_HOST, SC_DTC_PORT)
        reader, writer = await asyncio.open_connection(SC_DTC_HOST, SC_DTC_PORT)
        self._writer = writer

        # Binary encoding negotiation — must be first bytes, no response expected
        writer.write(_ENCODING_REQUEST_BYTES)
        await writer.drain()

        writer.write(_pack_json({
            "Type":                       _T_LOGON_REQUEST,
            "ProtocolVersion":            _DTC_PROTOCOL_VERSION,
            "Username":                   "",
            "Password":                   "",
            "HeartbeatIntervalInSeconds": HEARTBEAT_S,
            "ClientName":                 "SCTradeListener",
        }))
        await writer.drain()
        return reader, writer

    async def send(self, msg: Dict[str, Any]) -> None:
        if self._writer:
            self._writer.write(_pack_json(msg))
            await self._writer.drain()

    async def run_message_loop(self, reader: asyncio.StreamReader) -> None:
        while True:
            try:
                msg = await asyncio.wait_for(_read_dtc(reader), timeout=HEARTBEAT_S * 2)
            except asyncio.TimeoutError:
                raise OSError("DTC heartbeat timeout")
            if msg is None:
                raise OSError("DTC connection closed by server")

            t = msg.get("Type", 0)

            if t == _T_HEARTBEAT:
                await self.send({"Type": _T_HEARTBEAT})

            elif t == _T_LOGON_RESPONSE:
                result = msg.get("Result", 1)
                if result not in (0, 1):
                    raise ValueError(f"DTC logon rejected: {msg.get('ResultText')}")
                log.info("DTC logon OK  server=%s", msg.get("ServerName", "SC"))
                self._logon_ok.set()

            elif t == _T_ORDER_UPDATE:
                await self._handle_order_update(msg)

    async def _handle_order_update(self, msg: Dict[str, Any]) -> None:
        client_id  = msg.get("ClientOrderID", "")
        server_id  = msg.get("ServerOrderID", "")
        status     = msg.get("OrderStatus", 0)
        fill_price = float(msg.get("AverageFillPrice", 0.0))

        if server_id:
            self._tracker.update_server_id(client_id, server_id)

        if status == _ORDER_STATUS_FILLED:
            # Entry fill
            if client_id in self._fill_events:
                ev, slot = self._fill_events[client_id]
                slot[0] = fill_price
                ev.set()
            # TP fill
            elif client_id.startswith("tp_"):
                rec = self._tracker.on_tp_filled(client_id)
                if rec:
                    log.info(
                        "TP filled  client=%s  qty=%s  price=%.2f",
                        client_id, rec["qty"], fill_price,
                    )
        elif status == _ORDER_STATUS_CANCELED:
            log.info("Order cancelled  client=%s", client_id)

    async def wait_for_logon(self) -> None:
        await asyncio.wait_for(self._logon_ok.wait(), timeout=10.0)

    async def submit_market_order(
        self,
        client_id: str,
        side: int,
        qty: int,
    ) -> None:
        await self.send({
            "Type":           _T_SUBMIT_NEW_SINGLE_ORDER,
            "Symbol":         SC_DTC_SYMBOL,
            "Exchange":       "",
            "TradeAccount":   SC_ACCOUNT,
            "ClientOrderID":  client_id,
            "OrderType":      _ORDER_TYPE_MARKET,
            "BuySell":        side,
            "Quantity":       float(qty),
            "TimeInForce":    _TIF_DAY,
            "IsAutomatedOrder": 1,
        })

    async def submit_limit_order(
        self,
        client_id: str,
        side: int,
        qty: int,
        price: float,
    ) -> None:
        await self.send({
            "Type":           _T_SUBMIT_NEW_SINGLE_ORDER,
            "Symbol":         SC_DTC_SYMBOL,
            "Exchange":       "",
            "TradeAccount":   SC_ACCOUNT,
            "ClientOrderID":  client_id,
            "OrderType":      _ORDER_TYPE_LIMIT,
            "BuySell":        side,
            "Price1":         price,
            "Quantity":       float(qty),
            "TimeInForce":    _TIF_GTC,
            "IsAutomatedOrder": 1,
        })

    async def cancel_order(self, server_id: str, client_cancel_id: str) -> None:
        await self.send({
            "Type":           _T_CANCEL_ORDER,
            "ServerOrderID":  server_id,
            "ClientOrderID":  client_cancel_id,
        })

    async def wait_for_fill(self, client_id: str, timeout: float = FILL_TIMEOUT_S) -> float:
        ev   = asyncio.Event()
        slot = [0.0]
        self._fill_events[client_id] = (ev, slot)
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"No fill received for {client_id} within {timeout}s")
        finally:
            self._fill_events.pop(client_id, None)
        return slot[0]


# ---------------------------------------------------------------------------
# Signal handlers
# ---------------------------------------------------------------------------

async def handle_entry(
    signal: Dict[str, Any],
    session: _DTCSession,
    tracker: _OrderTracker,
    redis_client: redis.Redis,
    force_dry: bool,
) -> None:
    action   = signal["action"]
    qty      = int(signal["quantity"])
    tp_ticks = int(signal["tp_ticks"])
    tick_sz  = float(signal.get("tick_size", TICK_SIZE))
    dry_run  = force_dry or bool(signal.get("dry_run", True))
    req_id   = signal["request_id"]
    user_id  = signal.get("user_id", "")
    side     = _BUY if action == "buy" else _SELL
    ts_ms    = int(time.time() * 1000)

    if dry_run:
        tp_pts   = tp_ticks * tick_sz
        msg_text = (
            f"{'BUY' if side == _BUY else 'SELL'} {qty} {SC_DISPLAY} @ market | "
            f"TP {tp_ticks}t ({tp_pts:.2f} pts)"
        )
        log.info("DRY RUN — would submit: %s", msg_text)
        _publish_ack(redis_client, {
            "request_id": req_id,
            "user_id":    user_id,
            "status":     "dry_run",
            "dry_run":    True,
            "fill_price": 0.0,
            "tp_price":   0.0,
            "message":    msg_text,
            "ts_ms":      ts_ms,
        })
        return

    client_id = f"entry_{req_id}"
    tracker.track_entry(client_id, qty, side, signal)

    log.info(
        "Submitting %s %d %s @ market  client_id=%s",
        action.upper(), qty, SC_DISPLAY, client_id,
    )
    await session.submit_market_order(client_id, side, qty)

    try:
        fill_price = await session.wait_for_fill(client_id)
    except TimeoutError as exc:
        log.error("Entry fill timeout: %s", exc)
        _publish_ack(redis_client, {
            "request_id": req_id,
            "user_id":    user_id,
            "status":     "error",
            "message":    f"Fill timeout: {exc}",
            "ts_ms":      int(time.time() * 1000),
        })
        return

    tracker.on_entry_filled(client_id)

    # Calculate and submit TP
    tp_price = (
        fill_price + tp_ticks * tick_sz
        if side == _BUY
        else fill_price - tp_ticks * tick_sz
    )
    tp_side      = _SELL if side == _BUY else _BUY
    tp_client_id = f"tp_{req_id}"

    tracker.track_tp(tp_client_id, client_id, qty, tp_side, tp_price)
    log.info("Submitting TP limit @ %.2f  client_id=%s", tp_price, tp_client_id)
    await session.submit_limit_order(tp_client_id, tp_side, qty, tp_price)

    msg_text = (
        f"{'BUY' if side == _BUY else 'SELL'} {qty} {SC_DISPLAY} "
        f"filled @ {fill_price:.2f} | TP @ {tp_price:.2f} ({tp_ticks}t)"
    )
    log.info(msg_text)
    _publish_ack(redis_client, {
        "request_id": req_id,
        "user_id":    user_id,
        "status":     "filled",
        "fill_price": fill_price,
        "tp_price":   tp_price,
        "message":    msg_text,
        "ts_ms":      int(time.time() * 1000),
    })


async def handle_flatten(
    signal: Dict[str, Any],
    session: _DTCSession,
    tracker: _OrderTracker,
    redis_client: redis.Redis,
    force_dry: bool,
) -> None:
    req_id  = signal["request_id"]
    user_id = signal.get("user_id", "")
    dry_run = force_dry or bool(signal.get("dry_run", True))
    pos_qty = tracker.position_qty
    ts_ms   = int(time.time() * 1000)

    if dry_run:
        log.info("DRY RUN — would flatten position_qty=%d", pos_qty)
        _publish_ack(redis_client, {
            "request_id": req_id,
            "user_id":    user_id,
            "status":     "dry_run",
            "dry_run":    True,
            "message":    f"Flatten (dry run) — tracked position: {pos_qty}",
            "ts_ms":      ts_ms,
        })
        return

    # Cancel all open TPs
    tp_server_ids = tracker.all_open_tp_server_ids()
    for i, server_id in enumerate(tp_server_ids):
        cancel_id = f"cancel_{req_id}_{i}"
        log.info("Cancelling TP server_id=%s", server_id)
        await session.cancel_order(server_id, cancel_id)
    tracker.clear_tps()

    if pos_qty == 0:
        msg_text = f"No tracked position to flatten. Cancelled {len(tp_server_ids)} TP order(s)."
        log.info(msg_text)
        _publish_ack(redis_client, {
            "request_id": req_id,
            "user_id":    user_id,
            "status":     "cancelled",
            "message":    msg_text,
            "ts_ms":      int(time.time() * 1000),
        })
        return

    # Submit market close
    close_side   = _SELL if pos_qty > 0 else _BUY
    close_qty    = abs(pos_qty)
    close_id     = f"flat_{req_id}"
    log.info("Submitting market close  side=%s qty=%d", "SELL" if close_side == _SELL else "BUY", close_qty)
    await session.submit_market_order(close_id, close_side, close_qty)

    try:
        fill_price = await session.wait_for_fill(close_id)
    except TimeoutError as exc:
        log.error("Flatten fill timeout: %s", exc)
        _publish_ack(redis_client, {
            "request_id": req_id,
            "user_id":    user_id,
            "status":     "error",
            "message":    f"Flatten fill timeout: {exc}",
            "ts_ms":      int(time.time() * 1000),
        })
        return

    tracker.position_qty = 0
    msg_text = (
        f"Flattened {close_qty} {SC_DISPLAY} @ {fill_price:.2f} | "
        f"Cancelled {len(tp_server_ids)} TP order(s)."
    )
    log.info(msg_text)
    _publish_ack(redis_client, {
        "request_id": req_id,
        "user_id":    user_id,
        "status":     "filled",
        "fill_price": fill_price,
        "message":    msg_text,
        "ts_ms":      int(time.time() * 1000),
    })


def _publish_ack(redis_client: redis.Redis, payload: Dict[str, Any]) -> None:
    try:
        redis_client.publish(ACK_CHANNEL, json.dumps(payload))
        log.debug("Ack published → %s  status=%s", ACK_CHANNEL, payload.get("status"))
    except Exception as exc:
        log.error("Failed to publish ack: %s", exc)


# ---------------------------------------------------------------------------
# Main service loop
# ---------------------------------------------------------------------------

async def _run_listener(force_dry: bool) -> None:
    r_sync = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        db=REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=10,
        socket_keepalive=True,
    )
    log.info(
        "Redis connected  %s:%d  signal=%s  ack=%s",
        REDIS_HOST, REDIS_PORT, SIGNAL_CHANNEL, ACK_CHANNEL,
    )

    tracker = _OrderTracker()
    session = _DTCSession(tracker)

    # Connect to SC DTC and run message loop in background
    reader, writer = await session.connect()
    msg_task = asyncio.create_task(session.run_message_loop(reader))
    await session.wait_for_logon()
    log.info("SC DTC ready  symbol=%s  account=%s%s", SC_DTC_SYMBOL, SC_ACCOUNT,
             "  [FORCE DRY-RUN]" if force_dry else "")

    pubsub = r_sync.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(SIGNAL_CHANNEL)
    log.info("Subscribed to %s — waiting for signals …", SIGNAL_CHANNEL)

    try:
        while True:
            raw = await asyncio.to_thread(pubsub.get_message, timeout=0.5)
            if not raw:
                await asyncio.sleep(0.05)
                continue

            try:
                signal = json.loads(raw["data"])
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

            action = signal.get("action", "")
            log.info(
                "Signal received  action=%s  qty=%s  tp=%s  dry=%s  req=%s",
                action,
                signal.get("quantity"),
                signal.get("tp_ticks"),
                signal.get("dry_run"),
                signal.get("request_id"),
            )

            try:
                if action in ("buy", "sell"):
                    await handle_entry(signal, session, tracker, r_sync, force_dry)
                elif action == "flatten":
                    await handle_flatten(signal, session, tracker, r_sync, force_dry)
                else:
                    log.warning("Unknown action: %s", action)
            except Exception:
                log.exception("Error handling signal %s", signal.get("request_id"))
                _publish_ack(r_sync, {
                    "request_id": signal.get("request_id", ""),
                    "user_id":    signal.get("user_id", ""),
                    "status":     "error",
                    "message":    "Unhandled error — check sc_trade_listener logs",
                    "ts_ms":      int(time.time() * 1000),
                })

    finally:
        msg_task.cancel()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        pubsub.unsubscribe()
        r_sync.close()


async def _main(force_dry: bool) -> None:
    backoff = RECONNECT_BASE_S
    while True:
        try:
            await _run_listener(force_dry)
            backoff = RECONNECT_BASE_S
        except (ConnectionRefusedError, OSError) as exc:
            log.warning("Connection failed (%s) — retry in %.0fs", exc, backoff)
        except asyncio.CancelledError:
            log.info("Listener cancelled, exiting.")
            break
        except Exception:
            log.exception("Unexpected error — retry in %.0fs", backoff)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, RECONNECT_MAX_S)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sierra Chart trade listener")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run for all signals (signals publish/log but no SC orders submitted)",
    )
    args = parser.parse_args()
    force_dry = args.dry_run or ENV_DRY_RUN

    if force_dry:
        log.warning("*** FORCE DRY-RUN MODE — no orders will be submitted to Sierra Chart ***")
    else:
        log.warning("*** LIVE MODE — orders will be submitted to Sierra Chart ***")

    try:
        asyncio.run(_main(force_dry))
    except KeyboardInterrupt:
        log.info("Stopped.")
