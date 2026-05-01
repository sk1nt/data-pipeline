"""sierra_dom_bridge_service.py — bridges SC JSON files to Redis.

Reads dom_snapshot.json and trade_flow.json written by the ACSIL study
(data/dom_trade_bridge.cpp) every ~100 ms, validates freshness, and
publishes parsed payloads to Redis channels.

Also subscribes to sweep:danger:{symbol} and writes danger_trigger.json
back to the SC data directory so the ACSIL study can execute a flatten.

Redis channels published
────────────────────────
    market:dom:{symbol}
        Payload (from dom_snapshot.json):
            ts_ms           int      — Unix epoch milliseconds (UTC)
            symbol          str      — "MNQ" / "MES"
            price           float    — last trade price
            bids            list     — [{price, size}, …] nearest first, up to DOM_DEPTH levels
            asks            list     — [{price, size}, …] nearest first
            ofi_1s          float    — Order Flow Imbalance: (Δbid1 - Δask1) last 1 s
            bid_depth_1/5/10/20  float  — cumulative bid size at 1/5/10/20 ticks deep
            ask_depth_1/5/10/20  float  — cumulative ask size
            imbalance_ratio float    — bid_depth_5 / (bid_depth_5 + ask_depth_5)

    market:cvd:{symbol}
        Payload (from trade_flow.json):
            ts_ms           int      — Unix epoch milliseconds (UTC)
            symbol          str
            cvd_1min        float    — net CVD over last 60 s of completed bars
            cvd_5min        float    — net CVD over last 300 s
            cvd_15min       float    — net CVD over last 900 s
            cvd_running_day float    — cumulative day CVD since market open
            buy_vol_1min    float    — ask-side volume over last 60 s
            sell_vol_1min   float    — bid-side volume over last 60 s

Redis channel consumed
──────────────────────
    sweep:danger:{symbol}
        Payload: {"action": "flatten", "reason": "...", "severity": "critical"}
        This service writes that payload to danger_trigger.json, which the
        ACSIL study polls every 100 ms.  When it sees consumed==false it calls
        sc.FlattenAndCancelAllOrders() and sets consumed=true.

File → WSL path mapping
───────────────────────
    Windows path:  C:\\SierraChart\\Data\\dom_snapshot.json
    WSL path:      /mnt/c/SierraChart/Data/dom_snapshot.json
    Env var:       SC_DATA_DIR=/mnt/c/SierraChart/Data
    (Override individual files with SC_DOM_SNAPSHOT_PATH, SC_TRADE_FLOW_PATH,
     SC_DANGER_TRIGGER_PATH if SC writes to a non-default directory.)

Staleness guard
───────────────
    Payloads older than SC_BRIDGE_STALE_MS (default 2000 ms) are silently
    dropped.  This prevents stale DOM from triggering a flatten after a
    network hiccup.  If Sierra Chart is minimised or the chart is not
    receiving ticks, dom_snapshot.json stops updating and this guard fires.

Prerequisites on trading machine
─────────────────────────────────
    1. dom_trade_bridge.cpp compiled and loaded in Sierra Chart on the
       MNQ/MES chart (see data/dom_trade_bridge.cpp header for build steps).
    2. SC "Output Directory" input set to C:\\SierraChart\\Data\\ (the default).
    3. Redis running (WSL: sudo service redis-server start).
    4. SC_DATA_DIR in .env points to the WSL mount of that directory.
    5. sweep_runner.py starts this service via SweepRunner.start().
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import redis.asyncio as aioredis

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (all env-overridable)
# ---------------------------------------------------------------------------
SC_DATA_DIR        = os.getenv("SC_DATA_DIR",        "/mnt/c/SierraChart/Data")
SC_DOM_SNAPSHOT    = os.getenv("SC_DOM_SNAPSHOT_PATH", "")
SC_TRADE_FLOW      = os.getenv("SC_TRADE_FLOW_PATH",   "")
SC_DANGER_TRIGGER  = os.getenv("SC_DANGER_TRIGGER_PATH", "")

POLL_INTERVAL_MS   = int(os.getenv("SC_BRIDGE_POLL_MS", "100"))
STALE_THRESHOLD_MS = int(os.getenv("SC_BRIDGE_STALE_MS", "2000"))  # skip if > 2s old
DOM_SYMBOL         = os.getenv("SC_DOM_SYMBOL", "MNQ")
REDIS_URL          = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def _resolve_path(env_val: str, filename: str) -> Path:
    if env_val:
        return Path(env_val)
    return Path(SC_DATA_DIR) / filename


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class SierraDOMBridgeService:
    """Asyncio service: file-watcher ↔ Redis bridge for SC DOM data."""

    def __init__(self, redis_client: Optional[aioredis.Redis] = None) -> None:
        self._redis = redis_client
        self._dom_path     = _resolve_path(SC_DOM_SNAPSHOT, "dom_snapshot.json")
        self._flow_path    = _resolve_path(SC_TRADE_FLOW,   "trade_flow.json")
        self._danger_path  = _resolve_path(SC_DANGER_TRIGGER, "danger_trigger.json")
        self._symbol       = DOM_SYMBOL
        self._last_dom_ts  = 0
        self._last_flow_ts = 0
        self._running      = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._redis is None:
            self._redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
        self._running = True
        LOGGER.info(
            "SierraDOMBridgeService starting — polling %s every %d ms",
            SC_DATA_DIR, POLL_INTERVAL_MS,
        )
        await asyncio.gather(
            self._poll_loop(),
            self._danger_subscriber(),
        )

    async def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # File → Redis
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        interval = POLL_INTERVAL_MS / 1000.0
        while self._running:
            now_ms = int(time.time() * 1000)
            await asyncio.gather(
                self._maybe_publish_dom(now_ms),
                self._maybe_publish_flow(now_ms),
            )
            await asyncio.sleep(interval)

    async def _maybe_publish_dom(self, now_ms: int) -> None:
        payload = _read_json_file(self._dom_path)
        if payload is None:
            return
        ts_ms = payload.get("ts_ms", 0)
        if ts_ms <= self._last_dom_ts:
            return  # no new data
        if (now_ms - ts_ms) > STALE_THRESHOLD_MS:
            LOGGER.debug("DOM snapshot is stale by %d ms — skipping", now_ms - ts_ms)
            return
        self._last_dom_ts = ts_ms
        # Stamp the symbol in case the ACSIL study didn't set it
        payload.setdefault("symbol", self._symbol)
        channel = f"market:dom:{payload['symbol']}"
        try:
            await self._redis.publish(channel, json.dumps(payload))
        except Exception:
            LOGGER.debug("Redis publish failed for %s", channel, exc_info=True)

    async def _maybe_publish_flow(self, now_ms: int) -> None:
        payload = _read_json_file(self._flow_path)
        if payload is None:
            return
        ts_ms = payload.get("ts_ms", 0)
        if ts_ms <= self._last_flow_ts:
            return
        if (now_ms - ts_ms) > STALE_THRESHOLD_MS:
            return
        self._last_flow_ts = ts_ms
        payload.setdefault("symbol", self._symbol)
        channel = f"market:cvd:{payload['symbol']}"
        try:
            await self._redis.publish(channel, json.dumps(payload))
        except Exception:
            LOGGER.debug("Redis publish failed for %s", channel, exc_info=True)

    # ------------------------------------------------------------------
    # Redis → danger_trigger.json
    # ------------------------------------------------------------------

    async def _danger_subscriber(self) -> None:
        """Subscribe to sweep:danger:{symbol}; write danger_trigger.json."""
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
                "DOMBridge: wrote danger_trigger.json  severity=%s reason=%s",
                payload["severity"], payload["reason"],
            )
        except Exception:
            LOGGER.exception("Failed to write danger_trigger.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Standalone entry (for testing outside main runner)
# ---------------------------------------------------------------------------

async def _main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    svc = SierraDOMBridgeService()
    await svc.start()


if __name__ == "__main__":
    asyncio.run(_main())
