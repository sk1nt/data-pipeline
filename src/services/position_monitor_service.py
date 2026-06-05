"""position_monitor_service.py — danger level escalation and flatten trigger.

Subscribes to sweep:alert:{symbol}.  When a directional-break alert arrives
while the trader is positioned in the wrong direction and unrealized loss
exceeds a configured threshold, escalates through levels 1-3.

Escalation levels
─────────────────
    Level 1 — WARNING:
        confidence >= SWEEP_WARNING_CONFIDENCE (0.65) AND
        unrealized loss >= SWEEP_WARNING_TICKS (15 ticks)
        Action: publishes sweep:monitor:{symbol} with level=1 (dashboard alert)

    Level 2 — DANGER:
        confidence >= SWEEP_DANGER_CONFIDENCE (0.78) AND
        unrealized loss >= SWEEP_DANGER_TICKS (25 ticks)
        Action: publishes level=2, recommend reduce 50%
        Starts ACK_TIMEOUT_SECONDS (10 s) countdown to Level 3

    Level 3 — CRITICAL:
        confidence >= SWEEP_CRITICAL_CONFIDENCE (0.85) AND
        unrealized loss >= SWEEP_CRITICAL_TICKS (35 ticks) AND
        Level 2 fired AND no manual ack within ACK_TIMEOUT_SECONDS
        Action:
          • SWEEP_LIVE_MODE=true  → publishes sweep:danger:{symbol}
            → SierraDOMBridgeService writes danger_trigger.json
            → ACSIL study calls sc.FlattenAndCancelAllOrders()
          • SWEEP_LIVE_MODE=false → logs the escalation but does NOT flatten

Design philosophy — intelligent exits, not static stops
────────────────────────────────────────────────────────
    This service does NOT implement a traditional stop loss.  The tick
    thresholds (WARNING/DANGER/CRITICAL) are GATES, not triggers — they
    filter out noise so that only alerts with sufficient adverse excursion
    AND high classifier confidence result in action.

    A position is only closed when there is positive evidence that the
    move is DIRECTIONAL (sweep_classifier classifies it as such with high
    confidence), not simply because price moved X ticks away from entry.

    Future capabilities planned in this service:

    TOD Position Sizing
        Time-of-day aware contract scaling.  The danger thresholds and
        confidence requirements should relax or tighten based on session
        regime (power hour vs. midday chop vs. close).  Smaller size in
        low-edge windows means the same dollar risk is met at fewer ticks.
        Add to PositionMonitorService: a TOD regime lookup that adjusts
        DANGER_TICKS dynamically rather than using static env config.

    GEX Wall Take-Profit Targets
        Natural TP levels exist at major GEX walls (major_pos_call1_strike,
        major_neg_put1_strike from the gex:snapshot:stream payload).  Price
        tends to stall or reverse at these levels.  Add a TP monitor that:
          1. Reads current position direction and entry price.
          2. Identifies the nearest GEX wall in the profit direction.
          3. Publishes a TP signal to sweep:monitor:{symbol} when price
             approaches that wall, letting the trader decide to take partial
             or full profit at a structurally sound level.
        This gives a REASON to exit winners — not just losers.

Manual ack (suppresses all levels temporarily)
───────────────────────────────────────────────
    Publish any message to sweep:ack:{symbol}.
    Example from CLI:
        redis-cli publish sweep:ack:MNQ '{"note":"manual override"}'
    This sets a suppress timer equal to SWEEP_LEVEL2_ACKNOWLEDGE_SECONDS
    and resets the danger level to 0.

Redis channels consumed
───────────────────────
    sweep:alert:{symbol}   — SweepAlert from SweepClassifierService
    sweep:ack:{symbol}     — manual ack from trader / dashboard

Redis channels published
────────────────────────
    sweep:monitor:{symbol}
        Payload:
            ts_ms           int
            symbol          str
            level           int      — 0-3
            classification  str      — from sweep alert
            confidence      float
            direction       str      — "up" | "down"
            position_qty    int      — from position source
            position_dir    str      — "long" | "short" | None
            loss_ticks      float    — estimated unrealized loss in ticks
            message         str      — human-readable escalation note

    sweep:danger:{symbol}  — ONLY when SWEEP_LIVE_MODE=true AND level 3 fires
        Payload: {"action": "flatten", "reason": "...", "severity": "critical"}

Position source (priority order)
──────────────────────────────────
    1. SC position file:  SC_POSITIONS_PATH
       Format (JSON written by a second ACSIL study, e.g. position_bridge.cpp):
           {"quantity": 2, "entry_price": 19850.5, "symbol": "MNQ"}
       quantity: positive = long, negative = short, 0 = flat

    2. SC_POSITION_OVERRIDE env var (integer):
       Useful for testing: SC_POSITION_OVERRIDE=2  (long 2 contracts)
       Set SC_POSITION_OVERRIDE=-1 to simulate short 1 contract.

    3. Fallback: position unknown — monitor still emits level 1-2 warnings
       but danger_level on alerts will reflect that position is unverified.

NOTE: A position_bridge ACSIL study is not yet implemented.  Until it is,
use SC_POSITION_OVERRIDE or manually set the JSON file to match your
live position before arming SWEEP_LIVE_MODE=true.  The position_snapshot.json
file can also be written by any external process that reads SC's positions API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import redis.asyncio as aioredis

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REDIS_URL            = os.getenv("REDIS_URL",             "redis://localhost:6379/0")
DOM_SYMBOL           = os.getenv("SC_DOM_SYMBOL",         "MNQ")
MNQ_TICK_VALUE       = float(os.getenv("MNQ_TICK_VALUE",  "0.25"))
SWEEP_LIVE_MODE      = os.getenv("SWEEP_LIVE_MODE",       "false").lower() == "true"

WARNING_TICKS        = int(os.getenv("SWEEP_WARNING_TICKS",      "15"))
DANGER_TICKS         = int(os.getenv("SWEEP_DANGER_TICKS",       "25"))
CRITICAL_TICKS       = int(os.getenv("SWEEP_CRITICAL_TICKS",     "35"))
WARNING_CONFIDENCE   = float(os.getenv("SWEEP_WARNING_CONFIDENCE",  "0.65"))
DANGER_CONFIDENCE    = float(os.getenv("SWEEP_DANGER_CONFIDENCE",   "0.78"))
CRITICAL_CONFIDENCE  = float(os.getenv("SWEEP_CRITICAL_CONFIDENCE", "0.85"))
ACK_TIMEOUT_SECONDS  = int(os.getenv("SWEEP_LEVEL2_ACKNOWLEDGE_SECONDS", "10"))

SC_POSITIONS_PATH    = os.getenv("SC_POSITIONS_PATH", "")
SC_POSITION_OVERRIDE = os.getenv("SC_POSITION_OVERRIDE", "")  # integer, e.g. "2" or "-1"

# GEX wall TP: publish a TP signal when price is within this many ticks of a wall
TP_WALL_PROXIMITY_TICKS = int(os.getenv("SWEEP_TP_WALL_PROXIMITY_TICKS", "6"))
# GEX snapshot Redis key prefix (published by data-pipeline.py on server)
GEX_SNAPSHOT_CHANNEL   = os.getenv("GEX_SNAPSHOT_CHANNEL", "gex:snapshot:stream")

# Re-export label_outcome so tests can import it from this module
try:
    from src.ml.sweep_feature_extractor import label_outcome  # noqa: F401
except ImportError:
    def label_outcome(trigger_price: float, direction: str, price_t60: float) -> str:  # type: ignore[misc]
        """Stub when sweep_feature_extractor is unavailable."""
        return "ambiguous"

# TOD multipliers: scale confidence/tick thresholds per session
# Each session → (confidence_multiplier, tick_multiplier)
# e.g. midday (choppy): require MORE confidence but FEWER ticks to trigger
_TOD_MULTIPLIERS = {
    "power_open":  (1.00, 1.00),  # 09:30-10:00 ET  — standard
    "morning":     (1.00, 1.00),  # 10:00-12:00 ET  — standard
    "midday":      (1.05, 0.90),  # 12:00-14:00 ET  — more noise; require higher conf
    "power_close": (0.95, 1.10),  # 14:00-16:00 ET  — high edge; slightly looser conf
    "overnight":   (1.10, 0.80),  # outside RTH     — low edge; require strong signal
}


def _tod_session() -> str:
    """Return session name for the current UTC time mapped to ET."""
    try:
        now_utc = datetime.now(timezone.utc)
        # ET offset: -5h standard, -4h daylight saving.  Use a simple heuristic.
        # DST in effect roughly March 2nd Sunday through Nov 1st Sunday.
        month = now_utc.month
        dst = 3 <= month <= 11
        offset_hours = 4 if dst else 5
        et_hour = (now_utc.hour - offset_hours) % 24
        et_minutes = et_hour * 60 + now_utc.minute

        if   570 <= et_minutes <  600: return "power_open"    # 09:30-10:00
        elif 600 <= et_minutes <  720: return "morning"       # 10:00-12:00
        elif 720 <= et_minutes <  840: return "midday"        # 12:00-14:00
        elif 840 <= et_minutes <  960: return "power_close"   # 14:00-16:00
        else:                          return "overnight"
    except Exception:
        return "morning"


def _dynamic_thresholds() -> Tuple[float, float, float, int, int, int]:
    """Return (warning_conf, danger_conf, critical_conf, warning_t, danger_t, critical_t)
    adjusted for the current time-of-day session."""
    session = _tod_session()
    conf_mult, tick_mult = _TOD_MULTIPLIERS.get(session, (1.0, 1.0))
    wc  = min(WARNING_CONFIDENCE  * conf_mult, 0.95)
    dc  = min(DANGER_CONFIDENCE   * conf_mult, 0.97)
    cc  = min(CRITICAL_CONFIDENCE * conf_mult, 0.99)
    wt  = max(int(WARNING_TICKS  * tick_mult), 5)
    dt  = max(int(DANGER_TICKS   * tick_mult), 10)
    ct  = max(int(CRITICAL_TICKS * tick_mult), 15)
    return wc, dc, cc, wt, dt, ct


# ---------------------------------------------------------------------------
# Position reader
# ---------------------------------------------------------------------------

class PositionState:
    """Current position: quantity (positive = long, negative = short) + entry price."""
    def __init__(self) -> None:
        self.quantity:    int   = 0
        self.entry_price: float = 0.0
        self.source:      str   = "unknown"

    def unrealized_loss_ticks(self, current_price: float) -> float:
        """Positive = losing money (adverse price move in ticks, per-contract)."""
        if self.quantity == 0:
            return 0.0
        if self.quantity > 0:   # long position
            move = current_price - self.entry_price   # negative when losing
        else:                   # short position
            move = self.entry_price - current_price   # negative when losing
        # Loss ticks = adverse price distance / tick_size  (quantity-independent)
        return -move / MNQ_TICK_VALUE

    def direction(self) -> Optional[str]:
        if self.quantity > 0:
            return "long"
        if self.quantity < 0:
            return "short"
        return None

    def in_wrong_direction(self, alert_direction: str) -> bool:
        """True if position would lose money on a move in alert_direction."""
        d = self.direction()
        if d is None:
            return False
        return (d == "long"  and alert_direction == "down") or \
               (d == "short" and alert_direction == "up")


def _read_position_from_file() -> Optional[PositionState]:
    if not SC_POSITIONS_PATH:
        return None
    try:
        text = Path(SC_POSITIONS_PATH).read_text(encoding="utf-8")
        data = json.loads(text)
        pos  = PositionState()
        pos.quantity    = int(data.get("quantity",    0))
        pos.entry_price = float(data.get("entry_price", 0.0))
        pos.source = "sc_file"
        return pos
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _read_position_override() -> Optional[PositionState]:
    if not SC_POSITION_OVERRIDE:
        return None
    try:
        pos = PositionState()
        pos.quantity = int(SC_POSITION_OVERRIDE)
        pos.source   = "env_override"
        return pos
    except ValueError:
        return None


def get_current_position() -> PositionState:
    pos = _read_position_from_file() or _read_position_override()
    if pos is None:
        pos = PositionState()
        pos.source = "unknown"
    return pos


# ---------------------------------------------------------------------------
# Monitor service
# ---------------------------------------------------------------------------

class PositionMonitorService:
    def __init__(self, redis_client: Optional[aioredis.Redis] = None) -> None:
        self._redis   = redis_client
        self._symbol  = DOM_SYMBOL
        self._running = False

        # Level 2 ack timer: timestamp when level 2 was last fired
        self._level2_fired_at: Optional[float] = None
        # Suppress all danger levels until this timestamp (manual override)
        self._suppress_until:  float = 0.0
        # Dashboard ack channel subscription
        self._ack_received: bool = False
        # Current danger level
        self._current_level: int = 0
        # Latest GEX snapshot payload (updated by _gex_subscriber)
        self._gex_snapshot: Optional[dict] = None
        # Last TP signal time to avoid repeated TP spam
        self._last_tp_signal_at: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._redis is None:
            self._redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
        self._running = True
        LOGGER.info(
            "PositionMonitorService starting — LIVE_MODE=%s  thresholds: warn=%dt danger=%dt crit=%dt",
            SWEEP_LIVE_MODE, WARNING_TICKS, DANGER_TICKS, CRITICAL_TICKS,
        )
        await asyncio.gather(
            self._alert_subscriber(),
            self._ack_subscriber(),
            self._gex_subscriber(),
        )

    async def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Alert subscriber
    # ------------------------------------------------------------------

    async def _alert_subscriber(self) -> None:
        sub = self._redis.pubsub()
        await sub.subscribe(f"sweep:alert:{self._symbol}")
        async for msg in sub.listen():
            if not self._running:
                break
            if msg["type"] != "message":
                continue
            try:
                alert = json.loads(msg["data"])
            except (json.JSONDecodeError, TypeError):
                continue
            await self._handle_alert(alert)

    async def _gex_subscriber(self) -> None:
        """Track latest GEX snapshot for wall-based TP signals."""
        sub = self._redis.pubsub()
        await sub.subscribe(GEX_SNAPSHOT_CHANNEL)
        async for msg in sub.listen():
            if not self._running:
                break
            if msg["type"] != "message":
                continue
            try:
                snap = json.loads(msg["data"])
                self._gex_snapshot = snap
                await self._check_tp_signal(snap)
            except (json.JSONDecodeError, TypeError):
                continue

    async def _check_tp_signal(
        self, gex: dict
    ) -> None:
        """Publish a TP signal when price approaches the nearest GEX wall in the
        position's profit direction."""
        pos = get_current_position()
        if pos.quantity == 0 or pos.entry_price == 0.0:
            return

        # Throttle: only one TP signal per 60 seconds
        if time.time() - self._last_tp_signal_at < 60.0:
            return

        spot = gex.get("spot")
        if not isinstance(spot, (int, float)) or spot <= 0:
            return

        direction = pos.direction()  # "long" or "short"

        # For a long position the TP wall is the nearest POSITIVE (call) wall above price.
        # For a short position the TP wall is the nearest NEGATIVE (put) wall below price.
        if direction == "long":
            wall_price = gex.get("major_pos_vol") or gex.get("major_pos_call1_strike")
        else:
            wall_price = gex.get("major_neg_vol") or gex.get("major_neg_put1_strike")

        if not isinstance(wall_price, (int, float)) or wall_price <= 0:
            return

        # Check whether price is within TP_WALL_PROXIMITY_TICKS of the wall
        # and that it's in the profit direction from entry
        dist_ticks = abs(spot - wall_price) / MNQ_TICK_VALUE
        if dist_ticks > TP_WALL_PROXIMITY_TICKS:
            return

        # Confirm the wall is in the profitable direction from entry
        if direction == "long" and wall_price <= pos.entry_price:
            return
        if direction == "short" and wall_price >= pos.entry_price:
            return

        profit_ticks = (
            (spot - pos.entry_price) / MNQ_TICK_VALUE
            if direction == "long"
            else (pos.entry_price - spot) / MNQ_TICK_VALUE
        )

        self._last_tp_signal_at = time.time()
        payload = {
            "ts_ms":          int(time.time() * 1000),
            "symbol":         self._symbol,
            "event":          "tp_wall_proximity",
            "danger_level":   self._current_level,
            "position_dir":   direction,
            "position_qty":   abs(pos.quantity),
            "entry_price":    pos.entry_price,
            "current_price":  spot,
            "wall_price":     wall_price,
            "dist_ticks":     round(dist_ticks, 1),
            "profit_ticks":   round(profit_ticks, 1),
            "message":        (
                f"Price {spot:.2f} within {dist_ticks:.1f}t of GEX "
                f"{'call' if direction == 'long' else 'put'} wall {wall_price:.2f}  "
                f"(+{profit_ticks:.1f}t profit)"
            ),
        }
        try:
            await self._redis.publish(
                f"sweep:monitor:{self._symbol}",
                json.dumps(payload),
            )
            LOGGER.info(
                "PositionMonitor TP signal: %s wall=%.2f dist=%.1ft profit=%.1ft",
                direction, wall_price, dist_ticks, profit_ticks,
            )
        except Exception:
            LOGGER.debug("Failed to publish TP signal", exc_info=True)

    async def _ack_subscriber(self) -> None:
        """Dashboard sends 'suppress' commands here to silence danger escalation."""
        sub = self._redis.pubsub()
        await sub.subscribe(f"sweep:ack:{self._symbol}")
        async for msg in sub.listen():
            if not self._running:
                break
            if msg["type"] != "message":
                continue
            try:
                cmd = json.loads(msg["data"])
                suppress_minutes = int(cmd.get("suppress_minutes", 5))
                self._suppress_until = time.time() + suppress_minutes * 60
                self._level2_fired_at = None
                self._current_level = 0
                LOGGER.info(
                    "PositionMonitor: danger suppressed for %d minutes by user", suppress_minutes
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    async def _handle_alert(self, alert: dict) -> None:
        now = time.time()

        # User suppressed
        if now < self._suppress_until:
            return

        classification = alert.get("classification", "sweep")
        confidence     = float(alert.get("confidence", 0.0))
        direction      = alert.get("direction", "")
        trigger_price  = float(alert.get("trigger_price", 0.0))

        if classification != "directional":
            # Sweep — reset danger level
            if self._current_level > 0:
                self._current_level = 0
                self._level2_fired_at = None
                await self._publish_level_update(0, "sweep_detected", alert)
            return

        # Read current position
        pos = get_current_position()

        if pos.quantity == 0:
            LOGGER.debug("PositionMonitor: directional alert but no position open")
            return

        if not pos.in_wrong_direction(direction):
            LOGGER.debug("PositionMonitor: directional alert but position is in right direction")
            return

        loss_ticks = pos.unrealized_loss_ticks(trigger_price)

        # Determine appropriate level using TOD-adjusted thresholds
        wc, dc, cc, wt, dt, ct = _dynamic_thresholds()
        new_level = 0
        if confidence >= wc and loss_ticks >= wt:
            new_level = 1
        if confidence >= dc and loss_ticks >= dt:
            new_level = 2
        if confidence >= cc and loss_ticks >= ct:
            new_level = 3

        if new_level == 0:
            return

        # Level 1 — warning
        if new_level >= 1 and self._current_level < 1:
            self._current_level = 1
            await self._publish_level_update(1, "directional_warning", alert)
            LOGGER.warning(
                "PositionMonitor LEVEL 1 WARNING: conf=%.2f loss=%.1f ticks  pos=%s×%d",
                confidence, loss_ticks, pos.direction(), abs(pos.quantity),
            )

        # Level 2 — danger (reduce)
        if new_level >= 2 and self._current_level < 2:
            self._current_level = 2
            self._level2_fired_at = now
            self._ack_received = False
            await self._publish_level_update(2, "directional_danger", alert)
            LOGGER.error(
                "PositionMonitor LEVEL 2 DANGER: conf=%.2f loss=%.1f ticks — reduce position!",
                confidence, loss_ticks,
            )

        # Level 3 — critical flatten
        if new_level >= 3 and self._current_level < 3:
            # Only escalate to 3 if level 2 fired first and went unacknowledged
            if self._level2_fired_at is not None:
                seconds_since_l2 = now - self._level2_fired_at
                if seconds_since_l2 >= ACK_TIMEOUT_SECONDS and not self._ack_received:
                    self._current_level = 3
                    await self._fire_critical_flatten(alert, loss_ticks, confidence)
            else:
                # Level 2 skipped somehow (racing alert) — fire level 2 first
                self._level2_fired_at = now

    async def _fire_critical_flatten(
        self, alert: dict, loss_ticks: float, confidence: float
    ) -> None:
        reason = (
            f"sweep_classifier:directional_break:"
            f"confidence={confidence:.2f}:loss_ticks={loss_ticks:.1f}"
        )
        LOGGER.critical(
            "PositionMonitor LEVEL 3 CRITICAL: %s — %s",
            "ARMED" if SWEEP_LIVE_MODE else "DRY RUN",
            reason,
        )
        await self._publish_level_update(3, "critical_flatten", alert)

        if SWEEP_LIVE_MODE:
            danger_payload = {
                "action":   "flatten",
                "reason":   reason,
                "severity": "critical",
            }
            try:
                await self._redis.publish(
                    f"sweep:danger:{self._symbol}",
                    json.dumps(danger_payload),
                )
                LOGGER.critical("PositionMonitor: FLATTEN signal published to sweep:danger:%s", self._symbol)
            except Exception:
                LOGGER.exception("Failed to publish danger flatten")
        else:
            LOGGER.critical("PositionMonitor: DRY RUN — flatten NOT executed (SWEEP_LIVE_MODE=false)")

    async def _publish_level_update(self, level: int, event: str, alert: dict) -> None:
        payload = {
            "ts_ms":          int(time.time() * 1000),
            "symbol":         self._symbol,
            "danger_level":   level,
            "event":          event,
            "classification": alert.get("classification"),
            "confidence":     alert.get("confidence"),
            "direction":      alert.get("direction"),
            "trigger_price":  alert.get("trigger_price"),
            "gex_regime":     alert.get("gex_regime"),
            "at_wall":        alert.get("at_wall"),
            "through_wall":   alert.get("through_wall"),
            "live_mode":      SWEEP_LIVE_MODE,
        }
        try:
            await self._redis.publish(
                f"sweep:monitor:{self._symbol}",
                json.dumps(payload),
            )
        except Exception:
            LOGGER.debug("Failed to publish monitor level update", exc_info=True)
