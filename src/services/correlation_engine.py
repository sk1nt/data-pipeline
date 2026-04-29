"""Real-time correlation engine: detects when social/news events coincide with market signals."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import deque
from datetime import datetime, timezone, timedelta, time
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from src.lib.redis_client import RedisClient
from src.models.social_event import SocialEvent
from src.services.social_feed_service import SOCIAL_EVENTS_CHANNEL

LOGGER = logging.getLogger(__name__)

CORRELATION_ALERT_CHANNEL = "correlation:alerts:stream"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

# Only correlate TastyTrade data for these symbols
ALLOWED_SYMBOLS = {"MNQ", "MES"}

# Only use GEX snapshots for this underlying (NQ / NDX)
GEX_SYMBOL = "NQ_NDX"

# GEX is unreliable pre-open and late afternoon; only correlate during RTH core hours
_ET = ZoneInfo("America/New_York")
_GEX_WINDOW_START = time(9, 45)   # 9:45 AM ET
_GEX_WINDOW_END   = time(15, 0)   # 3:00 PM ET


def _in_gex_window() -> bool:
    """Return True when GEX readings are reliable (9:45 AM – 3:00 PM ET)."""
    now_et = datetime.now(_ET).time()
    return _GEX_WINDOW_START <= now_et < _GEX_WINDOW_END


# TastyTrade trade channel (volume + price source)
TASTYTRADE_TRADE_CHANNEL = "market_data:tastytrade:trades"


class MarketSignalSnapshot(BaseModel):
    """Point-in-time market microstructure signals."""

    timestamp: datetime
    symbol: str = ""
    # Volume
    volume_1min: Optional[float] = None
    volume_20bar_avg: Optional[float] = None
    volume_ratio: Optional[float] = None
    # GEX
    net_gex: Optional[float] = None
    prev_net_gex: Optional[float] = None
    gex_change_pct: Optional[float] = None
    # Price
    price: Optional[float] = None
    price_2min_ago: Optional[float] = None
    price_change_pct: Optional[float] = None


class CorrelationAlert(BaseModel):
    """An alert produced by the correlation engine."""

    alert_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    alert_type: str  # volume_spike, gex_shift, price_move, confluence
    social_event: SocialEvent
    market_signals: MarketSignalSnapshot
    signals_triggered: List[str] = Field(default_factory=list)
    message: str = ""
    severity: str = "medium"


# ---------------------------------------------------------------------------
# Rolling event window
# ---------------------------------------------------------------------------

class EventWindow:
    """Thread-safe rolling buffer of social events and market signal snapshots."""

    def __init__(self, window_seconds: int = 300) -> None:
        self.window_seconds = window_seconds
        self._social_events: deque[SocialEvent] = deque()
        self._signals: deque[MarketSignalSnapshot] = deque()
        self._lock = Lock()

    def add_social_event(self, event: SocialEvent) -> None:
        with self._lock:
            self._social_events.append(event)
            self._evict()

    def add_signal(self, signal: MarketSignalSnapshot) -> None:
        with self._lock:
            self._signals.append(signal)
            self._evict()

    def get_recent_social_events(self, within_seconds: Optional[int] = None) -> List[SocialEvent]:
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=within_seconds or self.window_seconds
        )
        with self._lock:
            self._evict()
            return [e for e in self._social_events if e.timestamp >= cutoff]

    def get_latest_signal(self) -> Optional[MarketSignalSnapshot]:
        with self._lock:
            return self._signals[-1] if self._signals else None

    def _evict(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.window_seconds)
        while self._social_events and self._social_events[0].timestamp < cutoff:
            self._social_events.popleft()
        while self._signals and self._signals[0].timestamp < cutoff:
            self._signals.popleft()


# ---------------------------------------------------------------------------
# Correlation rules
# ---------------------------------------------------------------------------

def _check_volume_spike(
    event: SocialEvent, signal: MarketSignalSnapshot, multiplier: float
) -> Optional[str]:
    if (
        signal.volume_ratio is not None
        and signal.volume_ratio >= multiplier
    ):
        return (
            f"🚨 **VOLUME SPIKE** after social event\n"
            f"> {event.text[:120]}\n"
            f"Volume ratio: **{signal.volume_ratio:.1f}×** avg "
            f"({signal.volume_1min:.0f} vs {signal.volume_20bar_avg:.0f}) | "
            f"{signal.symbol}"
        )
    return None


def _check_gex_shift(
    event: SocialEvent, signal: MarketSignalSnapshot, pct_threshold: float
) -> Optional[str]:
    if (
        signal.gex_change_pct is not None
        and abs(signal.gex_change_pct) >= pct_threshold
    ):
        direction = "⬆️" if signal.gex_change_pct > 0 else "⬇️"
        return (
            f"🧲 **NQ GEX SHIFT** {direction} after social event\n"
            f"> {event.text[:120]}\n"
            f"NQ Net GEX: {signal.prev_net_gex:.0f} → {signal.net_gex:.0f} "
            f"(**{signal.gex_change_pct:+.1f}%**)"
        )
    return None


def _check_price_move(
    event: SocialEvent, signal: MarketSignalSnapshot, pct_threshold: float
) -> Optional[str]:
    if (
        signal.price_change_pct is not None
        and abs(signal.price_change_pct) >= pct_threshold
    ):
        direction = "📈" if signal.price_change_pct > 0 else "📉"
        return (
            f"{direction} **PRICE MOVE** after social event\n"
            f"> {event.text[:120]}\n"
            f"{signal.symbol}: **{signal.price_change_pct:+.2f}%** "
            f"({signal.price_2min_ago:.2f} → {signal.price:.2f})"
        )
    return None


# ---------------------------------------------------------------------------
# Correlation Engine
# ---------------------------------------------------------------------------

class CorrelationEngine:
    """Subscribe to social events + market data, detect correlations, publish alerts."""

    def __init__(
        self,
        redis_client: RedisClient,
        *,
        window_seconds: int = 300,
        volume_spike_multiplier: float = 2.0,
        gex_shift_pct: float = 15.0,
        price_move_pct: float = 0.3,
        cooldown_seconds: int = 300,
    ) -> None:
        self.redis = redis_client
        self.window = EventWindow(window_seconds)
        self.volume_multiplier = volume_spike_multiplier
        self.gex_pct = gex_shift_pct
        self.price_pct = price_move_pct
        self.cooldown = cooldown_seconds
        self._cooldowns: Dict[str, datetime] = {}  # event_id:rule → last_alert_time
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Market state tracked from Redis streams (per-symbol)
        self._volume_bars: Dict[str, deque] = {}  # symbol → deque of (ts, vol)
        self._last_gex: Optional[float] = None
        self._price_history: Dict[str, deque] = {}  # symbol → deque of (ts, price)

    def start(self) -> None:
        if self._task and not self._task.done():
            LOGGER.warning("CorrelationEngine already running")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="correlation-engine")
        LOGGER.info("CorrelationEngine started")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._stop_event.set()
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        LOGGER.info("CorrelationEngine stopped")

    async def _run(self) -> None:
        """Main loop: subscribe to all channels and process messages."""
        pubsub = self.redis.client.pubsub()
        channels = [
            SOCIAL_EVENTS_CHANNEL,
            "gex:snapshot:stream",
            TASTYTRADE_TRADE_CHANNEL,
        ]
        pubsub.subscribe(*channels)
        LOGGER.info("CorrelationEngine subscribed to: %s", channels)

        try:
            while not self._stop_event.is_set():
                message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message["type"] == "message":
                    channel = message["channel"]
                    if isinstance(channel, bytes):
                        channel = channel.decode()
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode()

                    try:
                        self._handle_message(channel, data)
                    except Exception:
                        LOGGER.exception("Error handling message on %s", channel)

                await asyncio.sleep(0.01)
        finally:
            pubsub.unsubscribe()
            pubsub.close()

    def _handle_message(self, channel: str, raw_data: str) -> None:
        """Route incoming messages to appropriate handlers."""
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            return

        now = datetime.now(timezone.utc)

        if channel == SOCIAL_EVENTS_CHANNEL:
            event = SocialEvent(**data)
            self.window.add_social_event(event)
            # Check correlations immediately on new social event
            self._check_correlations(event)

        elif channel == "gex:snapshot:stream":
            # Only use NQ_NDX GEX snapshots
            snapshot_symbol = (data.get("symbol") or data.get("ticker") or "").upper()
            if snapshot_symbol != GEX_SYMBOL:
                return
            net_gex = data.get("net_gex") or data.get("sum_gex_vol")
            if net_gex is not None:
                prev = self._last_gex
                self._last_gex = float(net_gex)
                signal = self._build_signal(now)
                if prev is not None and prev != 0:
                    signal.prev_net_gex = prev
                    signal.net_gex = self._last_gex
                    signal.gex_change_pct = ((self._last_gex - prev) / abs(prev)) * 100
                    signal.symbol = GEX_SYMBOL
                self.window.add_signal(signal)
                self._check_correlations_for_market_signal()

        elif channel == TASTYTRADE_TRADE_CHANNEL:
            symbol = (data.get("symbol") or "").upper()
            if symbol not in ALLOWED_SYMBOLS:
                return
            price = data.get("price")
            volume = data.get("size") or data.get("volume", 0)
            if price is not None:
                hist = self._price_history.setdefault(symbol, deque(maxlen=120))
                hist.append((now, float(price)))
            if volume:
                self._update_volume_bar(symbol, now, float(volume))
            signal = self._build_signal(now, symbol)
            signal.symbol = symbol
            self.window.add_signal(signal)
            self._check_correlations_for_market_signal()

    def _build_signal(self, now: datetime, symbol: str = "") -> MarketSignalSnapshot:
        """Construct a MarketSignalSnapshot from current tracked state."""
        signal = MarketSignalSnapshot(timestamp=now, symbol=symbol)

        # Volume (per-symbol)
        bars = self._volume_bars.get(symbol)
        if bars and len(bars) >= 2:
            avg = sum(v for _, v in bars) / len(bars)
            latest = bars[-1][1]
            signal.volume_1min = latest
            signal.volume_20bar_avg = avg
            signal.volume_ratio = latest / avg if avg > 0 else None

        # Price (per-symbol)
        hist = self._price_history.get(symbol)
        if hist and len(hist) >= 2:
            signal.price = hist[-1][1]
            cutoff = now - timedelta(seconds=120)
            older = [p for t, p in hist if t <= cutoff]
            if older:
                signal.price_2min_ago = older[-1]
                if signal.price_2min_ago != 0:
                    signal.price_change_pct = (
                        (signal.price - signal.price_2min_ago) / signal.price_2min_ago
                    ) * 100

        return signal

    def _update_volume_bar(self, symbol: str, now: datetime, volume: float) -> None:
        """Accumulate volume into 1-minute bars (per-symbol)."""
        bars = self._volume_bars.setdefault(symbol, deque(maxlen=20))
        if bars:
            last_ts, last_vol = bars[-1]
            if (now - last_ts).total_seconds() < 60:
                bars[-1] = (last_ts, last_vol + volume)
                return
        bars.append((now, volume))

    def _check_correlations_for_market_signal(self) -> None:
        """When a new market signal arrives, check if any recent social events correlate."""
        recent_events = self.window.get_recent_social_events()
        for event in recent_events:
            self._check_correlations(event)

    def _check_correlations(self, event: SocialEvent) -> None:
        """Run all correlation rules for a social event against the latest market signal."""
        signal = self.window.get_latest_signal()
        if signal is None:
            return

        triggered_signals: List[str] = []
        messages: List[str] = []
        gex_active = _in_gex_window()

        # Rule 1: Volume spike
        msg = _check_volume_spike(event, signal, self.volume_multiplier)
        if msg and not self._is_cooled_down(event.event_id, "volume_spike"):
            triggered_signals.append("volume_spike")
            messages.append(msg)

        # Rule 2: GEX shift — only during RTH core hours (9:45 AM – 3:00 PM ET)
        if gex_active:
            msg = _check_gex_shift(event, signal, self.gex_pct)
            if msg and not self._is_cooled_down(event.event_id, "gex_shift"):
                triggered_signals.append("gex_shift")
                messages.append(msg)

        # Rule 3: Price move
        msg = _check_price_move(event, signal, self.price_pct)
        if msg and not self._is_cooled_down(event.event_id, "price_move"):
            triggered_signals.append("price_move")
            messages.append(msg)

        if not triggered_signals:
            return

        # Outside RTH core hours: require both volume spike AND price move (3-tuple
        # confirmation: social event + volume + price).  Single-signal alerts without
        # GEX context are too noisy pre-open or late in the session.
        if not gex_active and len(triggered_signals) < 2:
            return

        # Rule 5: Confluence if >= 2 signals
        if len(triggered_signals) >= 2:
            alert_type = "confluence"
            severity = "high"
            confluence_msg = (
                f"⚡ **CONFLUENCE ALERT**: {len(triggered_signals)} signals after social event\n"
                f"> {event.text[:120]}\n"
                f"Signals: {', '.join(triggered_signals)}"
            )
            combined_message = confluence_msg + "\n\n" + "\n\n".join(messages)
        else:
            alert_type = triggered_signals[0]
            severity = "medium"
            combined_message = messages[0]

        alert = CorrelationAlert(
            alert_type=alert_type,
            social_event=event,
            market_signals=signal,
            signals_triggered=triggered_signals,
            message=combined_message,
            severity=severity,
        )

        # Mark cooldowns
        for sig in triggered_signals:
            self._set_cooldown(event.event_id, sig)

        self._publish_alert(alert)

    def _is_cooled_down(self, event_id: str, rule: str) -> bool:
        key = f"{event_id}:{rule}"
        last = self._cooldowns.get(key)
        if last is None:
            return False
        return (datetime.now(timezone.utc) - last).total_seconds() < self.cooldown

    def _set_cooldown(self, event_id: str, rule: str) -> None:
        self._cooldowns[f"{event_id}:{rule}"] = datetime.now(timezone.utc)
        # Prune old cooldowns
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.cooldown * 2)
        self._cooldowns = {
            k: v for k, v in self._cooldowns.items() if v > cutoff
        }

    def _publish_alert(self, alert: CorrelationAlert) -> None:
        try:
            payload = alert.model_dump(mode="json")
            serialized = json.dumps(payload, default=str)
            self.redis.client.publish(CORRELATION_ALERT_CHANNEL, serialized)
            LOGGER.info(
                "Correlation alert published: type=%s, severity=%s, signals=%s",
                alert.alert_type,
                alert.severity,
                alert.signals_triggered,
            )
        except Exception:
            LOGGER.exception("Failed to publish correlation alert %s", alert.alert_id)
