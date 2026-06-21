"""Real-time options flow scanner for large / unusual option moves.

Subscribes to the UW option-trade Redis pubsub stream
(``uw:option_trade:stream``) and detects:

  * **IV spike**         -- |z-score| of IV > 2 over a 20-trade rolling window
  * **Premium surge**    -- trade premium > 5x rolling avg premium (50 trades)
  * **Volume/OI boom**   -- cumulative contracts traded > 2x open_interest in
                            a 5-minute window

On detection, alerts are published to ``options_flow:alert:{symbol}`` with the
payload ``{symbol, type, severity, value, threshold, details, trade_data}``.

Trade messages are the stamped UW dicts produced by ``UWMessageService`` (with a
``topic_symbol`` field and a nested ``data`` dict of ``OptionTradeData``), but a
raw option-data dict is also accepted.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.lib.redis_client import RedisClient

LOGGER = logging.getLogger(__name__)

_OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})[CP]\d{8}$")

# (symbol, iv, premium, contracts, open_interest, ts_ms)
ExtractedTrade = Tuple[
    str, Optional[float], Optional[float], Optional[int], Optional[int], float
]


class OptionsFlowScanner:
    """Scan UW option trades in real time and alert on unusual flow."""

    OPTION_TRADE_CHANNEL = "uw:option_trade:stream"
    ALERT_CHANNEL = "options_flow:alert:{symbol}"
    ALERT_HISTORY_KEY = "options_flow:alert:history"
    HISTORY_LIMIT = 200

    # Detector windows / thresholds
    IV_WINDOW = 20
    IV_MIN_SAMPLES = 10
    IV_ZSCORE_THRESHOLD = 2.0
    PREMIUM_WINDOW = 50
    PREMIUM_MIN_SAMPLES = 10
    PREMIUM_SURGE_MULTIPLIER = 5.0
    VOLUME_OI_WINDOW_SECONDS = 300
    VOLUME_OI_MULTIPLIER = 2.0

    def __init__(self, redis_client: RedisClient) -> None:
        self.redis_client = redis_client
        self._iv_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.IV_WINDOW)
        )
        self._premium_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.PREMIUM_WINDOW)
        )
        self._volume_window: Dict[str, Deque[Tuple[float, int]]] = defaultdict(
            lambda: deque(maxlen=2000)
        )
        self._last_oi: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._alerts_published = 0

    # ------------------------------------------------------------------ #
    # Trade ingestion
    # ------------------------------------------------------------------ #
    def process_trade(self, trade_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run all detectors against a single option trade.

        Args:
            trade_data: Stamped UW message dict (``topic_symbol`` + nested
                ``data``) or a raw option-data dict.

        Returns:
            List of alerts generated (each is also published to Redis).
        """
        parsed = self._extract(trade_data)
        if parsed is None:
            return []
        symbol, iv, premium, contracts, open_interest, ts_ms = parsed
        alerts: List[Dict[str, Any]] = []
        with self._lock:
            if iv is not None:
                alert = self._detect_iv_spike(symbol, iv, trade_data)
                if alert:
                    alerts.append(alert)
            if premium is not None:
                alert = self._detect_premium_surge(symbol, premium, trade_data)
                if alert:
                    alerts.append(alert)
            if contracts is not None:
                alert = self._detect_volume_oi_explosion(
                    symbol, contracts, open_interest, ts_ms, trade_data
                )
                if alert:
                    alerts.append(alert)
        for alert in alerts:
            self._publish_alert(symbol, alert)
        return alerts

    def _extract(self, trade_data: Dict[str, Any]) -> Optional[ExtractedTrade]:
        if not isinstance(trade_data, dict):
            return None
        data = trade_data.get("data", trade_data)
        if not isinstance(data, dict):
            return None

        symbol = str(
            trade_data.get("topic_symbol") or data.get("symbol") or ""
        ).upper() or None
        if not symbol:
            symbol = self._ticker_from_occ(data.get("option_chain_id"))
        if not symbol:
            return None

        iv = _to_float(data.get("implied_volatility"))
        premium = _to_float(data.get("premium"))
        size = _to_int(data.get("size"))
        volume = _to_int(data.get("volume"))
        # Prefer per-trade ``size`` (contracts); fall back to cumulative volume.
        contracts = size if size is not None else volume
        open_interest = _to_int(data.get("open_interest"))
        ts_ms = self._timestamp_ms(
            data.get("executed_at") or trade_data.get("received_at")
        )
        return symbol, iv, premium, contracts, open_interest, ts_ms

    @staticmethod
    def _ticker_from_occ(option_chain_id: Optional[str]) -> Optional[str]:
        if not option_chain_id:
            return None
        match = _OCC_RE.match(str(option_chain_id))
        return match.group(1) if match else None

    @staticmethod
    def _timestamp_ms(value: Any) -> float:
        if value is None:
            return time.time() * 1000.0
        if isinstance(value, (int, float)):
            return float(value)
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp() * 1000.0
        except ValueError:
            return time.time() * 1000.0

    # ------------------------------------------------------------------ #
    # Detectors
    # ------------------------------------------------------------------ #
    def _detect_iv_spike(
        self, symbol: str, iv: float, trade_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Flag when IV z-score over the rolling window exceeds the threshold."""
        history = self._iv_history[symbol]
        history.append(iv)
        if len(history) < self.IV_MIN_SAMPLES:
            return None
        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std = math.sqrt(variance)
        if std == 0:
            return None
        z = (iv - mean) / std
        if abs(z) <= self.IV_ZSCORE_THRESHOLD:
            return None
        return self._build_alert(
            symbol=symbol,
            alert_type="IV_SPIKE",
            severity="critical" if abs(z) > 3.0 else "high",
            value=round(z, 3),
            threshold=self.IV_ZSCORE_THRESHOLD,
            extra={
                "iv": iv,
                "rolling_mean": round(mean, 6),
                "rolling_std": round(std, 6),
                "window": self.IV_WINDOW,
                "direction": "up" if z > 0 else "down",
            },
            trade_data=trade_data,
        )

    def _detect_premium_surge(
        self, symbol: str, premium: float, trade_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Flag when a trade's premium is a large multiple of the rolling avg."""
        history = self._premium_history[symbol]
        history.append(premium)
        if len(history) < self.PREMIUM_MIN_SAMPLES:
            return None
        avg = sum(history) / len(history)
        if avg <= 0:
            return None
        multiple = premium / avg
        if multiple < self.PREMIUM_SURGE_MULTIPLIER:
            return None
        return self._build_alert(
            symbol=symbol,
            alert_type="PREMIUM_SURGE",
            severity="critical" if multiple > 10.0 else "high",
            value=round(multiple, 2),
            threshold=self.PREMIUM_SURGE_MULTIPLIER,
            extra={
                "premium": premium,
                "rolling_avg": round(avg, 6),
                "window": self.PREMIUM_WINDOW,
            },
            trade_data=trade_data,
        )

    def _detect_volume_oi_explosion(
        self,
        symbol: str,
        contracts: int,
        open_interest: Optional[int],
        ts_ms: float,
        trade_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Flag when cumulative contracts in the window exceed 2x open interest."""
        if open_interest and open_interest > 0:
            self._last_oi[symbol] = open_interest
        window = self._volume_window[symbol]
        if contracts and contracts > 0:
            window.append((ts_ms, contracts))
        cutoff = ts_ms - self.VOLUME_OI_WINDOW_SECONDS * 1000.0
        while window and window[0][0] < cutoff:
            window.popleft()
        oi = self._last_oi.get(symbol)
        if not oi or oi <= 0:
            return None
        cumulative = sum(v for _, v in window)
        ratio = cumulative / oi
        if ratio < self.VOLUME_OI_MULTIPLIER:
            return None
        return self._build_alert(
            symbol=symbol,
            alert_type="VOLUME_OI_EXPLOSION",
            severity="high",
            value=round(ratio, 2),
            threshold=self.VOLUME_OI_MULTIPLIER,
            extra={
                "cumulative_volume": cumulative,
                "open_interest": oi,
                "window_seconds": self.VOLUME_OI_WINDOW_SECONDS,
            },
            trade_data=trade_data,
        )

    @staticmethod
    def _build_alert(
        symbol: str,
        alert_type: str,
        severity: str,
        value: float,
        threshold: float,
        extra: Dict[str, Any],
        trade_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "type": alert_type,
            "severity": severity,
            "value": value,
            "threshold": threshold,
            "details": extra,
            "trade_data": trade_data,
        }

    # ------------------------------------------------------------------ #
    # Publishing
    # ------------------------------------------------------------------ #
    def _publish_alert(self, symbol: str, alert: Dict[str, Any]) -> None:
        self._alerts_published += 1
        try:
            conn = self.redis_client.client
            serialized = json.dumps(alert, default=str)
            conn.publish(self.ALERT_CHANNEL.format(symbol=symbol), serialized)
            conn.lpush(self.ALERT_HISTORY_KEY, serialized)
            conn.ltrim(self.ALERT_HISTORY_KEY, 0, self.HISTORY_LIMIT - 1)
            LOGGER.info(
                "Published options flow alert: %s %s (value=%.2f)",
                symbol,
                alert["type"],
                alert["value"],
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to publish options flow alert: %s", exc)

    # ------------------------------------------------------------------ #
    # Pubsub subscription loop
    # ------------------------------------------------------------------ #
    def subscribe(
        self, stop_event: Optional[threading.Event] = None
    ) -> None:
        """Block on the option-trade Redis pubsub channel and scan trades.

        Intended to run in a dedicated thread. Returns when ``stop_event`` is
        set or the connection drops.
        """
        conn = self.redis_client.client
        pubsub = conn.pubsub()
        pubsub.subscribe(self.OPTION_TRADE_CHANNEL)
        LOGGER.info(
            "OptionsFlowScanner subscribed to %s", self.OPTION_TRADE_CHANNEL
        )
        try:
            while stop_event is None or not stop_event.is_set():
                message = pubsub.get_message(timeout=1.0)
                if message is None:
                    continue
                if message.get("type") != "message":
                    continue
                self._handle_raw(message.get("data"))
        finally:
            try:
                pubsub.unsubscribe(self.OPTION_TRADE_CHANNEL)
                pubsub.close()
            except Exception:  # noqa: BLE001
                pass
            LOGGER.info("OptionsFlowScanner unsubscribed")

    def _handle_raw(self, raw: Any) -> None:
        try:
            payload = (
                json.loads(raw) if isinstance(raw, (str, bytes)) else raw
            )
        except (TypeError, ValueError) as exc:
            LOGGER.warning("Skipping non-JSON option trade message: %s", exc)
            return
        if not isinstance(payload, dict):
            return
        try:
            self.process_trade(payload)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Error processing option trade: %s", exc)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
