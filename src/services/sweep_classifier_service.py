"""sweep_classifier_service.py — detects fast moves and classifies sweep vs directional.

Pipeline
────────
1. Subscribe to market:dom:{symbol} and market:cvd:{symbol}
   (published by sierra_dom_bridge_service from SC DOM/CVD JSON files).
2. Subscribe to gex:snapshot:stream for live GEX walls + regime
   (published by data-pipeline.py on the Linux server).
3. Maintain a rolling price buffer; fire a trigger when price moves
   >= SWEEP_TRIGGER_TICKS in <= SWEEP_TRIGGER_WINDOW_SECONDS.
4. Extract features at trigger time (GEX context, order flow, DOM state, session).
5. Classify: rule-based fallback until sweep_model.pkl is trained.
6. Publish sweep:alert:{symbol} with classification + confidence.
7. Persist every trigger to DuckDB fast_moves table for ML training data.

Redis channels consumed
───────────────────────
    market:dom:{symbol}    — DOM snapshot from SierraDOMBridgeService
    market:cvd:{symbol}    — CVD snapshot from SierraDOMBridgeService
    gex:snapshot:stream    — GEX snapshot from data-pipeline.py on the server
                             Required fields: ticker, spot, zero_gamma,
                             major_pos_call1_strike, major_neg_put1_strike

Redis channels published
────────────────────────
    sweep:alert:{symbol}
        Payload (SweepAlert):
            ts_ms           int      — trigger time (UTC ms)
            symbol          str
            direction       str      — "up" | "down"
            trigger_price   float
            trigger_ticks   float    — price move that triggered this alert
            classification  str      — "sweep" | "directional"
            confidence      float    — 0.0 – 0.90 (rule-based cap) or 0.0 – 1.0 (ML)
            danger_level    int      — 0-3 escalation hint for PositionMonitorService
            model_version   str      — "rules_v1" | "xgb_v{n}"
            features        dict     — full SweepFeatures dict for downstream use

DuckDB — fast_moves table
──────────────────────────
    Path: SWEEP_DB_PATH (default: data/fast_moves.db)
    Every trigger (both classifications) is persisted here.
    The outcome column is NULL until sweep_feature_extractor.py backfills it
    using tick parquet data at T+60 seconds.
    This table is the training set for the XGBoost model.

    Running on trading machine: the data/fast_moves.db DuckDB file will
    accumulate locally.  Sync it to the dev machine for model training
    (or run sweep_trainer.py directly on the trading machine).

GEX feed dependency
───────────────────
    SweepClassifierService subscribes to gex:snapshot:stream.  This channel
    is published by GEXBotPoller in data-pipeline.py.  For GEX features to
    be available the Linux server must be running AND Redis must be reachable
    from the trading machine (shared Redis or LAN).

    If gex:snapshot:stream has never published a message the service will
    classify using rule-based signals with gex_regime="unknown" and
    reduced confidence.  All other features remain fully functional.

Rule-based classifier signals (8 votes)
────────────────────────────────────────
    1. GEX regime     — positive GEX → sweeps more likely to reverse
    2. Wall proximity — price within 5 ticks of a major GEX wall
    3. Through wall   — price passed through a wall → directional signal
    4. CVD agreement  — CVD direction matches price move
    5. CVD pre-trend  — CVD was building before trigger
    6. Buy/sell ratio — lopsided volume in prior minute
    7. DOM imbalance  — bid/ask depth imbalance at trigger
    8. OFI            — order flow imbalance in last 1 second
    Confidence = votes / 8, capped at 0.90 for rule-based.

Outcome labelling (retroactive, run by sweep_trainer.py)
──────────────────────────────────────────────────────────
   After T+60 seconds, compare price_t60 to trigger price:
     sweep:       price returned within REVERT_TICKS (default 5) of trigger
     directional: price extended >= EXTEND_TICKS (default 8) beyond trigger
     ambiguous:   neither condition met

   Run nightly after market close:
       python -m src.ml.sweep_feature_extractor
       python -m src.ml.sweep_trainer
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import pickle
import time
from collections import deque
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import duckdb
import redis.asyncio as aioredis
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REDIS_URL                    = os.getenv("REDIS_URL",                     "redis://localhost:6379/0")
SWEEP_DB_PATH                = os.getenv("SWEEP_DB_PATH",                 "data/fast_moves.db")
SWEEP_TRIGGER_TICKS          = float(os.getenv("SWEEP_TRIGGER_TICKS",     "10"))
SWEEP_TRIGGER_WINDOW_SECONDS = int(os.getenv("SWEEP_TRIGGER_WINDOW_SECONDS", "30"))
SWEEP_COOLDOWN_SECONDS       = int(os.getenv("SWEEP_COOLDOWN_SECONDS",    "60"))
SWEEP_MODEL_PATH             = Path(os.getenv("SWEEP_MODEL_PATH",         "src/ml/sweep_model.pkl"))
DOM_SYMBOL                   = os.getenv("SC_DOM_SYMBOL",                 "MNQ")
MNQ_TICK_VALUE               = float(os.getenv("MNQ_TICK_VALUE",          "0.25"))  # price per tick

# GEX suffix for NQ
GEX_SUFFIX = "NQ_NDX"

# ---------------------------------------------------------------------------
# DuckDB schema
# ---------------------------------------------------------------------------
_CREATE_FAST_MOVES_SQL = """
CREATE TABLE IF NOT EXISTS fast_moves (
    id                     INTEGER PRIMARY KEY,
    ts_ms                  BIGINT  NOT NULL,
    symbol                 VARCHAR NOT NULL,
    trigger_price          DOUBLE,
    price_at_window_start  DOUBLE,
    trigger_ticks          DOUBLE,
    trigger_window_seconds INTEGER,
    direction              VARCHAR,      -- 'up' | 'down'
    -- GEX features
    gex_regime             VARCHAR,      -- 'positive' | 'negative' | 'unknown'
    net_gex_abs            DOUBLE,
    dist_to_pos_wall_ticks DOUBLE,
    dist_to_neg_wall_ticks DOUBLE,
    at_wall                BOOLEAN,
    through_wall           BOOLEAN,
    maxchange_proximity    DOUBLE,
    -- Order flow
    cvd_during_move        DOUBLE,
    cvd_sign_agrees        BOOLEAN,
    cvd_1min_prior         DOUBLE,
    buy_sell_ratio_1min    DOUBLE,
    -- DOM state
    bid_depth_5            DOUBLE,
    ask_depth_5            DOUBLE,
    imbalance_ratio        DOUBLE,
    ofi_1s                 DOUBLE,
    -- Session
    time_of_day_minutes    INTEGER,
    -- Classification
    classification         VARCHAR,      -- 'sweep' | 'directional'
    confidence             DOUBLE,
    model_version          VARCHAR,
    rule_signals           VARCHAR,      -- JSON summary of rule votes
    -- Outcome (filled retroactively by sweep_trainer.py)
    outcome                VARCHAR,      -- 'sweep' | 'directional' | 'ambiguous'
    outcome_price_t60      DOUBLE,
    outcome_move_ticks     DOUBLE,
    labeled_at             TIMESTAMP
)
"""

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class SweepFeatures(BaseModel):
    ts_ms:                  int
    symbol:                 str
    trigger_price:          float
    price_at_window_start:  float
    trigger_ticks:          float
    trigger_window_seconds: int
    direction:              str          # 'up' | 'down'
    gex_regime:             str = "unknown"
    net_gex_abs:            Optional[float] = None
    dist_to_pos_wall_ticks: Optional[float] = None
    dist_to_neg_wall_ticks: Optional[float] = None
    at_wall:                bool = False
    through_wall:           bool = False
    maxchange_proximity:    Optional[float] = None
    cvd_during_move:        Optional[float] = None
    cvd_sign_agrees:        Optional[bool]  = None
    cvd_1min_prior:         Optional[float] = None
    buy_sell_ratio_1min:    Optional[float] = None
    bid_depth_5:            Optional[float] = None
    ask_depth_5:            Optional[float] = None
    imbalance_ratio:        Optional[float] = None
    ofi_1s:                 Optional[float] = None
    time_of_day_minutes:    int = 0


class SweepAlert(BaseModel):
    ts_ms:          int
    symbol:         str
    classification: str     # 'sweep' | 'directional'
    confidence:     float = Field(ge=0.0, le=1.0)
    direction:      str
    trigger_price:  float
    trigger_ticks:  float
    gex_regime:     str = "unknown"
    at_wall:        bool = False
    through_wall:   bool = False
    model_version:  str = "rules-v1"
    danger:         bool = False   # True when directional + high confidence
    danger_level:   int  = 0      # 0-3 (passed to position_monitor)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_rule_based(f: SweepFeatures) -> Tuple[str, float, Dict[str, Any]]:
    """Rule-based sweep classifier.  Returns (outcome, confidence, signals)."""
    sweep_votes = 0
    dir_votes   = 0
    signals: Dict[str, Any] = {}

    # 1. GEX regime: positive = market maker pinning → sweeps mean-revert
    if f.gex_regime == "positive":
        sweep_votes += 1
        signals["gex_pinning"] = True
    elif f.gex_regime == "negative":
        dir_votes += 1
        signals["gex_explosive"] = True

    # 2. Wall proximity: move terminating at a wall suggests sweep bounce
    if f.at_wall:
        sweep_votes += 2
        signals["at_wall"] = True
    if f.through_wall:
        dir_votes += 3           # wall breach is the strongest directional signal
        signals["through_wall"] = True

    # 3. CVD agreement with price direction
    if f.cvd_during_move is not None and f.cvd_sign_agrees is not None:
        if f.cvd_sign_agrees:
            dir_votes   += 2     # price AND delta both moved same way = conviction
            signals["cvd_agrees"] = True
        else:
            sweep_votes += 2     # price moved but delta disagreed = stop run
            signals["cvd_disagrees"] = True

    # 4. DOM imbalance: strong defending side at the extreme = sweep
    if f.imbalance_ratio is not None:
        if f.direction == "down" and f.imbalance_ratio > 0.60:
            sweep_votes += 1     # bids dominate at the low → bounce
            signals["bid_wall_holding"] = True
        elif f.direction == "up" and f.imbalance_ratio < 0.40:
            sweep_votes += 1     # asks dominate at the high → fade
            signals["ask_wall_holding"] = True

    # 5. Pre-move CVD trend: if CVD was already trending in move direction,
    #    it suggests the move is continuation, not just a liquidity grab
    if f.cvd_1min_prior is not None:
        if f.direction == "up"   and f.cvd_1min_prior > 100:
            dir_votes   += 1
            signals["cvd_trending_up"] = True
        elif f.direction == "down" and f.cvd_1min_prior < -100:
            dir_votes   += 1
            signals["cvd_trending_down"] = True

    total = sweep_votes + dir_votes
    if total == 0:
        return ("sweep", 0.50, signals)   # no signal → slight sweep lean (range regime)

    dir_prob = dir_votes / total
    outcome  = "directional" if dir_prob >= 0.5 else "sweep"
    confidence = max(dir_prob, 1.0 - dir_prob)
    # Clamp: rule-based tops out at 0.90 to reflect model uncertainty
    confidence = min(confidence, 0.90)
    signals["sweep_votes"] = sweep_votes
    signals["dir_votes"]   = dir_votes
    return (outcome, round(confidence, 3), signals)


def classify_ml(
    model: Any,
    f: SweepFeatures,
) -> Tuple[str, float]:
    """Run the trained XGBoost/sklearn model."""
    import numpy as np
    features = [
        1 if f.gex_regime == "positive" else (0 if f.gex_regime == "negative" else 0.5),
        f.net_gex_abs or 0.0,
        f.dist_to_pos_wall_ticks or 999.0,
        f.dist_to_neg_wall_ticks or 999.0,
        1.0 if f.at_wall else 0.0,
        1.0 if f.through_wall else 0.0,
        f.cvd_during_move or 0.0,
        1.0 if f.cvd_sign_agrees else 0.0,
        f.cvd_1min_prior or 0.0,
        f.buy_sell_ratio_1min or 1.0,
        f.bid_depth_5 or 0.0,
        f.ask_depth_5 or 0.0,
        f.imbalance_ratio or 0.5,
        f.ofi_1s or 0.0,
        float(f.time_of_day_minutes),
        f.trigger_ticks,
        1.0 if f.direction == "up" else 0.0,
    ]
    X = np.array([features])
    proba = model.predict_proba(X)[0]
    # Assume class 0 = sweep, class 1 = directional
    dir_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
    outcome  = "directional" if dir_prob >= 0.5 else "sweep"
    return (outcome, round(dir_prob if outcome == "directional" else 1 - dir_prob, 3))


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class SweepClassifierService:
    def __init__(self, redis_client: Optional[aioredis.Redis] = None) -> None:
        self._redis   = redis_client
        self._symbol  = DOM_SYMBOL
        self._running = False

        # Price history: deque of (ts_ms, price)
        self._prices: Deque[Tuple[int, float]] = deque(maxlen=5000)
        # CVD at trigger start (set when fast move begins)
        self._cvd_at_trigger_start: Optional[float] = None
        self._last_cvd_1min: Optional[float] = None
        self._last_cvd_5min: Optional[float] = None
        self._last_dom: Optional[Dict[str, Any]] = None
        self._last_flow: Optional[Dict[str, Any]] = None
        self._last_trigger_ms: int = 0

        # GEX state (from gex:snapshot:stream)
        self._gex: Dict[str, Any] = {}

        # ML model (loaded once at startup if available)
        self._model: Optional[Any] = None
        self._model_version: str = "rules-v1"
        self._load_model()

        # DuckDB (opened lazily)
        self._db_path = SWEEP_DB_PATH

    def _load_model(self) -> None:
        if SWEEP_MODEL_PATH.exists():
            try:
                with open(SWEEP_MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
                self._model_version = "xgb-v1"
                LOGGER.info("Loaded sweep model from %s", SWEEP_MODEL_PATH)
            except Exception:
                LOGGER.warning("Failed to load sweep model — using rules", exc_info=True)

    def _ensure_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(duckdb.connect(self._db_path)) as conn:
            conn.execute(_CREATE_FAST_MOVES_SQL)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._redis is None:
            self._redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
        self._running = True
        self._ensure_db()
        LOGGER.info("SweepClassifierService starting for symbol=%s", self._symbol)
        await asyncio.gather(
            self._dom_subscriber(),
            self._cvd_subscriber(),
            self._gex_subscriber(),
        )

    async def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    async def _dom_subscriber(self) -> None:
        sub = self._redis.pubsub()
        await sub.subscribe(f"market:dom:{self._symbol}")
        async for msg in sub.listen():
            if not self._running:
                break
            if msg["type"] != "message":
                continue
            try:
                data = json.loads(msg["data"])
            except (json.JSONDecodeError, TypeError):
                continue
            self._last_dom = data
            price = data.get("price")
            ts_ms = data.get("ts_ms", int(time.time() * 1000))
            if price:
                self._prices.append((ts_ms, float(price)))
                await self._check_trigger(ts_ms, float(price))

    async def _cvd_subscriber(self) -> None:
        sub = self._redis.pubsub()
        await sub.subscribe(f"market:cvd:{self._symbol}")
        async for msg in sub.listen():
            if not self._running:
                break
            if msg["type"] != "message":
                continue
            try:
                data = json.loads(msg["data"])
            except (json.JSONDecodeError, TypeError):
                continue
            self._last_flow = data
            self._last_cvd_1min = data.get("cvd_1min")
            self._last_cvd_5min = data.get("cvd_5min")

    async def _gex_subscriber(self) -> None:
        sub = self._redis.pubsub()
        await sub.subscribe("gex:snapshot:stream")
        async for msg in sub.listen():
            if not self._running:
                break
            if msg["type"] != "message":
                continue
            try:
                data = json.loads(msg["data"])
                # Accept snapshots for NQ/NDX family
                if GEX_SUFFIX in str(data.get("suffix", data.get("ticker", ""))):
                    self._gex = data
            except (json.JSONDecodeError, TypeError):
                continue

    # ------------------------------------------------------------------
    # Fast-move detection
    # ------------------------------------------------------------------

    async def _check_trigger(self, now_ms: int, current_price: float) -> None:
        if not self._prices:
            return

        # Cooldown: don't fire again for SWEEP_COOLDOWN_SECONDS after last trigger
        if (now_ms - self._last_trigger_ms) < SWEEP_COOLDOWN_SECONDS * 1000:
            return

        window_start_ms = now_ms - SWEEP_TRIGGER_WINDOW_SECONDS * 1000
        old_price: Optional[float] = None
        for ts, px in self._prices:
            if ts >= window_start_ms:
                old_price = px
                break  # first price within the window (oldest)

        if old_price is None:
            return

        ticks = abs(current_price - old_price) / MNQ_TICK_VALUE
        if ticks < SWEEP_TRIGGER_TICKS:
            return

        self._last_trigger_ms = now_ms
        direction = "up" if current_price > old_price else "down"
        LOGGER.info(
            "SweepClassifier: fast move detected — %.2f → %.2f  %.1f ticks  %s",
            old_price, current_price, ticks, direction,
        )

        features = self._extract_features(
            ts_ms=now_ms,
            trigger_price=current_price,
            price_at_window_start=old_price,
            ticks=ticks,
            direction=direction,
        )
        alert = self._classify(features)
        await self._publish_alert(alert)
        await self._persist_trigger(features, alert)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(
        self,
        ts_ms: int,
        trigger_price: float,
        price_at_window_start: float,
        ticks: float,
        direction: str,
    ) -> SweepFeatures:
        gex = self._gex

        # GEX regime
        net_gex   = gex.get("net_gex") or gex.get("net_gex_at_alert")
        gex_regime = "unknown"
        net_gex_abs: Optional[float] = None
        if net_gex is not None:
            net_gex_abs = abs(float(net_gex))
            gex_regime = "positive" if float(net_gex) > 0 else "negative"

        # Wall proximity
        pos_wall = gex.get("major_pos_vol") or gex.get("call_wall")
        neg_wall = gex.get("major_neg_vol") or gex.get("put_wall")
        dist_pos: Optional[float] = None
        dist_neg: Optional[float] = None
        if pos_wall:
            dist_pos = abs(trigger_price - float(pos_wall)) / MNQ_TICK_VALUE
        if neg_wall:
            dist_neg = abs(trigger_price - float(neg_wall)) / MNQ_TICK_VALUE

        at_wall    = (dist_pos is not None and dist_pos <= 3.0) or \
                     (dist_neg is not None and dist_neg <= 3.0)
        through_wall = False
        if pos_wall and neg_wall:
            lo = min(float(pos_wall), float(neg_wall))
            hi = max(float(pos_wall), float(neg_wall))
            # Did the move cross a wall?  Price passed it during the window.
            if direction == "up" and price_at_window_start < hi <= trigger_price:
                through_wall = True
            if direction == "down" and price_at_window_start > lo >= trigger_price:
                through_wall = True

        # CVD change during move
        cvd_now    = self._last_cvd_1min
        cvd_during: Optional[float] = None
        cvd_sign_agrees: Optional[bool] = None
        if cvd_now is not None and self._cvd_at_trigger_start is not None:
            cvd_during = cvd_now - self._cvd_at_trigger_start
            cvd_sign_agrees = (direction == "up" and cvd_during > 0) or \
                              (direction == "down" and cvd_during < 0)
        # Reset stored CVD for next trigger
        self._cvd_at_trigger_start = cvd_now

        # Buy/sell ratio 1min
        bsr: Optional[float] = None
        if self._last_flow:
            buy  = self._last_flow.get("buy_vol_1min", 0) or 0
            sell = self._last_flow.get("sell_vol_1min", 0) or 0
            bsr  = buy / sell if sell > 0 else None

        # DOM state
        dom = self._last_dom or {}

        # Session time
        dt_utc = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        tod_minutes = dt_utc.hour * 60 + dt_utc.minute

        return SweepFeatures(
            ts_ms=ts_ms,
            symbol=self._symbol,
            trigger_price=trigger_price,
            price_at_window_start=price_at_window_start,
            trigger_ticks=round(ticks, 1),
            trigger_window_seconds=SWEEP_TRIGGER_WINDOW_SECONDS,
            direction=direction,
            gex_regime=gex_regime,
            net_gex_abs=net_gex_abs,
            dist_to_pos_wall_ticks=round(dist_pos, 1) if dist_pos is not None else None,
            dist_to_neg_wall_ticks=round(dist_neg, 1) if dist_neg is not None else None,
            at_wall=at_wall,
            through_wall=through_wall,
            cvd_during_move=cvd_during,
            cvd_sign_agrees=cvd_sign_agrees,
            cvd_1min_prior=self._last_cvd_1min,
            buy_sell_ratio_1min=round(bsr, 3) if bsr is not None else None,
            bid_depth_5=dom.get("bid_depth_5"),
            ask_depth_5=dom.get("ask_depth_5"),
            imbalance_ratio=dom.get("imbalance_ratio"),
            ofi_1s=dom.get("ofi_1s"),
            time_of_day_minutes=tod_minutes,
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(self, f: SweepFeatures) -> SweepAlert:
        if self._model is not None:
            classification, confidence = classify_ml(self._model, f)
            signals: Dict[str, Any] = {}
        else:
            classification, confidence, signals = classify_rule_based(f)

        # Danger flag: directional with confidence above level-2 threshold
        danger_confidence_threshold = float(os.getenv("SWEEP_DANGER_CONFIDENCE", "0.78"))
        danger = (
            classification == "directional"
            and confidence >= danger_confidence_threshold
        )
        danger_level = 0
        if classification == "directional":
            if confidence >= float(os.getenv("SWEEP_CRITICAL_CONFIDENCE", "0.85")):
                danger_level = 3
            elif confidence >= float(os.getenv("SWEEP_DANGER_CONFIDENCE", "0.78")):
                danger_level = 2
            elif confidence >= float(os.getenv("SWEEP_WARNING_CONFIDENCE", "0.65")):
                danger_level = 1

        return SweepAlert(
            ts_ms=f.ts_ms,
            symbol=f.symbol,
            classification=classification,
            confidence=confidence,
            direction=f.direction,
            trigger_price=f.trigger_price,
            trigger_ticks=f.trigger_ticks,
            gex_regime=f.gex_regime,
            at_wall=f.at_wall,
            through_wall=f.through_wall,
            model_version=self._model_version,
            danger=danger,
            danger_level=danger_level,
        )

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    async def _publish_alert(self, alert: SweepAlert) -> None:
        channel = f"sweep:alert:{alert.symbol}"
        payload = alert.model_dump()
        try:
            await self._redis.publish(channel, json.dumps(payload))
            LOGGER.info(
                "SweepAlert: %s  conf=%.2f  danger=%s  gex=%s  wall=%s  pierce=%s",
                alert.classification.upper(),
                alert.confidence,
                alert.danger,
                alert.gex_regime,
                alert.at_wall,
                alert.through_wall,
            )
        except Exception:
            LOGGER.exception("Failed to publish sweep alert")

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    async def _persist_trigger(self, f: SweepFeatures, alert: SweepAlert) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_fast_move, f, alert)

    def _write_fast_move(self, f: SweepFeatures, alert: SweepAlert) -> None:
        try:
            with closing(duckdb.connect(self._db_path)) as conn:
                conn.execute("""
                    INSERT INTO fast_moves (
                        ts_ms, symbol, trigger_price, price_at_window_start,
                        trigger_ticks, trigger_window_seconds, direction,
                        gex_regime, net_gex_abs,
                        dist_to_pos_wall_ticks, dist_to_neg_wall_ticks,
                        at_wall, through_wall, maxchange_proximity,
                        cvd_during_move, cvd_sign_agrees, cvd_1min_prior,
                        buy_sell_ratio_1min,
                        bid_depth_5, ask_depth_5, imbalance_ratio, ofi_1s,
                        time_of_day_minutes,
                        classification, confidence, model_version, rule_signals
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, [
                    f.ts_ms, f.symbol, f.trigger_price, f.price_at_window_start,
                    f.trigger_ticks, f.trigger_window_seconds, f.direction,
                    f.gex_regime, f.net_gex_abs,
                    f.dist_to_pos_wall_ticks, f.dist_to_neg_wall_ticks,
                    f.at_wall, f.through_wall, f.maxchange_proximity,
                    f.cvd_during_move, f.cvd_sign_agrees, f.cvd_1min_prior,
                    f.buy_sell_ratio_1min,
                    f.bid_depth_5, f.ask_depth_5, f.imbalance_ratio, f.ofi_1s,
                    f.time_of_day_minutes,
                    alert.classification, alert.confidence, alert.model_version,
                    json.dumps({}),
                ])
        except Exception:
            LOGGER.exception("Failed to persist fast_move row")
