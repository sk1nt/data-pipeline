# Spec 002 — Sweep Classifier & Sierra Chart DOM Bridge

> **Implementation:** `~/projects/data-trading` (trading machine).  
> **This repo:** GEX ingest + `gex:snapshot:stream` + dashboard `/ws/sweep` relay only.  
> See `docs/SWEEP_MOVED_TO_DATA_TRADING.md`.

**Status**: Implementation ready  
**Priority**: P0 — directly supports live manual trading  
**Language decisions**: C++ (ACSIL study), Python (bridge service + ML), with notes on when to diverge

---

## Problem Statement

When MNQ/NQ makes a fast move of 8–20 ticks, the trader needs to know within 30–60 seconds whether it is:

- **Liquidity sweep**: Aggressive orders hit stops, defending side absorbs → mean reversion expected. Fade or hold.
- **Directional break**: Same initial signature, but defending side fails → move extends. Get out or flip.

The current setup has GEX levels (walls, zero gamma) and unsigned tick data. It is missing:
1. True bid/ask trade classification (Sierra Chart has this natively)
2. Full DOM depth (TastyTrade depth stopped Nov 2025, was only 5 levels)
3. Real-time OFI (Order Flow Imbalance) from DOM delta
4. An automated danger detector tied to actual position state

---

## Architecture Overview

```
Sierra Chart (Windows / WSL)
    └── ACSIL Study: dom_trade_bridge.cpp
            ├── Writes dom_snapshot.json    (250ms, full DOM)
            ├── Writes trade_flow.json      (250ms, CVD + recent trades)
            └── Polls  danger_trigger.json  (100ms, emergency flatten signal)
                        │
                 [WSL /mnt/c/SierraChart/Data/]
                        │
Python: sierra_dom_bridge_service.py
    ├── Reads dom_snapshot.json + trade_flow.json (inotify or 100ms poll)
    ├── Publishes to Redis: market:dom:MNQ, market:cvd:MNQ
    └── Subscribes to Redis: sweep:danger:MNQ → writes danger_trigger.json

Python: sweep_classifier_service.py
    ├── Subscribes: market:dom:MNQ, market:cvd:MNQ, gex:snapshot:stream
    ├── Detects fast-move trigger (configurable ticks in configurable window)
    ├── Extracts features (see below)
    ├── Runs trained model (XGBoost, falls back to rule-based when model not ready)
    ├── Publishes: sweep:alert:MNQ (classification + confidence + danger flag)
    └── Writes labeled rows to DuckDB: fast_moves table

Python: position_monitor_service.py  
    ├── Subscribes: sweep:alert:MNQ, market:dom:MNQ
    ├── Reads current position from SC position file (or TT API)
    └── Publishes: sweep:danger:MNQ  IF:
            - Position open AND
            - Sweep alert says DIRECTIONAL AND confidence >= threshold AND
            - Position is in the WRONG direction AND
            - Loss is already beyond WARNING_TICKS (NOT immediate — graduated)

ML: sweep_model/ 
    ├── sweep_feature_extractor.py   (DuckDB backfill against tick + DOM history)
    ├── sweep_trainer.py             (XGBoost, trains nightly when >= 200 samples)
    └── sweep_model.pkl              (loaded by sweep_classifier_service at startup)

Dashboard: intelligence.html (new screen, spec 003)
    ├── Sweep probability gauge (live, from Redis)
    ├── DOM heatmap (bid/ask depth visualized as heat)
    ├── CVD mini-chart (1 min rolling)
    └── GEX walls overlaid on price ladder
```

---

## Data: Sierra Chart ACSIL Study

### What SC has that is unavailable elsewhere

| Data | Description | ACSIL API |
|---|---|---|
| Full DOM | All price levels bid/ask (up to 100+) | `sc.GetLevel2Pointer()` |
| True trade side | Each trade flagged as bid or ask aggressor | `sc.Trades[i].TradeCondition` |
| Native CVD | Running cumulative delta already computed | Computed from flagged trades |
| Footprint | Volume at each price level per bar | `sc.VolumeAtPriceForBars` |
| OFI | Bid/ask size changes between snapshots | Computed from DOM delta |

### File outputs (all written to `SC_DATA_DIR`, readable by WSL as `/mnt/c/SierraChart/Data/`)

**`dom_snapshot.json`** — written every 250ms
```json
{
  "ts_ms": 1777525977138,
  "symbol": "MNQM26",
  "price": 27557.50,
  "bids": [[27557.25, 38], [27557.00, 22], ...],
  "asks": [[27557.50, 45], [27557.75, 12], ...],
  "bid_levels": 100,
  "ask_levels": 100,
  "ofi_1s": 234,
  "ofi_5s": -412,
  "bid_depth_1": 38,
  "ask_depth_1": 45,
  "bid_depth_5": 134,
  "ask_depth_5": 122,
  "bid_depth_10": 310,
  "ask_depth_10": 295,
  "bid_depth_20": 580,
  "ask_depth_20": 541,
  "imbalance_ratio": 0.62
}
```

**`trade_flow.json`** — written every 250ms, rolling 500-trade buffer
```json
{
  "ts_ms": 1777525977138,
  "symbol": "MNQM26",
  "cvd_1min": -342,
  "cvd_5min": -1204,
  "cvd_15min": -876,
  "cvd_running_day": -2341,
  "buy_vol_1min": 1203,
  "sell_vol_1min": 1545,
  "last_trades": [
    {"ts_ms": 1777525977100, "price": 27557.50, "size": 3, "side": "ask"},
    {"ts_ms": 1777525977050, "price": 27557.25, "size": 1, "side": "bid"}
  ]
}
```

**`danger_trigger.json`** — polled by SC at 100ms; Python writes this to trigger emergency stop
```json
{
  "ts_ms": 1777525977138,
  "action": "flatten",
  "reason": "sweep_classifier:directional_break:confidence=0.84",
  "severity": "critical",
  "consumed": false
}
```

SC study: when `action == "flatten"` and `consumed == false` → calls `sc.FlattenAndCancelAllOrders()`, writes `consumed: true` back.

---

## Danger/Emergency Stop — Design

**This is graduated, not a hair trigger.**

```
Level 0: WATCH
  Condition: Fast move detected, model running
  Action: Sweep probability gauge updates on screen
  SC action: None

Level 1: WARNING  (sweep:alert published, confidence >= 0.65)
  Condition: Directional probability > 65%, position open in wrong direction,
             unrealized loss >= WARNING_TICKS (default: 15 ticks)
  Action: Audible alert (sc.PlaySound), dashboard flashes yellow
  SC action: None — trader is in control

Level 2: DANGER   (confidence >= 0.78, loss >= DANGER_TICKS default: 25 ticks)
  Condition: High-confidence directional break + position losing
  Action: Dashboard flashes red, persistent alert tone  
  SC action: Reduce position by 50% (cancel working orders, submit market reduce)

Level 3: CRITICAL  (confidence >= 0.85, loss >= CRITICAL_TICKS default: 35 ticks)
  Condition: Very high confidence + loss approaching max threshold AND
             no manual acknowledgment of Level 2 within 10 seconds
  Action: Write danger_trigger.json → SC flattens everything
  SC action: sc.FlattenAndCancelAllOrders()
```

**Key safety properties:**
- Each level requires BOTH: model confidence AND actual realized loss threshold
- Level 3 requires Level 2 to have fired first with no human response
- All thresholds are configurable in `.env`, not hardcoded
- A "suppress for N minutes" button on the dashboard lets the trader override
- Manual trading mode: by default Level 3 fires only if loss > CRITICAL_TICKS regardless of model (pure loss protection, same as Position_PNL_Manager but triggered by flow data)

---

## Sweep Classifier — Features

All measured at the moment a fast-move trigger fires:

### GEX context (from Redis gex:snapshot:stream)
| Feature | Description |
|---|---|
| `gex_regime` | sign(net_gex) → positive = pinning, negative = explosive |
| `net_gex_abs` | Absolute GEX level — how much force |
| `dist_to_pos_wall_ticks` | Ticks from current price to major_pos_vol level |
| `dist_to_neg_wall_ticks` | Ticks from current price to major_neg_vol level |
| `at_wall` | Price within 3 ticks of any major wall |
| `through_wall` | Did the move pierce a wall? |
| `maxchange_proximity` | Distance to nearest maxchange strike |

### Order flow (from trade_flow.json)
| Feature | Description |
|---|---|
| `cvd_during_move` | CVD change during the fast-move window |
| `cvd_sign_agrees` | CVD direction agrees with price direction (True = aggressive) |
| `cvd_1min_prior` | CVD in the 60s before the move started |
| `buy_sell_ratio_1min` | buy_vol / sell_vol in 1 min window |
| `move_velocity` | Ticks per second during the trigger window |
| `volume_during_move` | Total size traded during the move |
| `volume_ratio` | volume_during_move / 20-min baseline |

### DOM state (from dom_snapshot.json)
| Feature | Description |
|---|---|
| `ofi_at_trigger` | OFI (order flow imbalance) at moment of trigger |
| `bid_ask_imbalance` | (bid_depth_10 - ask_depth_10) / (bid_depth_10 + ask_depth_10) |
| `depth_thinning` | Was DOM thinning on the side being hit before the move? |
| `iceberg_detected` | Ask/bid at one level was repeatedly refreshed (absorption) |

### Session context
| Feature | Description |
|---|---|
| `hour_of_day` | 0–23 |
| `minutes_since_open` | Minutes since 09:30 ET |
| `session` | pre_market / open_30 / mid_day / close_30 / after_hours |
| `day_of_week` | 0=Mon … 4=Fri |

### Label (backfilled 3 minutes after trigger)
- `outcome`: `sweep` (price returned within 8 ticks in 3 min) | `directional` (extended 15+ ticks) | `unclear` (discard)
- `max_adverse_excursion_ticks`: max drawdown from trigger point
- `max_favorable_ticks`: max extension in direction of original move

---

## DuckDB Schema: `fast_moves` table

```sql
CREATE TABLE fast_moves (
    move_id              VARCHAR PRIMARY KEY,   -- SHA(symbol+ts_ms)
    ts_ms                BIGINT NOT NULL,
    symbol               VARCHAR NOT NULL,
    trigger_price        DOUBLE,
    move_direction       VARCHAR,               -- up | down
    move_ticks           DOUBLE,
    move_seconds         DOUBLE,
    -- GEX
    gex_regime           VARCHAR,
    net_gex_abs          DOUBLE,
    dist_to_pos_wall     DOUBLE,
    dist_to_neg_wall     DOUBLE,
    at_wall              BOOLEAN,
    through_wall         BOOLEAN,
    maxchange_proximity  DOUBLE,
    -- Order flow
    cvd_during_move      DOUBLE,
    cvd_sign_agrees      BOOLEAN,
    cvd_1min_prior       DOUBLE,
    buy_sell_ratio_1min  DOUBLE,
    move_velocity        DOUBLE,
    volume_ratio         DOUBLE,
    -- DOM
    ofi_at_trigger       DOUBLE,
    bid_ask_imbalance    DOUBLE,
    depth_thinning       BOOLEAN,
    -- Session
    hour_of_day          INTEGER,
    session              VARCHAR,
    day_of_week          INTEGER,
    -- Predicted (at trigger time)
    predicted_outcome    VARCHAR,
    prediction_confidence DOUBLE,
    model_version        VARCHAR,
    -- Actual outcome (backfilled ~3 min later)
    actual_outcome       VARCHAR,
    mae_ticks            DOUBLE,
    mfe_ticks            DOUBLE,
    -- Position at trigger
    position_size        INTEGER,
    position_direction   VARCHAR,
    unrealized_pnl_ticks DOUBLE,
    danger_fired         BOOLEAN,
    danger_level         INTEGER,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Language Decisions

| Component | Language | Rationale |
|---|---|---|
| ACSIL bridge study | **C++** | Required by Sierra Chart |
| Bridge file reader / Redis publisher | **Python** (asyncio) | 250ms update cycle, I/O bound — asyncio is fine |
| Sweep feature extraction | **Python** | DuckDB queries, pandas/polars |
| ML training | **Python** | XGBoost/LightGBM — standard ML stack |
| Real-time classification | **Python** | Single model.predict() call <1ms, no GIL issue |
| Position monitor / danger trigger | **Python** | Logic is simple, latency not critical (<100ms) |
| Dashboard | **HTML/JS** | Extends existing gex_monitor.html pattern |

**No need to switch languages.** The bottleneck is not CPU — it's data quality and sample accumulation. Python handles 250ms cycles without strain. If in the future the sweep classifier runs at 10ms frequency (tick-by-tick), that's the time to consider Rust for the hot path. Not now.

---

## Path to Automation

This same stack transitions to fully automated with two additions:
1. A `signal_executor.py` that converts high-confidence predictions into order placement via TastyTrade/Schwab API (already in codebase)
2. A `risk_guard.py` wrapper that enforces position limits, daily loss cap, and time-of-day rules before any order is sent

The sweep classifier naturally becomes an entry signal: when confidence > 0.82 for a sweep (mean reversion expected), that's a fade entry. When confidence > 0.82 for directional, that's either a momentum entry or a get-out signal depending on existing position.

The model transitions from "second opinion" to "primary signal" as the win rate is validated over 500+ labeled examples.

---

## Implementation Order

| Phase | Task | File | Dependency |
|---|---|---|---|
| 1 | ACSIL bridge study | `data/dom_trade_bridge.cpp` | None — compile in SC |
| 2 | Bridge reader service | `src/services/sierra_dom_bridge_service.py` | Phase 1 running |
| 3 | Fast-moves DB schema | migration in `correlation_alert_service.py` | None |
| 4 | Feature extractor | `src/services/sweep_classifier_service.py` | Phase 2 |
| 5 | Historical backfill | `scripts/backfill_fast_moves.py` | Phase 4 |
| 6 | Model trainer | `src/ml/sweep_trainer.py` | Phase 5 (≥200 samples) |
| 7 | Position monitor / danger | `src/services/position_monitor_service.py` | Phase 2 + 4 |
| 8 | Dashboard panel | `frontend/src/intelligence.html` | Phase 4 |

**Phase 1–4 can be done now.** Phase 5 runs overnight. Phase 6 unlocks when 200 labeled events exist. Phase 7 should be tested in paper mode first.

---

## Testing Strategy

### Unit tests (no live data required)
- `tests/unit/test_sweep_features.py` — feature extraction with synthetic tick sequences
- `tests/unit/test_dom_bridge_reader.py` — JSON parsing, Redis publish mocking
- `tests/unit/test_danger_levels.py` — level escalation logic with mock positions

### Integration tests (uses historical parquet)
- `tests/integration/test_sweep_backfill.py` — run detector over 5 known trading days, check trigger count is reasonable
- `tests/integration/test_dom_file_watch.py` — write mock JSON files, verify Redis messages

### Regression tests (run after every feature add)
- `tests/regression/test_sweep_no_false_flattens.py` — replay last 30 days, verify danger_level==3 fires ≤ N times per day in paper mode
- Labeled golden set of 20 manually-verified moves (10 sweeps, 10 directional) — classifier must get ≥ 80% on these before any model version goes live

### Paper mode
- All danger actions (Level 2 reduce, Level 3 flatten) write to a `danger_log.json` with `dry_run: true` until `SWEEP_LIVE_MODE=true` is set in `.env`
- Position monitor reads from `dry_run` first, simulates outcomes, builds track record

---

## Configuration (`.env` additions)

```bash
# Sierra Chart bridge paths
SC_DATA_DIR=/mnt/c/SierraChart/Data
SC_DOM_SNAPSHOT_PATH=/mnt/c/SierraChart/Data/dom_snapshot.json
SC_TRADE_FLOW_PATH=/mnt/c/SierraChart/Data/trade_flow.json
SC_DANGER_TRIGGER_PATH=/mnt/c/SierraChart/Data/danger_trigger.json

# Sweep classifier
SWEEP_TRIGGER_TICKS=10
SWEEP_TRIGGER_WINDOW_SECONDS=30
SWEEP_LIVE_MODE=false            # must be explicitly set to enable Level 2/3 actions

# Danger thresholds
SWEEP_WARNING_TICKS=15
SWEEP_DANGER_TICKS=25
SWEEP_CRITICAL_TICKS=35
SWEEP_WARNING_CONFIDENCE=0.65
SWEEP_DANGER_CONFIDENCE=0.78
SWEEP_CRITICAL_CONFIDENCE=0.85
SWEEP_LEVEL2_ACKNOWLEDGE_SECONDS=10  # how long trader has to respond before Level 3

# DOM settings
SC_DOM_LEVELS=100
SC_DOM_UPDATE_MS=250
```
