"""sweep_feature_extractor.py — retroactively labels fast_moves rows with ground truth.

After T+60 seconds from each trigger, compare the outcome price to the trigger
price and classify:
  - sweep:       price returned within REVERT_TICKS of trigger price by T+60
  - directional: price extended >= EXTEND_TICKS beyond trigger price by T+60
  - ambiguous:   neither condition met

Also backfills outcome_price_t60 from the MNQ tick parquet.

Run as a nightly job or after market close:
    python -m src.ml.sweep_feature_extractor
"""

from __future__ import annotations

import logging
import os
from contextlib import closing
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import duckdb

LOGGER = logging.getLogger(__name__)

SWEEP_DB_PATH  = os.getenv("SWEEP_DB_PATH",  "data/fast_moves.db")
TICK_DIR       = os.getenv("TICK_PARQUET_DIR", "data/parquet/tick")
MNQ_TICK_VALUE = float(os.getenv("MNQ_TICK_VALUE", "0.25"))

# Labelling thresholds
REVERT_TICKS  = float(os.getenv("SWEEP_LABEL_REVERT_TICKS",  "5"))   # reversion = sweep
EXTEND_TICKS  = float(os.getenv("SWEEP_LABEL_EXTEND_TICKS",  "8"))   # extension = directional


def _price_at_t60(ts_ms: int, symbol: str) -> Optional[float]:
    """Return the first tick price at or after ts_ms + 60s from tick parquet."""
    t60_ms   = ts_ms + 60 * 1000
    event_dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    tick_file = Path(TICK_DIR) / symbol / f"{event_dt.strftime('%Y%m%d')}.parquet"
    if not tick_file.exists():
        return None
    query = f"""
        SELECT price FROM read_parquet({str(tick_file)!r})
        WHERE timestamp_ms BETWEEN {t60_ms} AND {t60_ms + 10_000}
        ORDER BY timestamp_ms
        LIMIT 1
    """
    try:
        row = duckdb.query(query).fetchone()
        return float(row[0]) if row else None
    except Exception:
        LOGGER.debug("t60 price query failed for %s at %d", symbol, ts_ms, exc_info=True)
        return None


def label_outcome(
    trigger_price: float,
    direction: str,
    price_t60: float,
) -> str:
    move_ticks = (price_t60 - trigger_price) / MNQ_TICK_VALUE
    # Positive move_ticks = price went up
    if direction == "up":
        if move_ticks >= EXTEND_TICKS:
            return "directional"  # kept going up — directional
        if move_ticks <= -REVERT_TICKS:
            return "sweep"        # reversed below entry — sweep confirmed
    else:  # direction == "down"
        if move_ticks <= -EXTEND_TICKS:
            return "directional"  # kept going down — directional
        if move_ticks >= REVERT_TICKS:
            return "sweep"        # reversed above entry — sweep confirmed
    return "ambiguous"


def backfill_outcomes(min_age_seconds: int = 120) -> int:
    """Label all unlabelled rows older than min_age_seconds.  Returns count updated."""
    if not Path(SWEEP_DB_PATH).exists():
        LOGGER.warning("fast_moves.db not found at %s", SWEEP_DB_PATH)
        return 0

    cutoff_ms = int((datetime.now(timezone.utc).timestamp() - min_age_seconds) * 1000)
    updated = 0

    with closing(duckdb.connect(SWEEP_DB_PATH)) as conn:
        rows = conn.execute("""
            SELECT id, ts_ms, symbol, trigger_price, direction
            FROM fast_moves
            WHERE outcome IS NULL
              AND ts_ms < ?
            ORDER BY ts_ms
        """, [cutoff_ms]).fetchall()

        for row_id, ts_ms, symbol, trigger_price, direction in rows:
            p60 = _price_at_t60(ts_ms, symbol)
            if p60 is None:
                continue
            move_ticks = abs(p60 - trigger_price) / MNQ_TICK_VALUE * (
                1 if ((direction == "up" and p60 > trigger_price) or
                      (direction == "down" and p60 < trigger_price)) else -1
            )
            outcome = label_outcome(trigger_price, direction, p60)
            conn.execute("""
                UPDATE fast_moves
                SET outcome = ?, outcome_price_t60 = ?, outcome_move_ticks = ?,
                    labeled_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, [outcome, p60, round(move_ticks, 2), row_id])
            updated += 1

    LOGGER.info("sweep_feature_extractor: labelled %d rows", updated)
    return updated


def load_training_data(min_samples: int = 200):
    """Return (X, y) numpy arrays for training.  Returns (None, None) if < min_samples."""
    if not Path(SWEEP_DB_PATH).exists():
        return None, None
    try:
        import numpy as np
        with closing(duckdb.connect(SWEEP_DB_PATH, read_only=True)) as conn:
            rows = conn.execute("""
                SELECT
                    CASE gex_regime WHEN 'positive' THEN 1 WHEN 'negative' THEN -1 ELSE 0 END,
                    COALESCE(net_gex_abs, 0),
                    COALESCE(dist_to_pos_wall_ticks, 999),
                    COALESCE(dist_to_neg_wall_ticks, 999),
                    CASE WHEN at_wall    THEN 1.0 ELSE 0.0 END,
                    CASE WHEN through_wall THEN 1.0 ELSE 0.0 END,
                    COALESCE(cvd_during_move, 0),
                    CASE WHEN cvd_sign_agrees THEN 1.0 ELSE 0.0 END,
                    COALESCE(cvd_1min_prior, 0),
                    COALESCE(buy_sell_ratio_1min, 1),
                    COALESCE(bid_depth_5, 0),
                    COALESCE(ask_depth_5, 0),
                    COALESCE(imbalance_ratio, 0.5),
                    COALESCE(ofi_1s, 0),
                    COALESCE(time_of_day_minutes, 0),
                    trigger_ticks,
                    CASE WHEN direction = 'up' THEN 1.0 ELSE 0.0 END,
                    -- label: 1 = directional, 0 = sweep
                    CASE outcome WHEN 'directional' THEN 1 WHEN 'sweep' THEN 0 ELSE NULL END
                FROM fast_moves
                WHERE outcome IN ('sweep', 'directional')
            """).fetchall()

        if len(rows) < min_samples:
            LOGGER.info("Only %d labelled rows — need %d for training", len(rows), min_samples)
            return None, None

        arr = np.array(rows, dtype=float)
        X   = arr[:, :-1]
        y   = arr[:,  -1].astype(int)
        LOGGER.info("Loaded %d labelled training samples", len(X))
        return X, y
    except Exception:
        LOGGER.exception("Failed to load training data")
        return None, None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    n = backfill_outcomes()
    print(f"Labelled {n} rows")
