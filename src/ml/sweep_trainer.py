"""sweep_trainer.py — trains an XGBoost binary classifier on labelled fast_moves.

Run nightly after backfill_outcomes():
    python -m src.ml.sweep_trainer

Produces: src/ml/sweep_model.pkl  (loaded by sweep_classifier_service at startup)

Requirements: xgboost, scikit-learn  (already in requirements.txt under [ml] extra)
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

LOGGER = logging.getLogger(__name__)

MODEL_OUT = Path(os.getenv("SWEEP_MODEL_PATH", "src/ml/sweep_model.pkl"))
MIN_SAMPLES = int(os.getenv("SWEEP_MIN_TRAIN_SAMPLES", "200"))

FEATURE_NAMES = [
    "gex_regime_enc",
    "net_gex_abs",
    "dist_to_pos_wall_ticks",
    "dist_to_neg_wall_ticks",
    "at_wall",
    "through_wall",
    "cvd_during_move",
    "cvd_sign_agrees",
    "cvd_1min_prior",
    "buy_sell_ratio_1min",
    "bid_depth_5",
    "ask_depth_5",
    "imbalance_ratio",
    "ofi_1s",
    "time_of_day_minutes",
    "trigger_ticks",
    "direction_up",
]


def train() -> bool:
    """Train model. Returns True if a new model was saved, False otherwise."""
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except ImportError:
        LOGGER.error("xgboost / scikit-learn not installed — skipping training")
        return False

    from src.ml.sweep_feature_extractor import load_training_data

    X, y = load_training_data(min_samples=MIN_SAMPLES)
    if X is None:
        LOGGER.info("sweep_trainer: not enough data — skipping")
        return False

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )

    # Cross-validate first so we know quality before saving
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    LOGGER.info("sweep_trainer: CV AUC = %.3f ± %.3f", aucs.mean(), aucs.std())

    if aucs.mean() < 0.55:
        LOGGER.warning("sweep_trainer: model AUC below threshold (%.3f) — not saving", aucs.mean())
        return False

    model.fit(X, y)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)

    LOGGER.info(
        "sweep_trainer: saved model to %s  (n=%d, AUC=%.3f)",
        MODEL_OUT, len(X), aucs.mean(),
    )

    # Log feature importances
    try:
        importances = model.feature_importances_
        ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
        LOGGER.info("Feature importances:")
        for name, imp in ranked[:10]:
            LOGGER.info("  %-35s %.4f", name, imp)
    except Exception:
        pass

    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    success = train()
    print("Model saved." if success else "No model saved (insufficient data or low AUC).")
