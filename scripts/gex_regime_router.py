#!/usr/bin/env python3
"""
GEX regime classifier + router.

Usage:
  python scripts/gex_regime_router.py --parquet ml/output/mnq_2025-11-11.parquet --out models/gex_regime.joblib

What it does:
  - Trains a tiny classifier on net GEX and related fields to label each bar as +GEX (pinning/mean reversion) or -GEX (momentum/expansion).
  - Provides a simple router that picks between two model predictions based on the regime.

Notes:
  - Designed to be fast and low-variance; falls back to LogisticRegression if LightGBM is unavailable.
  - Uses cost-aware thresholding on the regime classifier (defaults to 0.5 but expose --threshold to sweep).
"""

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Normalize casing
    if "net_gex" not in df.columns and "netGex" in df.columns:
        df = df.rename(columns={"netGex": "net_gex"})
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    feats["net_gex"] = df.get("net_gex", 0.0)
    feats["gex_zero"] = df.get("gex_zero", 0.0)
    feats["sum_gex_vol"] = df.get("sum_gex_vol", 0.0)
    feats["major_pos_vol"] = df.get("major_pos_vol", 0.0)
    feats["major_neg_vol"] = df.get("major_neg_vol", 0.0)
    # Distances to flip/walls if price columns exist
    price = df.get("Close") if "Close" in df.columns else df.get("close")
    if price is not None and not price.isna().all():
        feats["dist_to_zero"] = price - feats["gex_zero"]
    else:
        feats["dist_to_zero"] = 0.0
    feats = feats.fillna(0.0)
    return feats


def make_labels(
    df: pd.DataFrame, pos_thresh: float = 0.0, neg_thresh: float = 0.0
) -> np.ndarray:
    """Label +1 for strong positive GEX, 0 for negative/neutral."""
    net = df.get("net_gex", pd.Series(0, index=df.index)).fillna(0.0)
    labels = (net > pos_thresh).astype(int)
    # Optionally mask low-magnitude regimes by setting to 0
    labels[(net < neg_thresh)] = 0
    return labels.values


def train_regime_classifier(X: pd.DataFrame, y: np.ndarray):
    try:
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=200,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    except Exception:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    return model


def route_predictions(
    regime_probs: np.ndarray,
    preds_pos: np.ndarray,
    preds_neg: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Select prediction from pos-model when regime prob > threshold else neg-model."""
    mask = regime_probs > threshold
    return np.where(mask, preds_pos, preds_neg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--parquet", required=True, help="Input parquet with net_gex and price columns."
    )
    p.add_argument(
        "--out", default="models/gex_regime.joblib", help="Output path for classifier."
    )
    p.add_argument(
        "--threshold", type=float, default=0.5, help="Decision threshold for routing."
    )
    args = p.parse_args()

    df = load_parquet(Path(args.parquet))
    X = make_features(df)
    y = make_labels(df)
    model = train_regime_classifier(X, y)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": list(X.columns), "threshold": args.threshold},
        out_path,
    )
    print(f"Saved regime classifier to {out_path} (features={len(X.columns)})")


if __name__ == "__main__":
    main()
