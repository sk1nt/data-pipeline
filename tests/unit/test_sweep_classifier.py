"""tests/unit/test_sweep_classifier.py — unit tests for the rule-based classifier
and feature extraction helpers.  No Redis, no DuckDB, no file I/O required.
"""

from __future__ import annotations

import pytest

from src.services.sweep_classifier_service import (
    SweepFeatures,
    classify_rule_based,
)


def _features(**kwargs) -> SweepFeatures:
    """Build a SweepFeatures with sensible defaults; override via kwargs."""
    defaults = dict(
        ts_ms=1_700_000_000_000,
        symbol="MNQ",
        trigger_price=20000.0,
        price_at_window_start=19990.0,
        trigger_ticks=10.0,
        trigger_window_seconds=30,
        direction="up",
        gex_regime="positive",
        net_gex_abs=5.0,
        dist_to_pos_wall_ticks=None,
        dist_to_neg_wall_ticks=None,
        at_wall=False,
        through_wall=False,
        cvd_during_move=None,
        cvd_sign_agrees=None,
        cvd_1min_prior=None,
        buy_sell_ratio_1min=None,
        bid_depth_5=None,
        ask_depth_5=None,
        imbalance_ratio=None,
        ofi_1s=None,
        time_of_day_minutes=600,
    )
    defaults.update(kwargs)
    return SweepFeatures(**defaults)


# ---------------------------------------------------------------------------
# Outcome label tests
# ---------------------------------------------------------------------------

class TestClassifyRuleBased:

    def test_positive_gex_leans_sweep(self):
        f = _features(gex_regime="positive")
        outcome, conf, signals = classify_rule_based(f)
        assert outcome == "sweep"
        assert "gex_pinning" in signals

    def test_negative_gex_leans_directional(self):
        f = _features(gex_regime="negative")
        outcome, conf, signals = classify_rule_based(f)
        assert outcome == "directional"
        assert "gex_explosive" in signals

    def test_wall_pierce_overrides_positive_gex(self):
        """Piercing a wall (through_wall=True) should beat positive GEX pinning."""
        f = _features(gex_regime="positive", through_wall=True)
        outcome, conf, signals = classify_rule_based(f)
        assert outcome == "directional", "Wall breach must override GEX pinning"
        assert "through_wall" in signals

    def test_at_wall_pushes_sweep(self):
        f = _features(gex_regime="unknown", at_wall=True)
        outcome, conf, signals = classify_rule_based(f)
        assert outcome == "sweep"
        assert "at_wall" in signals

    def test_cvd_agreement_pushes_directional(self):
        f = _features(
            gex_regime="unknown",
            direction="up",
            cvd_during_move=500.0,
            cvd_sign_agrees=True,
        )
        outcome, conf, signals = classify_rule_based(f)
        assert outcome == "directional"
        assert "cvd_agrees" in signals

    def test_cvd_disagreement_pushes_sweep(self):
        """Price up but delta was negative → stop-run / sweep."""
        f = _features(
            gex_regime="unknown",
            direction="up",
            cvd_during_move=-300.0,
            cvd_sign_agrees=False,
        )
        outcome, conf, signals = classify_rule_based(f)
        assert outcome == "sweep"
        assert "cvd_disagrees" in signals

    def test_bid_wall_at_low_pushes_sweep(self):
        """Move down but bids dominate at the extreme → absorption, expect bounce."""
        f = _features(
            gex_regime="unknown",
            direction="down",
            imbalance_ratio=0.70,
        )
        outcome, conf, signals = classify_rule_based(f)
        assert outcome == "sweep"
        assert "bid_wall_holding" in signals

    def test_confidence_bounded_by_90_pct(self):
        """Rules should not claim more than 90% confidence."""
        f = _features(
            gex_regime="negative",
            through_wall=True,
            cvd_during_move=800.0,
            cvd_sign_agrees=True,
            direction="up",
        )
        _, conf, _ = classify_rule_based(f)
        assert conf <= 0.90

    def test_no_signals_returns_sweep_at_50_pct(self):
        """With no information, default is slight sweep lean at 50%."""
        f = _features(gex_regime="unknown")
        outcome, conf, signals = classify_rule_based(f)
        assert outcome == "sweep"
        assert conf == 0.50

    def test_strong_directional_scenario(self):
        """Wall breach + explosive GEX + CVD agrees → high-confidence directional."""
        f = _features(
            gex_regime="negative",
            through_wall=True,
            cvd_during_move=1200.0,
            cvd_sign_agrees=True,
            direction="up",
            cvd_1min_prior=400.0,
        )
        outcome, conf, signals = classify_rule_based(f)
        assert outcome == "directional"
        assert conf > 0.75

    def test_strong_sweep_scenario(self):
        """Positive GEX + at wall + CVD disagrees → high-confidence sweep."""
        f = _features(
            gex_regime="positive",
            at_wall=True,
            cvd_during_move=-500.0,
            cvd_sign_agrees=False,
            direction="up",
            imbalance_ratio=0.65,
        )
        outcome, conf, signals = classify_rule_based(f)
        assert outcome == "sweep"
        assert conf > 0.70


# ---------------------------------------------------------------------------
# Danger level derivation (via alert construction)
# ---------------------------------------------------------------------------

class TestDangerLevel:
    """Test that the classifier correctly sets danger and danger_level on alerts."""

    def _alert_from_features(self, **kwargs):
        from src.services.sweep_classifier_service import SweepClassifierService
        svc = SweepClassifierService.__new__(SweepClassifierService)
        svc._model = None
        svc._model_version = "rules-v1"
        f = _features(**kwargs)
        return svc._classify(f)

    def test_sweep_alert_has_no_danger(self):
        alert = self._alert_from_features(
            gex_regime="positive",
            at_wall=True,
            cvd_sign_agrees=False,
        )
        assert alert.classification == "sweep"
        assert alert.danger is False
        assert alert.danger_level == 0

    def test_low_confidence_directional_is_level_1_or_0(self):
        # Just barely directional, low confidence
        alert = self._alert_from_features(gex_regime="negative")
        # Single negative GEX vote gives ~0.5 conf — may not reach level 1
        assert alert.danger_level in (0, 1)

    def test_high_confidence_directional_is_danger(self):
        alert = self._alert_from_features(
            gex_regime="negative",
            through_wall=True,
            cvd_sign_agrees=True,
            cvd_during_move=800.0,
            direction="up",
        )
        assert alert.classification == "directional"
        assert alert.danger is True
        assert alert.danger_level >= 2
