"""Unit tests for MarketMoverAnalyzer and its pure-function helpers."""

from __future__ import annotations

import os
import tempfile
from contextlib import closing
from datetime import datetime, timezone, timedelta

import duckdb
import pytest

from src.services.market_mover_analyzer import (
    MarketMoverAnalyzer,
    _best_price_move,
    _compute_impact_score,
    _infer_sentiment,
)


# ---------------------------------------------------------------------------
# Pure-function helpers
# ---------------------------------------------------------------------------

class TestBestPriceMove:
    def test_selects_largest_absolute_move(self):
        assert _best_price_move(100.0, 100.3, 99.5, 100.1) == pytest.approx(
            ((99.5 - 100.0) / 100.0) * 100, rel=1e-6
        )

    def test_returns_none_when_t0_is_none(self):
        assert _best_price_move(None, 101.0, 102.0, 103.0) is None

    def test_returns_none_when_t0_is_zero(self):
        assert _best_price_move(0.0, 1.0, 2.0, 3.0) is None

    def test_returns_none_when_no_future_prices(self):
        assert _best_price_move(100.0, None, None, None) is None

    def test_positive_move(self):
        result = _best_price_move(5000.0, None, 5050.0, None)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_negative_move(self):
        result = _best_price_move(5000.0, 4950.0, None, None)
        assert result == pytest.approx(-1.0, rel=1e-4)


class TestComputeImpactScore:
    def test_all_none_returns_zero(self):
        assert _compute_impact_score(None, None, None) == 0.0

    def test_large_price_move_dominates(self):
        # 2% move should score close to PRICE_WEIGHT (50 pts)
        score = _compute_impact_score(2.0, None, None)
        assert 30 < score <= 50

    def test_confluence_drives_higher_score(self):
        solo = _compute_impact_score(0.5, None, None)
        combo = _compute_impact_score(0.5, 20.0, 3.0)
        assert combo > solo

    def test_score_capped_at_100(self):
        # Extreme inputs must never exceed 100
        assert _compute_impact_score(100.0, 100.0, 100.0) <= 100.0

    def test_volume_below_one_ignored(self):
        # volume_ratio < 1 should not contribute
        score_no_vol = _compute_impact_score(0.5, None, None)
        score_low_vol = _compute_impact_score(0.5, None, 0.5)
        assert score_no_vol == score_low_vol

    def test_score_is_non_negative(self):
        # Bearish price move should still produce a positive impact score
        score = _compute_impact_score(-1.0, -30.0, 4.0)
        assert score >= 0.0


class TestInferSentiment:
    def test_positive_move_is_bullish(self):
        assert _infer_sentiment(0.5) == "bullish"

    def test_negative_move_is_bearish(self):
        assert _infer_sentiment(-0.5) == "bearish"

    def test_tiny_move_is_neutral(self):
        assert _infer_sentiment(0.01) == "neutral"

    def test_none_is_neutral(self):
        assert _infer_sentiment(None) == "neutral"


# ---------------------------------------------------------------------------
# MarketMoverAnalyzer with temporary DuckDB
# ---------------------------------------------------------------------------

def _make_db(path: str) -> None:
    """Seed a minimal correlation_events.db for testing."""
    with closing(duckdb.connect(path)) as conn:
        conn.execute("""
            CREATE TABLE correlation_events (
                timestamp              TIMESTAMP,
                social_event_id        VARCHAR,
                social_source          VARCHAR,
                social_author          VARCHAR,
                social_text            VARCHAR,
                social_score           INTEGER,
                social_url             VARCHAR,
                alert_type             VARCHAR,
                alert_fired            BOOLEAN,
                signals_triggered      VARCHAR,
                volume_ratio           DOUBLE,
                gex_change_pct         DOUBLE,
                price_change_pct       DOUBLE,
                config_snapshot        VARCHAR,
                created_at             TIMESTAMP,
                realized_impact_score  DOUBLE,
                price_t0               DOUBLE,
                price_t15              DOUBLE,
                price_ticker           VARCHAR,
                is_noise               BOOLEAN
            )
        """)
        # Insert 3 events: 2 recent, 1 old (outside lookback)
        now = datetime.now(timezone.utc)
        rows = [
            (
                (now - timedelta(hours=2)).isoformat(),
                "evt001", "twitter", "@DeItaone",
                "FED SURPRISES MARKETS WITH EMERGENCY RATE CUT",
                9, None, "volume_spike", True, '["volume_spike"]',
                3.2, -22.0, -0.8, None, now.isoformat(),
                None, None, None, None, None,
            ),
            (
                (now - timedelta(hours=6)).isoformat(),
                "evt002", "news_rss", "Google News",
                "Tariffs on China expanded to cover all electronics",
                6, "https://example.com", "price_move", True, '["price_move"]',
                1.1, 5.0, -0.3, None, now.isoformat(),
                None, None, None, None, None,
            ),
            (
                # 40 days ago — should be excluded by 21-day lookback
                (now - timedelta(days=40)).isoformat(),
                "evt003", "twitter", "@WatcherGuru",
                "Breaking: central bank meeting cancelled",
                3, None, None, False, '[]',
                None, None, None, None, now.isoformat(),
                None, None, None, None, None,
            ),
        ]
        conn.executemany("""
            INSERT INTO correlation_events VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
        """, rows)


def _tmp_db_path() -> str:
    """Return a path to a non-existent temp file (DuckDB creates it fresh)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as f:
        path = f.name
    # File is deleted after the 'with' block; return the path so DuckDB creates it
    return path


class TestMarketMoverAnalyzerWithDB:
    def _make_analyzer(self, db_path: str) -> MarketMoverAnalyzer:
        return MarketMoverAnalyzer(
            correlation_db_path=db_path,
            # Point at non-existent parquet dirs so price/GEX lookups return None
            timeseries_parquet_dir="/tmp/nonexistent_ts",
            tick_parquet_dir="/tmp/nonexistent_tick",
            noise_floor=5.0,
        )

    def test_respects_lookback_window(self):
        db_path = _tmp_db_path()
        try:
            _make_db(db_path)
            analyzer = self._make_analyzer(db_path)
            results = analyzer.analyze(lookback_days=21)
            event_ids = [r.social_event_id for r in results]
            assert "evt001" in event_ids
            assert "evt002" in event_ids
            assert "evt003" not in event_ids  # too old
        finally:
            os.unlink(db_path)

    def test_missing_db_returns_empty(self):
        analyzer = MarketMoverAnalyzer(
            correlation_db_path="/tmp/does_not_exist_99999.db",
            timeseries_parquet_dir="/tmp/nonexistent_ts",
            tick_parquet_dir="/tmp/nonexistent_tick",
        )
        assert analyzer.analyze() == []

    def test_events_sorted_by_impact_desc(self):
        db_path = _tmp_db_path()
        try:
            _make_db(db_path)
            analyzer = self._make_analyzer(db_path)
            results = analyzer.analyze(lookback_days=21)
            scores = [r.realized_impact_score for r in results]
            assert scores == sorted(scores, reverse=True)
        finally:
            os.unlink(db_path)

    def test_noise_flagged_but_included_by_default(self):
        """Events without any pre-computed signals and no parquet data → noise.
        Events WITH pre-computed volume_ratio (stored by correlation engine) can
        still be scored as movers even without live parquet data.
        """
        db_path = _tmp_db_path()
        try:
            _make_db(db_path)
            analyzer = self._make_analyzer(db_path)
            results = analyzer.analyze(lookback_days=21)
            # Both events come back (no min_realized_impact filter)
            assert len(results) == 2
            # evt001 has volume_ratio=3.2 pre-computed → NOT noise
            # evt002 has volume_ratio=1.1 (barely above 1, very low impact) + no parquet
            movers = [r for r in results if not r.is_noise]
            noise  = [r for r in results if r.is_noise]
            # At least evt001 (high volume_ratio) should be a mover
            assert any(r.social_event_id == "evt001" for r in movers)
            # Events flagged as noise have a reason
            for r in noise:
                assert r.noise_reason is not None
        finally:
            os.unlink(db_path)

    def test_min_realized_impact_filter(self):
        db_path = _tmp_db_path()
        try:
            _make_db(db_path)
            analyzer = self._make_analyzer(db_path)
            results = analyzer.analyze(lookback_days=21, min_realized_impact=99.0)
            # No event reaches 99/100 without real parquet data
            assert results == []
        finally:
            os.unlink(db_path)

    def test_top_n_respected(self):
        db_path = _tmp_db_path()
        try:
            _make_db(db_path)
            analyzer = self._make_analyzer(db_path)
            results = analyzer.analyze(lookback_days=21, top_n=1)
            assert len(results) == 1
        finally:
            os.unlink(db_path)

    def test_noise_reason_populated(self):
        db_path = _tmp_db_path()
        try:
            _make_db(db_path)
            analyzer = self._make_analyzer(db_path)
            results = analyzer.analyze(lookback_days=21)
            noisy = [r for r in results if r.is_noise]
            for r in noisy:
                assert r.noise_reason is not None and len(r.noise_reason) > 0
        finally:
            os.unlink(db_path)

    def test_summarize_output(self):
        db_path = _tmp_db_path()
        try:
            _make_db(db_path)
            analyzer = self._make_analyzer(db_path)
            results = analyzer.analyze(lookback_days=21)
            summary = MarketMoverAnalyzer.summarize(results)
            assert "Market Movers Summary" in summary
            assert "noise events pruned" in summary
        finally:
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# CorrelationAlertService — backfill + new columns
# ---------------------------------------------------------------------------

class TestCorrelationAlertServiceBackfill:
    def test_backfill_updates_row(self):
        from src.services.correlation_alert_service import CorrelationAlertService

        db_path = _tmp_db_path()
        try:
            svc = CorrelationAlertService(db_path=db_path)
            alert = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "alert_type": "volume_spike",
                "signals_triggered": ["volume_spike"],
                "severity": "medium",
                "message": "test",
                "social_event": {
                    "event_id": "testevt01",
                    "source": "twitter",
                    "author": "@test",
                    "text": "Tariffs raised",
                    "relevance_score": 5,
                    "url": None,
                },
                "market_signals": {
                    "volume_ratio": 2.5,
                    "gex_change_pct": -10.0,
                    "price_change_pct": -0.5,
                },
            }
            svc.log_correlation_event(alert)
            svc.backfill_realized_impact(
                social_event_id="testevt01",
                realized_impact_score=42.7,
                price_t0=5000.0,
                price_t15=4975.0,
                price_ticker="ES_SPX",
                is_noise=False,
            )
            rows = svc.query_events(limit=10)
            assert len(rows) == 1
            assert rows[0]["realized_impact_score"] == pytest.approx(42.7, rel=1e-4)
            assert rows[0]["price_ticker"] == "ES_SPX"
            assert rows[0]["is_noise"] is False
        finally:
            os.unlink(db_path)

    def test_log_with_realized_impact_kwarg(self):
        from src.services.correlation_alert_service import CorrelationAlertService

        db_path = _tmp_db_path()
        try:
            svc = CorrelationAlertService(db_path=db_path)
            alert = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "alert_type": "price_move",
                "signals_triggered": [],
                "severity": "high",
                "message": "big move",
                "social_event": {
                    "event_id": "testevt02",
                    "source": "news_rss",
                    "author": "Reuters",
                    "text": "Fed cuts rates by 75bps",
                    "relevance_score": 9,
                    "url": "https://reuters.com/article/1",
                },
                "market_signals": {},
            }
            svc.log_correlation_event(
                alert,
                alert_fired=True,
                realized_impact_score=77.3,
                price_t0=5100.0,
                price_t15=5165.0,
                price_ticker="ES_SPX",
                is_noise=False,
            )
            rows = svc.query_events(limit=5)
            assert rows[0]["realized_impact_score"] == pytest.approx(77.3, rel=1e-4)
            assert rows[0]["is_noise"] is False
        finally:
            os.unlink(db_path)
