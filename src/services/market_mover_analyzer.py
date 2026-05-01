"""Market Mover Analyzer — retroactively measures the actual price impact that followed
each social/news event stored in correlation_events.db, then ranks and prunes by
realized market impact rather than keyword-score alone.

Design
------
For every social event in the lookback window the analyzer:
1. Locates the matching GEX spot / tick parquet files around the event timestamp.
2. Extracts price windows T0 (event) → T+5 min, T+15 min, T+30 min.
3. Extracts GEX (sum_gex_vol) delta over the same horizon.
4. Extracts volume ratio from tick parquet (optional, symbol must be in tick dir).
5. Combines the three signals into a 0–100 `realized_impact_score`.
6. Flags events below `noise_floor` as noise and exposes the reason.
7. Returns results sorted by realized_impact_score descending.

Parquet layout expected
-----------------------
  data/parquet/timeseries/{YYYY-MM-DD}/flush_*.parquet
      columns: key (str), ts (int64 ms-epoch), value (float64), day (date)
  data/parquet/tick/{SYMBOL}/{YYYYMMDD}.parquet
      columns: symbol, source, timestamp, timestamp_ms (int64 ms), price, size

GEX spot key pattern:  ts:gex:spot:{TICKER}
GEX vol  key pattern:  ts:gex:sum_gex_vol:{TICKER}
"""

from __future__ import annotations

import logging
import os
from contextlib import closing
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import duckdb
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
TIMESERIES_PARQUET_DIR = os.getenv("TIMESERIES_PARQUET_DIR", "data/parquet/timeseries")
TICK_PARQUET_DIR = os.getenv("TICK_PARQUET_DIR", "data/parquet/tick")
CORRELATION_DB_PATH = os.getenv("CORRELATION_DB_PATH", "data/correlation_events.db")

# Instruments whose timeseries parquet keys are checked by default.
# Map: friendly ticker → GEX key suffix
DEFAULT_TICKERS: Dict[str, str] = {
    "ES_SPX": "ES_SPX",
    "SPY":    "SPY",
    "QQQ":    "QQQ",
}

# Symbols available in the tick parquet directory (volume + high-res price)
TICK_SYMBOLS = {"MNQ", "MES", "NQ", "QQQ", "SPY"}

# Measurement windows in seconds
WINDOWS_SECONDS = {
    "t5":  5  * 60,
    "t15": 15 * 60,
    "t30": 30 * 60,
    "t60": 60 * 60,
}

# Impact scoring weights (must sum to 100)
PRICE_WEIGHT = 50   # % price move component
GEX_WEIGHT   = 30   # % GEX shift component
VOL_WEIGHT   = 20   # % volume spike component

# Normalization anchors (1 unit = 50 pts of contribution)
PRICE_HALF_SCORE_PCT = 0.5   # 0.5 % price move → 25 pts
GEX_HALF_SCORE_PCT   = 20.0  # 20 % GEX shift  → 15 pts
VOL_HALF_SCORE_RATIO = 2.0   # 2× volume ratio → 10 pts


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class MarketMoverResult(BaseModel):
    """Ranked social event with measured market impact."""

    social_event_id:      str
    timestamp:            datetime
    social_source:        str
    social_author:        str
    social_text:          str
    social_score:         int = 0
    social_url:           Optional[str] = None
    realized_impact_score: float = Field(
        ge=0.0, le=100.0,
        description="0–100 composite score of actual price/GEX/volume move",
    )
    price_ticker:         str = ""
    # GEX timeseries prices (1-min resolution from parquet)
    price_t0:             Optional[float] = None
    price_t5:             Optional[float] = None
    price_t15:            Optional[float] = None
    price_t30:            Optional[float] = None
    price_t60:            Optional[float] = None
    price_move_pct:       Optional[float] = Field(
        default=None,
        description="Best horizon price move (% from T0)",
    )
    # GEX aggregate signal
    gex_t0:               Optional[float] = None
    gex_t15:              Optional[float] = None
    gex_shift_pct:        Optional[float] = None
    gex_regime:           Optional[str] = None   # above_zero_gamma | below_zero_gamma | unknown
    net_gex_at_alert:     Optional[float] = None
    # Tick-level high-resolution prices (second-resolution from MNQ tick parquet)
    tick_price_t0:        Optional[float] = None
    tick_price_t5:        Optional[float] = None
    tick_price_t15:       Optional[float] = None
    tick_price_t30:       Optional[float] = None
    tick_price_t60:       Optional[float] = None
    tick_move_t5_pct:     Optional[float] = None
    tick_move_t15_pct:    Optional[float] = None
    tick_move_t30_pct:    Optional[float] = None
    tick_move_t60_pct:    Optional[float] = None
    tick_move_direction:  Optional[str] = None   # up | down | flat
    volume_ratio:         Optional[float] = None
    sentiment:            str = "neutral"
    is_noise:             bool = False
    noise_reason:         Optional[str] = None


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class MarketMoverAnalyzer:
    """Load social events from DuckDB, cross-reference parquet market data,
    and rank by realized market impact."""

    def __init__(
        self,
        *,
        correlation_db_path: str = CORRELATION_DB_PATH,
        timeseries_parquet_dir: str = TIMESERIES_PARQUET_DIR,
        tick_parquet_dir: str = TICK_PARQUET_DIR,
        tickers: Optional[Dict[str, str]] = None,
        noise_floor: float = 5.0,
    ) -> None:
        self.db_path = correlation_db_path
        self.ts_dir = timeseries_parquet_dir.rstrip("/")
        self.tick_dir = tick_parquet_dir.rstrip("/")
        self.tickers = tickers or DEFAULT_TICKERS
        self.noise_floor = noise_floor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        lookback_days: int = 21,
        min_realized_impact: float = 0.0,
        top_n: int = 100,
    ) -> List[MarketMoverResult]:
        """Return social events ranked by realized market impact.

        Parameters
        ----------
        lookback_days:
            How many calendar days of history to examine (default 3 weeks).
        min_realized_impact:
            Minimum realized_impact_score to include.  Events below
            *noise_floor* are still returned but flagged as noise.
        top_n:
            Maximum results to return.
        """
        events = self._load_social_events(lookback_days)
        if not events:
            LOGGER.info("No social events found for the last %d days", lookback_days)
            return []

        results: List[MarketMoverResult] = []
        for ev in events:
            result = self._score_event(ev)
            if result.realized_impact_score >= min_realized_impact:
                results.append(result)

        results.sort(key=lambda r: r.realized_impact_score, reverse=True)
        return results[:top_n]

    def analyze_realtime(self, lookback_hours: int = 24, top_n: int = 50) -> List[MarketMoverResult]:
        """Convenience method: last *lookback_hours* of events only."""
        return self.analyze(
            lookback_days=max(1, (lookback_hours + 23) // 24),
            top_n=top_n,
        )

    # ------------------------------------------------------------------
    # Internal: load events from DuckDB
    # ------------------------------------------------------------------

    def _load_social_events(self, lookback_days: int) -> List[Dict[str, Any]]:
        if not os.path.exists(self.db_path):
            LOGGER.warning("Correlation DB not found at %s — no historical events", self.db_path)
            return []
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
        try:
            with closing(duckdb.connect(self.db_path, read_only=True)) as conn:
                rows = conn.execute(
                    """
                    SELECT
                        social_event_id, timestamp, social_source, social_author,
                        social_text, social_score, social_url, alert_type,
                        volume_ratio, gex_change_pct, price_change_pct
                    FROM correlation_events
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    """,
                    [cutoff],
                ).fetchall()
                cols = [
                    "social_event_id", "timestamp", "social_source", "social_author",
                    "social_text", "social_score", "social_url", "alert_type",
                    "volume_ratio", "gex_change_pct", "price_change_pct",
                ]
                return [dict(zip(cols, row)) for row in rows]
        except Exception:
            LOGGER.exception("Failed to load social events from %s", self.db_path)
            return []

    # ------------------------------------------------------------------
    # Internal: score one event
    # ------------------------------------------------------------------

    def _score_event(self, ev: Dict[str, Any]) -> MarketMoverResult:
        ts = ev["timestamp"]
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except Exception:
                ts = datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        ts_ms = int(ts.timestamp() * 1000)

        # Try each tracked ticker until we get price data
        price_ticker = ""
        price_t0 = price_t5 = price_t15 = price_t30 = price_t60 = None
        gex_t0 = gex_t15 = None

        for ticker_key, gex_suffix in self.tickers.items():
            p = self._fetch_price_windows(gex_suffix, ts_ms)
            if p["t0"] is not None:
                price_ticker  = ticker_key
                price_t0  = p["t0"]
                price_t5  = p["t5"]
                price_t15 = p["t15"]
                price_t30 = p["t30"]
                price_t60 = p["t60"]
                g = self._fetch_gex_windows(gex_suffix, ts_ms)
                gex_t0  = g["t0"]
                gex_t15 = g["t15"]
                break

        # High-res tick prices from MNQ parquet
        tick = self._fetch_tick_price_windows(ts_ms)
        tick_price_t0  = tick["t0"]
        tick_price_t5  = tick["t5"]
        tick_price_t15 = tick["t15"]
        tick_price_t30 = tick["t30"]
        tick_price_t60 = tick["t60"]

        # Tick move %s
        def _tick_move(end: Optional[float]) -> Optional[float]:
            if tick_price_t0 and end and tick_price_t0 != 0:
                return round(((end - tick_price_t0) / tick_price_t0) * 100, 3)
            return None

        tick_move_t5_pct  = _tick_move(tick_price_t5)
        tick_move_t15_pct = _tick_move(tick_price_t15)
        tick_move_t30_pct = _tick_move(tick_price_t30)
        tick_move_t60_pct = _tick_move(tick_price_t60)

        # Directional summary using best available T+15 tick or price
        _best_move = tick_move_t15_pct
        tick_move_direction: Optional[str] = None
        if _best_move is not None:
            tick_move_direction = "up" if _best_move > 0.05 else ("down" if _best_move < -0.05 else "flat")

        # GEX regime at alert time
        gex_regime, net_gex_at_alert = self._fetch_gex_regime(list(self.tickers.values())[0], ts_ms)

        # Best price move (largest absolute % across horizons; prefer tick data)
        price_move_pct = _best_price_move(price_t0, price_t5, price_t15, price_t30, price_t60)

        # GEX shift
        gex_shift_pct: Optional[float] = None
        if gex_t0 is not None and gex_t15 is not None and gex_t0 != 0:
            gex_shift_pct = ((gex_t15 - gex_t0) / abs(gex_t0)) * 100

        # Volume ratio from tick parquet (best available symbol)
        volume_ratio = ev.get("volume_ratio")  # pre-computed in correlation engine
        if volume_ratio is None:
            volume_ratio = self._fetch_volume_ratio(ts_ms)

        # Realized impact score (use tick move if available, otherwise GEX timeseries)
        effective_move = tick_move_t15_pct if tick_move_t15_pct is not None else price_move_pct
        realized_impact = _compute_impact_score(effective_move, gex_shift_pct, volume_ratio)

        # Noise classification
        is_noise = False
        noise_reason: Optional[str] = None
        if realized_impact < self.noise_floor:
            is_noise = True
            parts = []
            if effective_move is None:
                parts.append("no price data")
            elif abs(effective_move) < 0.05:
                parts.append(f"price flat ({effective_move:+.2f}%)")
            if volume_ratio is None or volume_ratio < 1.2:
                parts.append("volume normal")
            if gex_shift_pct is None or abs(gex_shift_pct) < 2.0:
                parts.append("GEX unchanged")
            noise_reason = "; ".join(parts) if parts else "low composite impact"

        return MarketMoverResult(
            social_event_id=ev.get("social_event_id", ""),
            timestamp=ts,
            social_source=ev.get("social_source", ""),
            social_author=ev.get("social_author", ""),
            social_text=ev.get("social_text", ""),
            social_score=ev.get("social_score", 0),
            social_url=ev.get("social_url"),
            realized_impact_score=round(realized_impact, 2),
            price_ticker=price_ticker,
            price_t0=price_t0,
            price_t5=price_t5,
            price_t15=price_t15,
            price_t30=price_t30,
            price_t60=price_t60,
            price_move_pct=round(price_move_pct, 3) if price_move_pct is not None else None,
            gex_t0=gex_t0,
            gex_t15=gex_t15,
            gex_shift_pct=round(gex_shift_pct, 2) if gex_shift_pct is not None else None,
            gex_regime=gex_regime,
            net_gex_at_alert=net_gex_at_alert,
            tick_price_t0=tick_price_t0,
            tick_price_t5=tick_price_t5,
            tick_price_t15=tick_price_t15,
            tick_price_t30=tick_price_t30,
            tick_price_t60=tick_price_t60,
            tick_move_t5_pct=tick_move_t5_pct,
            tick_move_t15_pct=tick_move_t15_pct,
            tick_move_t30_pct=tick_move_t30_pct,
            tick_move_t60_pct=tick_move_t60_pct,
            tick_move_direction=tick_move_direction,
            volume_ratio=round(volume_ratio, 2) if volume_ratio is not None else None,
            sentiment=_infer_sentiment(effective_move),
            is_noise=is_noise,
            noise_reason=noise_reason,
        )

    # ------------------------------------------------------------------
    # Internal: parquet queries
    # ------------------------------------------------------------------

    def _fetch_price_windows(
        self, gex_suffix: str, ts_ms: int
    ) -> Dict[str, Optional[float]]:
        """Return spot prices at T0, T+5, T+15, T+30 from GEX timeseries parquet."""
        key = f"ts:gex:spot:{gex_suffix}"
        glob = self._ts_glob_for_ms(ts_ms, extra_days=1)
        if not glob:
            return {"t0": None, "t5": None, "t15": None, "t30": None}

        t5  = ts_ms + WINDOWS_SECONDS["t5"]  * 1_000
        t15 = ts_ms + WINDOWS_SECONDS["t15"] * 1_000
        t30 = ts_ms + WINDOWS_SECONDS["t30"] * 1_000
        t60 = ts_ms + WINDOWS_SECONDS["t60"] * 1_000

        query = f"""
            SELECT
                min(CASE WHEN ts >= {ts_ms} AND ts < {ts_ms + 60_000}
                          THEN value END) AS p_t0,
                min(CASE WHEN ts >= {t5}   AND ts < {t5  + 60_000}
                          THEN value END) AS p_t5,
                min(CASE WHEN ts >= {t15}  AND ts < {t15 + 60_000}
                          THEN value END) AS p_t15,
                min(CASE WHEN ts >= {t30}  AND ts < {t30 + 60_000}
                          THEN value END) AS p_t30,
                min(CASE WHEN ts >= {t60}  AND ts < {t60 + 60_000}
                          THEN value END) AS p_t60
            FROM read_parquet({glob!r}, union_by_name=true)
            WHERE key = {key!r}
              AND ts BETWEEN {ts_ms} AND {t60 + 60_000}
        """
        return self._run_scalar_query(query, ["t0", "t5", "t15", "t30", "t60"])

    def _fetch_gex_windows(
        self, gex_suffix: str, ts_ms: int
    ) -> Dict[str, Optional[float]]:
        """Return sum_gex_vol at T0 and T+15 from GEX timeseries parquet."""
        key = f"ts:gex:sum_gex_vol:{gex_suffix}"
        glob = self._ts_glob_for_ms(ts_ms, extra_days=1)
        if not glob:
            return {"t0": None, "t15": None}

        t15 = ts_ms + WINDOWS_SECONDS["t15"] * 1_000
        query = f"""
            SELECT
                min(CASE WHEN ts >= {ts_ms} AND ts < {ts_ms + 60_000}
                          THEN value END) AS g_t0,
                min(CASE WHEN ts >= {t15}   AND ts < {t15  + 60_000}
                          THEN value END) AS g_t15
            FROM read_parquet({glob!r}, union_by_name=true)
            WHERE key = {key!r}
              AND ts BETWEEN {ts_ms} AND {t15 + 60_000}
        """
        return self._run_scalar_query(query, ["t0", "t15"])

    def _fetch_tick_price_windows(self, ts_ms: int) -> Dict[str, Optional[float]]:
        """Return MNQ tick prices at T0, T+5, T+15, T+30, T+60 from tick parquet."""
        event_dt = datetime.fromtimestamp(ts_ms / 1_000, tz=timezone.utc)
        results: Dict[str, Optional[float]] = {k: None for k in ("t0", "t5", "t15", "t30", "t60")}
        for symbol in ("MNQ", "MES"):
            day_str = event_dt.strftime("%Y%m%d")
            tick_file = f"{self.tick_dir}/{symbol}/{day_str}.parquet"
            if not os.path.exists(tick_file):
                continue
            windows = [
                ("t0",  ts_ms,                              ts_ms + 10_000),
                ("t5",  ts_ms + WINDOWS_SECONDS["t5"]  * 1_000, ts_ms + WINDOWS_SECONDS["t5"]  * 1_000 + 10_000),
                ("t15", ts_ms + WINDOWS_SECONDS["t15"] * 1_000, ts_ms + WINDOWS_SECONDS["t15"] * 1_000 + 10_000),
                ("t30", ts_ms + WINDOWS_SECONDS["t30"] * 1_000, ts_ms + WINDOWS_SECONDS["t30"] * 1_000 + 10_000),
                ("t60", ts_ms + WINDOWS_SECONDS["t60"] * 1_000, ts_ms + WINDOWS_SECONDS["t60"] * 1_000 + 10_000),
            ]
            cases = " ".join(
                f"min(CASE WHEN timestamp_ms BETWEEN {lo} AND {hi} THEN price END) AS p_{k}"
                for k, lo, hi in windows
            )
            t_end = ts_ms + WINDOWS_SECONDS["t60"] * 1_000 + 10_000
            query = f"""
                SELECT {cases}
                FROM read_parquet({tick_file!r})
                WHERE timestamp_ms BETWEEN {ts_ms} AND {t_end}
            """
            try:
                row = duckdb.query(query).fetchone()
                if row:
                    for (k, *_), val in zip(windows, row):
                        if val is not None:
                            results[k] = float(val)
                    if results["t0"] is not None:
                        return results  # found data, stop trying symbols
            except Exception:
                LOGGER.debug("Tick price query failed for %s on %s", symbol, day_str, exc_info=True)
        return results

    def _fetch_gex_regime(
        self, gex_suffix: str, ts_ms: int
    ) -> Tuple[str, Optional[float]]:
        """Return (regime_label, net_gex_value) from GEX timeseries parquet at T0."""
        spot_key     = f"ts:gex:spot:{gex_suffix}"
        net_gex_key  = f"ts:gex:net_gex:{gex_suffix}"
        zero_key     = f"ts:gex:zero_gamma:{gex_suffix}"
        glob = self._ts_glob_for_ms(ts_ms)
        if not glob:
            return ("unknown", None)
        query = f"""
            SELECT
                max(CASE WHEN key = {spot_key!r}    THEN value END) AS spot,
                max(CASE WHEN key = {net_gex_key!r} THEN value END) AS net_gex,
                max(CASE WHEN key = {zero_key!r}    THEN value END) AS zero_gamma
            FROM read_parquet({glob!r}, union_by_name=true)
            WHERE key IN ({spot_key!r}, {net_gex_key!r}, {zero_key!r})
              AND ts BETWEEN {ts_ms - 120_000} AND {ts_ms + 60_000}
        """
        try:
            row = duckdb.query(query).fetchone()
            if row:
                spot, net_gex, zero_gamma = row
                if spot is not None and zero_gamma is not None:
                    regime = "above_zero_gamma" if float(spot) > float(zero_gamma) else "below_zero_gamma"
                    return (regime, float(net_gex) if net_gex is not None else None)
        except Exception:
            LOGGER.debug("GEX regime query failed for %s", gex_suffix, exc_info=True)
        return ("unknown", None)

    def _fetch_volume_ratio(self, ts_ms: int) -> Optional[float]:
        """Compute volume ratio from tick parquet (uses MNQ, falls back to SPY)."""
        event_dt = datetime.fromtimestamp(ts_ms / 1_000, tz=timezone.utc)
        day_str = event_dt.strftime("%Y%m%d")

        for symbol in ("MNQ", "MES", "SPY"):
            tick_file = f"{self.tick_dir}/{symbol}/{day_str}.parquet"
            if not os.path.exists(tick_file):
                continue
            try:
                t_start = ts_ms - 5 * 60 * 1_000   # 5-min pre-event baseline
                t_end   = ts_ms + 5 * 60 * 1_000   # 5-min post-event window

                result = duckdb.query(f"""
                    WITH pre AS (
                        SELECT sum(size) AS vol
                        FROM read_parquet({tick_file!r})
                        WHERE timestamp_ms BETWEEN {t_start - 20*60*1000}
                                               AND {t_start}
                    ),
                    post AS (
                        SELECT sum(size) AS vol
                        FROM read_parquet({tick_file!r})
                        WHERE timestamp_ms BETWEEN {ts_ms} AND {t_end}
                    )
                    SELECT post.vol, pre.vol FROM post, pre
                """).fetchone()

                if result and result[1] and result[1] > 0:
                    # Normalize: pre covers 20 min, post covers 5 min → scale pre to 5-min rate
                    pre_rate = result[1] / 4.0  # 20-min window → 4 × 5-min periods
                    if pre_rate > 0:
                        return result[0] / pre_rate
            except Exception:
                LOGGER.debug("Volume ratio query failed for %s on %s", symbol, day_str, exc_info=True)
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ts_glob_for_ms(self, ts_ms: int, extra_days: int = 1) -> Optional[str]:
        """Build a DuckDB glob string that covers the date of ts_ms plus extra days."""
        event_dt = datetime.fromtimestamp(ts_ms / 1_000, tz=timezone.utc)
        dates = [event_dt + timedelta(days=i) for i in range(extra_days + 1)]
        patterns = [
            f"{self.ts_dir}/{d.strftime('%Y-%m-%d')}/*.parquet"
            for d in dates
        ]
        # Only include patterns where the directory actually exists
        valid = [
            p for p in patterns
            if os.path.isdir(os.path.dirname(p))
        ]
        if not valid:
            return None
        if len(valid) == 1:
            return valid[0]
        # DuckDB list-of-globs syntax
        return "[" + ", ".join(f"'{p}'" for p in valid) + "]"

    def _run_scalar_query(
        self, query: str, keys: List[str]
    ) -> Dict[str, Optional[float]]:
        try:
            row = duckdb.query(query).fetchone()
            if row:
                return {k: (float(v) if v is not None else None) for k, v in zip(keys, row)}
        except Exception:
            LOGGER.debug("Parquet scalar query failed", exc_info=True)
        return {k: None for k in keys}

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def summarize(results: List[MarketMoverResult]) -> str:
        """Return a compact human-readable summary of the top movers."""
        movers = [r for r in results if not r.is_noise]
        noise  = [r for r in results if r.is_noise]
        lines = [
            f"Market Movers Summary  ({len(movers)} movers / {len(noise)} noise events pruned)",
            "-" * 72,
        ]
        for i, r in enumerate(movers[:25], 1):
            pm   = f"{r.price_move_pct:+.2f}%" if r.price_move_pct is not None else "n/a"
            gex  = f"{r.gex_shift_pct:+.1f}%" if r.gex_shift_pct is not None else "n/a"
            vol  = f"{r.volume_ratio:.1f}×"    if r.volume_ratio   is not None else "n/a"
            ts_s = r.timestamp.strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"#{i:>3} [{r.realized_impact_score:>5.1f}]  {ts_s}  "
                f"{r.social_author:<20}  price={pm:>7}  GEX={gex:>7}  vol={vol:>5}\n"
                f"      {r.social_text[:90]}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pure functions (easily unit-testable)
# ---------------------------------------------------------------------------

def _best_price_move(
    t0: Optional[float],
    t5: Optional[float],
    t15: Optional[float],
    t30: Optional[float],
    t60: Optional[float] = None,
) -> Optional[float]:
    """Return the largest absolute % move from T0 across all horizons."""
    if t0 is None or t0 == 0:
        return None
    candidates: List[float] = []
    for p in (t5, t15, t30, t60):
        if p is not None:
            candidates.append(((p - t0) / t0) * 100)
    if not candidates:
        return None
    return max(candidates, key=abs)


def _compute_impact_score(
    price_move_pct: Optional[float],
    gex_shift_pct: Optional[float],
    volume_ratio: Optional[float],
) -> float:
    """Combine price / GEX / volume into a 0–100 realized impact score.

    Each component is sigmoid-normalised so extreme moves asymptotically
    approach their maximum weight without ever exceeding it.
    """
    def _sigmoid(x: float, half: float, weight: float) -> float:
        # maps 0 → 0, half → weight/2, ∞ → weight
        import math
        return weight * (1 - math.exp(-abs(x) / half))

    score = 0.0
    if price_move_pct is not None:
        score += _sigmoid(price_move_pct, PRICE_HALF_SCORE_PCT, PRICE_WEIGHT)
    if gex_shift_pct is not None:
        score += _sigmoid(gex_shift_pct, GEX_HALF_SCORE_PCT, GEX_WEIGHT)
    if volume_ratio is not None and volume_ratio > 1.0:
        score += _sigmoid(volume_ratio - 1.0, VOL_HALF_SCORE_RATIO - 1.0, VOL_WEIGHT)

    return min(score, 100.0)


def _infer_sentiment(price_move_pct: Optional[float]) -> str:
    if price_move_pct is None:
        return "neutral"
    if price_move_pct > 0.05:
        return "bullish"
    if price_move_pct < -0.05:
        return "bearish"
    return "neutral"
