"""IVR (Implied Volatility Rank) computation service.

Reads implied_volatility from option_trades DuckDB table, aggregates
per-symbol daily IV snapshots, and computes IVR = (current_IV - IV_low_252d)
/ (IV_high_252d - IV_low_252d). Caches results to Redis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb

LOGGER = logging.getLogger(__name__)

IVR_LOOKBACK_DAYS = 252
IVR_CACHE_TTL = 86400  # 24 hours
IVR_HIGH_THRESHOLD = 80.0
IVR_LOW_THRESHOLD = 20.0


try:
    from ..config import settings as _settings
    _default_db = Path(_settings.data_path) / "gex_data.db"
except Exception:
    _default_db = Path("data/gex_data.db")


@dataclass
class IVRServiceSettings:
    db_path: Path = _default_db
    option_trades_db: Path = _default_db
    iv_history_table: str = "iv_history"
    redis_prefix: str = "ivr:"
    lookback_days: int = IVR_LOOKBACK_DAYS


class IVRService:
    """Compute and cache IVR per symbol from persisted option trade IV data."""

    def __init__(
        self,
        config: Optional[IVRServiceSettings] = None,
        redis_client=None,
    ) -> None:
        self.settings = config or IVRServiceSettings()
        self._redis = redis_client
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create iv_history table if it does not exist."""
        conn = duckdb.connect(str(self.settings.option_trades_db))
        try:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.settings.iv_history_table} (
                    symbol VARCHAR,
                    trade_date DATE,
                    avg_iv DOUBLE,
                    min_iv DOUBLE,
                    max_iv DOUBLE,
                    trade_count INTEGER,
                    PRIMARY KEY (symbol, trade_date)
                )
                """
            )
        finally:
            conn.close()

    def aggregate_daily_iv(self) -> int:
        """Aggregate IV from option_trades into iv_history.

        Returns number of symbol-day rows written.
        """
        conn = duckdb.connect(str(self.settings.option_trades_db))
        try:
            try:
                cols = {r[0] for r in conn.execute("DESCRIBE option_trades").fetchall()}
            except duckdb.CatalogException:
                LOGGER.debug("option_trades table does not exist yet")
                return 0
            if "implied_volatility" not in cols:
                LOGGER.warning("option_trades table has no implied_volatility column")
                return 0

            rows = conn.execute(
                """
                SELECT
                    ticker AS symbol,
                    CAST(received_at AS DATE) AS trade_date,
                    AVG(implied_volatility) AS avg_iv,
                    MIN(implied_volatility) AS min_iv,
                    MAX(implied_volatility) AS max_iv,
                    COUNT(*) AS trade_count
                FROM option_trades
                WHERE implied_volatility IS NOT NULL
                  AND implied_volatility > 0
                GROUP BY ticker, CAST(received_at AS DATE)
                """
            ).fetchall()

            if not rows:
                return 0

            for symbol, trade_date, avg_iv, min_iv, max_iv, trade_count in rows:
                conn.execute(
                    f"""
                    INSERT OR REPLACE INTO {self.settings.iv_history_table}
                    (symbol, trade_date, avg_iv, min_iv, max_iv, trade_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [symbol, trade_date, avg_iv, min_iv, max_iv, int(trade_count)],
                )

            LOGGER.info("Aggregated IV history: %d symbol-day rows", len(rows))
            return len(rows)
        finally:
            conn.close()

    def compute_ivr(self, symbol: str) -> Dict[str, Any]:
        """Compute IVR for a symbol from iv_history.

        IVR = (current_IV - IV_low_252d) / (IV_high_252d - IV_low_252d)
        Scaled to 0-100.
        """
        conn = duckdb.connect(str(self.settings.option_trades_db))
        try:
            rows = conn.execute(
                """
                SELECT trade_date, avg_iv
                FROM {tbl}
                WHERE symbol = ?
                ORDER BY trade_date DESC
                LIMIT {days}
                """.format(tbl=self.settings.iv_history_table, days=self.settings.lookback_days),
                [symbol],
            ).fetchall()

            if not rows:
                return {
                    "ivr": None,
                    "iv_percentile": None,
                    "current_iv": None,
                    "iv_252_high": None,
                    "iv_252_low": None,
                    "trade_days": 0,
                }

            ivs = [r[1] for r in rows if r[1] is not None and r[1] > 0]
            if not ivs:
                return {
                    "ivr": None,
                    "iv_percentile": None,
                    "current_iv": None,
                    "iv_252_high": None,
                    "iv_252_low": None,
                    "trade_days": 0,
                }

            current_iv = ivs[0]
            iv_high = max(ivs)
            iv_low = min(ivs)

            if iv_high == iv_low:
                ivr = 50.0
            else:
                ivr = ((current_iv - iv_low) / (iv_high - iv_low)) * 100.0
                ivr = max(0.0, min(100.0, ivr))

            iv_percentile = (sum(1 for iv in ivs if iv <= current_iv) / len(ivs)) * 100.0

            result = {
                "ivr": round(ivr, 2),
                "iv_percentile": round(iv_percentile, 2),
                "current_iv": round(current_iv, 4),
                "iv_252_high": round(iv_high, 4),
                "iv_252_low": round(iv_low, 4),
                "trade_days": len(ivs),
            }

            if self._redis:
                import json
                self._redis.client.setex(
                    f"{self.settings.redis_prefix}{symbol}",
                    IVR_CACHE_TTL,
                    json.dumps(result),
                )
                if ivr >= IVR_HIGH_THRESHOLD:
                    self._redis.client.publish(
                        f"ivr:alert:{symbol}",
                        json.dumps({"symbol": symbol, "type": "iv_high", "ivr": ivr, "threshold": IVR_HIGH_THRESHOLD}),
                    )
                elif ivr <= IVR_LOW_THRESHOLD:
                    self._redis.client.publish(
                        f"ivr:alert:{symbol}",
                        json.dumps({"symbol": symbol, "type": "iv_low", "ivr": ivr, "threshold": IVR_LOW_THRESHOLD}),
                    )

            return result
        finally:
            conn.close()

    def compute_ivr_batch(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Compute IVR for all symbols (or a subset)."""
        conn = duckdb.connect(str(self.settings.option_trades_db))
        try:
            if symbols is None:
                rows = conn.execute(
                    f"SELECT DISTINCT symbol FROM {self.settings.iv_history_table}"
                ).fetchall()
                symbols = [r[0] for r in rows]
        finally:
            conn.close()

        return {sym: self.compute_ivr(sym) for sym in symbols}
