"""
Export 1-second MNQ OHLCV bars enriched with GEX fields into daily Parquet files.

The exporter:
- resamples MNQ ticks to a dense 1-second grid for the RTH session (09:30–16:00 America/New_York),
  filling missing seconds with the prior close and flagging gaps.
- pulls matching GEX snapshots + strike candidates for the same session (including the pre-open
  snapshot from the prior close), forward-filling across seconds.
- writes one Parquet per trade day to `data/enriched/<symbol>/<YYYYMMDD>.parquet`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo
import duckdb

NY_TZ = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    symbol: str = "MNQ"
    gex_ticker: str = "NQ_NDX"
    tick_root: Path = Path("data/parquet/tick")
    tick_db: Path = Path("data/tick_data.db")
    gex_db: Path = Path("data/gex_data.db")
    output_root: Path = Path("data/enriched")
    session_start: dt.time = dt.time(9, 30)
    session_end: dt.time = dt.time(16, 0)  # inclusive second
    strikes_source: str = "gex_zero"


class EnrichedExporter:
    def __init__(self, config: ExportConfig):
        self.config = config
        self.config.output_root.mkdir(parents=True, exist_ok=True)

    # ---------- Public entrypoints ----------
    def export_range(self, start_date: dt.date, end_date: dt.date) -> None:
        current = start_date
        while current <= end_date:
            self.export_day(current)
            current += dt.timedelta(days=1)

    def export_day(self, trade_date: dt.date) -> None:
        cfg = self.config
        tick_file = cfg.tick_root / cfg.symbol / f"{trade_date:%Y%m%d}.parquet"
        output_dir = cfg.output_root / cfg.symbol
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{trade_date:%Y%m%d}.parquet"

        if not tick_file.exists():
            raise FileNotFoundError(f"Tick file missing for {trade_date}: {tick_file}")

        session_start_utc, session_end_utc = self._session_bounds_utc(trade_date)
        snap_window_start_utc = session_start_utc - dt.timedelta(hours=18)

        logger.info(
            "Exporting %s (%s to %s UTC) -> %s",
            trade_date,
            session_start_utc,
            session_end_utc,
            output_file,
        )

        con = duckdb.connect()
        try:
            con.execute(f"ATTACH '{cfg.gex_db.as_posix()}' AS gexdb")
            if cfg.tick_db.exists():
                con.execute(f"ATTACH '{cfg.tick_db.as_posix()}' AS tickdb")

            self._create_seconds_table(con, session_start_utc, session_end_utc)
            self._load_ticks(con, tick_file, session_start_utc, session_end_utc)
            self._load_snapshots(con, snap_window_start_utc, session_end_utc)
            self._load_strikes(con, snap_window_start_utc, session_end_utc)
            self._join_and_write(con, output_file)

        finally:
            con.close()

    # ---------- DuckDB staging helpers ----------
    def _session_bounds_utc(
        self, trade_date: dt.date
    ) -> tuple[dt.datetime, dt.datetime]:
        start_local = dt.datetime.combine(trade_date, self.config.session_start, NY_TZ)
        end_local = dt.datetime.combine(trade_date, self.config.session_end, NY_TZ)
        return start_local.astimezone(UTC), end_local.astimezone(UTC)

    def _create_seconds_table(
        self,
        con: duckdb.DuckDBPyConnection,
        start_utc: dt.datetime,
        end_utc: dt.datetime,
    ) -> None:
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE seconds AS
            SELECT g.ts
            FROM generate_series(?, ?, INTERVAL 1 second) AS g(ts)
            """,
            [start_utc, end_utc],
        )

    def _load_ticks(
        self,
        con: duckdb.DuckDBPyConnection,
        tick_file: Path,
        start_utc: dt.datetime,
        end_utc: dt.datetime,
    ) -> None:
        # Build dense OHLCV with gap flag; fill missing seconds with prior close.
        if tick_file.exists():
            con.execute(
                """
                CREATE OR REPLACE TEMP TABLE ticks AS
                SELECT *
                FROM read_parquet(?)
                WHERE timestamp BETWEEN ? AND ?
                """,
                [tick_file.as_posix(), start_utc, end_utc],
            )
        elif con.execute(
            "SELECT 1 FROM information_schema.schemata WHERE catalog_name='tickdb'"
        ).fetchone():
            con.execute(
                """
                CREATE OR REPLACE TEMP TABLE ticks AS
                SELECT timestamp, price, volume
                FROM tickdb.tick_data
                WHERE symbol = ?
                  AND timestamp BETWEEN ? AND ?
                """,
                [self.config.symbol, start_utc, end_utc],
            )
        else:
            raise FileNotFoundError(
                f"No tick parquet {tick_file} and tick_db unavailable"
            )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE ohlcv AS
            WITH agg AS (
                SELECT
                    date_trunc('second', timestamp) AS ts,
                    first(price) AS open,
                    max(price) AS high,
                    min(price) AS low,
                    last(price) AS close,
                    sum(volume) AS volume
                FROM ticks
                GROUP BY ts
            ),
            grid AS (
                SELECT
                    s.ts,
                    a.open,
                    a.high,
                    a.low,
                    a.close,
                    a.volume
                FROM seconds s
                LEFT JOIN agg a ON a.ts = s.ts
                ORDER BY s.ts
            ),
            filled AS (
                SELECT
                    ts,
                    a.open,
                    a.high,
                    a.low,
                    a.close,
                    a.volume,
                    arg_max(a.close, a.ts) OVER w AS last_close,
                    a.open IS NULL AS tick_gap
                FROM grid a
                WINDOW w AS (ORDER BY a.ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            ),
            filled2 AS (
                SELECT
                    ts,
                    COALESCE(open, last_close) AS open_f,
                    COALESCE(high, last_close) AS high_f,
                    COALESCE(low,  last_close) AS low_f,
                    COALESCE(close, last_close) AS close_f,
                    COALESCE(volume, 0) AS volume_f,
                    tick_gap
                FROM filled
            )
            SELECT ts, open_f AS open, high_f AS high, low_f AS low, close_f AS close, volume_f AS volume, tick_gap
            FROM filled2
            """,
            [],
        )

    def _load_snapshots(
        self,
        con: duckdb.DuckDBPyConnection,
        start_utc: dt.datetime,
        end_utc: dt.datetime,
    ) -> None:
        cfg = self.config
        start_ms = int(start_utc.timestamp() * 1000)
        end_ms = int(end_utc.timestamp() * 1000)
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE snapshots AS
            SELECT
                timestamp AS snapshot_ts_ms,
                to_timestamp(timestamp/1000.0) AS snapshot_ts,
                spot_price AS spot,
                zero_gamma,
                net_gex,
                net_gex AS net_gex_vol,
                net_gex AS sum_gex_vol,
                major_pos_vol,
                major_pos_oi,
                major_neg_vol,
                major_neg_oi,
                sum_gex_oi AS net_gex_oi,
                delta_risk_reversal,
                min_dte,
                sec_min_dte,
                max_priors
            FROM gexdb.gex_snapshots
            WHERE ticker = ?
              AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
            """,
            [cfg.gex_ticker, start_ms, end_ms],
        )

    def _load_strikes(
        self,
        con: duckdb.DuckDBPyConnection,
        start_utc: dt.datetime,
        end_utc: dt.datetime,
    ) -> None:
        cfg = self.config
        start_ms = int(start_utc.timestamp() * 1000)
        end_ms = int(end_utc.timestamp() * 1000)
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE strikes AS
            SELECT timestamp, strike, gamma, oi_gamma
            FROM gexdb.gex_strikes
            WHERE ticker = ?
              AND timestamp BETWEEN ? AND ?
            """,
            [cfg.gex_ticker, start_ms, end_ms],
        )
        self._create_strike_candidates(con)

    @staticmethod
    def _create_strike_candidates(con: duckdb.DuckDBPyConnection) -> None:
        """Compute top/bottom three strikes per snapshot into strike_candidates."""
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE strike_candidates AS
            WITH ranked_pos AS (
                SELECT
                    timestamp,
                    strike,
                    gamma,
                    oi_gamma,
                    ROW_NUMBER() OVER (PARTITION BY timestamp ORDER BY gamma DESC, oi_gamma DESC) AS rn
                FROM strikes
                WHERE gamma IS NOT NULL AND gamma > 0
            ),
            ranked_neg AS (
                SELECT
                    timestamp,
                    strike,
                    gamma,
                    oi_gamma,
                    ROW_NUMBER() OVER (PARTITION BY timestamp ORDER BY gamma ASC, oi_gamma ASC) AS rn
                FROM strikes
                WHERE gamma IS NOT NULL AND gamma < 0
            )
            SELECT
                COALESCE(p.timestamp, n.timestamp) AS timestamp,
                max(CASE WHEN p.rn = 1 THEN p.gamma/1e6 END) AS major_pos_vol,
                max(CASE WHEN p.rn = 1 THEN p.strike END) AS major_pos_strike,
                max(CASE WHEN p.rn = 2 THEN p.gamma/1e6 END) AS major_pos_vol_2,
                max(CASE WHEN p.rn = 2 THEN p.strike END) AS major_pos_strike_2,
                max(CASE WHEN p.rn = 3 THEN p.gamma/1e6 END) AS major_pos_vol_3,
                max(CASE WHEN p.rn = 3 THEN p.strike END) AS major_pos_strike_3,
                max(CASE WHEN n.rn = 1 THEN n.gamma/1e6 END) AS major_neg_vol,
                max(CASE WHEN n.rn = 1 THEN n.strike END) AS major_neg_strike,
                max(CASE WHEN n.rn = 2 THEN n.gamma/1e6 END) AS major_neg_vol_2,
                max(CASE WHEN n.rn = 2 THEN n.strike END) AS major_neg_strike_2,
                max(CASE WHEN n.rn = 3 THEN n.gamma/1e6 END) AS major_neg_vol_3,
                max(CASE WHEN n.rn = 3 THEN n.strike END) AS major_neg_strike_3
            FROM ranked_pos p
            FULL OUTER JOIN ranked_neg n ON p.timestamp = n.timestamp
            GROUP BY COALESCE(p.timestamp, n.timestamp)
            """,
            [],
        )

    def _join_and_write(
        self, con: duckdb.DuckDBPyConnection, output_file: Path
    ) -> None:
        # Combine snapshots with strike candidates and forward fill across seconds.
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE snaps_enriched AS
            SELECT
                s.snapshot_ts,
                s.snapshot_ts_ms,
                s.spot,
                s.zero_gamma,
                s.net_gex,
                s.net_gex_vol,
                s.sum_gex_vol,
                COALESCE(c.major_pos_vol, s.major_pos_vol) AS major_pos_vol,
                s.major_pos_oi,
                COALESCE(c.major_neg_vol, s.major_neg_vol) AS major_neg_vol,
                s.major_neg_oi,
                s.net_gex_oi,
                s.delta_risk_reversal,
                s.min_dte,
                s.sec_min_dte,
                s.max_priors,
                COALESCE(c.major_pos_strike, s.major_pos_strike) AS major_pos_strike,
                COALESCE(c.major_neg_strike, s.major_neg_strike) AS major_neg_strike,
                c.major_pos_vol_2,
                c.major_pos_strike_2,
                c.major_pos_vol_3,
                c.major_pos_strike_3,
                c.major_neg_vol_2,
                c.major_neg_strike_2,
                c.major_neg_vol_3,
                c.major_neg_strike_3
            FROM snapshots s
            LEFT JOIN strike_candidates c ON c.timestamp = s.snapshot_ts_ms
            ORDER BY s.snapshot_ts
            """,
            [],
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE gex_ff AS
            WITH timeline AS (
                SELECT ts AS node_ts, 'second' AS kind FROM seconds
                UNION
                SELECT snapshot_ts AS node_ts, 'snapshot' AS kind FROM snaps_enriched
            ),
            ordered AS (
                SELECT
                    t.node_ts,
                    t.kind,
                    s.snapshot_ts,
                    s.snapshot_ts_ms,
                    s.spot,
                    s.zero_gamma,
                    s.net_gex,
                    s.net_gex_vol,
                    s.sum_gex_vol,
                    s.major_pos_vol,
                    s.major_pos_oi,
                    s.major_neg_vol,
                    s.major_neg_oi,
                    s.net_gex_oi,
                    s.delta_risk_reversal,
                    s.min_dte,
                    s.sec_min_dte,
                    s.max_priors,
                    s.major_pos_strike,
                    s.major_neg_strike,
                    s.major_pos_vol_2,
                    s.major_pos_strike_2,
                    s.major_pos_vol_3,
                    s.major_pos_strike_3,
                    s.major_neg_vol_2,
                    s.major_neg_strike_2,
                    s.major_neg_vol_3,
                    s.major_neg_strike_3
                FROM timeline t
                LEFT JOIN snaps_enriched s ON s.snapshot_ts = t.node_ts
            ),
            filled AS (
                SELECT
                    node_ts,
                    kind,
                    arg_max(snapshot_ts, node_ts) OVER w AS snapshot_ts,
                    arg_max(snapshot_ts_ms, node_ts) OVER w AS snapshot_ts_ms,
                    arg_max(spot, node_ts) OVER w AS spot,
                    arg_max(zero_gamma, node_ts) OVER w AS zero_gamma,
                    arg_max(net_gex, node_ts) OVER w AS net_gex,
                    arg_max(net_gex_vol, node_ts) OVER w AS net_gex_vol,
                    arg_max(sum_gex_vol, node_ts) OVER w AS sum_gex_vol,
                    arg_max(major_pos_vol, node_ts) OVER w AS major_pos_vol,
                    arg_max(major_pos_oi, node_ts) OVER w AS major_pos_oi,
                    arg_max(major_neg_vol, node_ts) OVER w AS major_neg_vol,
                    arg_max(major_neg_oi, node_ts) OVER w AS major_neg_oi,
                    arg_max(net_gex_oi, node_ts) OVER w AS net_gex_oi,
                    arg_max(delta_risk_reversal, node_ts) OVER w AS delta_risk_reversal,
                    arg_max(min_dte, node_ts) OVER w AS min_dte,
                    arg_max(sec_min_dte, node_ts) OVER w AS sec_min_dte,
                    arg_max(max_priors, node_ts) OVER w AS max_priors,
                    arg_max(major_pos_strike, node_ts) OVER w AS major_pos_strike,
                    arg_max(major_neg_strike, node_ts) OVER w AS major_neg_strike,
                    arg_max(major_pos_vol_2, node_ts) OVER w AS major_pos_vol_2,
                    arg_max(major_pos_strike_2, node_ts) OVER w AS major_pos_strike_2,
                    arg_max(major_pos_vol_3, node_ts) OVER w AS major_pos_vol_3,
                    arg_max(major_pos_strike_3, node_ts) OVER w AS major_pos_strike_3,
                    arg_max(major_neg_vol_2, node_ts) OVER w AS major_neg_vol_2,
                    arg_max(major_neg_strike_2, node_ts) OVER w AS major_neg_strike_2,
                    arg_max(major_neg_vol_3, node_ts) OVER w AS major_neg_vol_3,
                    arg_max(major_neg_strike_3, node_ts) OVER w AS major_neg_strike_3
                FROM ordered
                WINDOW w AS (ORDER BY node_ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            )
            SELECT
                node_ts AS timestamp,
                snapshot_ts,
                snapshot_ts_ms,
                spot,
                zero_gamma,
                net_gex,
                net_gex_vol,
                sum_gex_vol,
                major_pos_vol,
                major_pos_oi,
                major_neg_vol,
                major_neg_oi,
                net_gex_oi,
                delta_risk_reversal,
                min_dte,
                sec_min_dte,
                max_priors,
                major_pos_strike,
                major_neg_strike,
                major_pos_vol_2,
                major_pos_strike_2,
                major_pos_vol_3,
                major_pos_strike_3,
                major_neg_vol_2,
                major_neg_strike_2,
                major_neg_vol_3,
                major_neg_strike_3
            FROM filled
            WHERE kind = 'second'
            """,
            [],
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE final AS
            SELECT
                o.ts AS timestamp,
                o.open,
                o.high,
                o.low,
                o.close,
                o.volume,
                g.snapshot_ts_ms AS source_snapshot_ts,
                g.spot,
                g.zero_gamma,
                g.net_gex,
                g.net_gex_vol,
                g.sum_gex_vol,
                g.net_gex_oi,
                g.major_pos_vol,
                g.major_pos_oi,
                g.major_neg_vol,
                g.major_neg_oi,
                g.major_pos_strike,
                g.major_neg_strike,
                g.major_pos_vol_2,
                g.major_pos_strike_2,
                g.major_pos_vol_3,
                g.major_pos_strike_3,
                g.major_neg_vol_2,
                g.major_neg_strike_2,
                g.major_neg_vol_3,
                g.major_neg_strike_3,
                g.delta_risk_reversal,
                g.min_dte,
                g.sec_min_dte,
                g.max_priors,
                o.tick_gap OR (g.snapshot_ts IS NULL) AS gap_filled
            FROM ohlcv o
            LEFT JOIN gex_ff g ON g.timestamp = o.ts
            ORDER BY o.ts
            """,
            [],
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        con.execute(
            "COPY final TO ? (FORMAT 'parquet', COMPRESSION 'zstd', ROW_GROUP_SIZE 10000)",
            [output_file.as_posix()],
        )

        gap_count = con.execute(
            "SELECT count(*) FROM final WHERE gap_filled"
        ).fetchone()[0]
        row_count = con.execute("SELECT count(*) FROM final").fetchone()[0]
        logger.info(
            "Wrote %s rows to %s (gaps flagged: %s)", row_count, output_file, gap_count
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MNQ 1s OHLCV + GEX enriched Parquet."
    )
    parser.add_argument(
        "--start-date", required=True, help="Start trade date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", help="End trade date (YYYY-MM-DD); defaults to start-date"
    )
    parser.add_argument(
        "--symbol", default="MNQ", help="Symbol for tick data folder (default: MNQ)"
    )
    parser.add_argument(
        "--gex-ticker",
        default="NQ_NDX",
        help="Ticker in gex_snapshots/strikes (default: NQ_NDX)",
    )
    parser.add_argument(
        "--tick-root",
        default="data/parquet/tick",
        help="Root folder for tick parquet files",
    )
    parser.add_argument(
        "--gex-db",
        default="data/gex_data.db",
        help="Path to DuckDB containing gex_snapshots/strikes",
    )
    parser.add_argument(
        "--output-root",
        default="data/enriched",
        help="Root folder for enriched parquet output",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    start_date = dt.date.fromisoformat(args.start_date)
    end_date = dt.date.fromisoformat(args.end_date) if args.end_date else start_date
    config = ExportConfig(
        symbol=args.symbol.upper(),
        gex_ticker=args.gex_ticker.upper(),
        tick_root=Path(args.tick_root),
        gex_db=Path(args.gex_db),
        output_root=Path(args.output_root),
    )
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    exporter = EnrichedExporter(config)
    exporter.export_range(start_date, end_date)


if __name__ == "__main__":
    main()
