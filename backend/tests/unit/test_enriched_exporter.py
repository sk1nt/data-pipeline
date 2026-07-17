import datetime as dt
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.enriched_exporter import (  # noqa: E402
    EnrichedExporter,
    ExportConfig,
)


def test_load_snapshots_uses_canonical_snapshot_values(tmp_path):
    con = duckdb.connect()
    try:
        con.execute("CREATE SCHEMA gexdb")
        con.execute(
            """
            CREATE TABLE gexdb.gex_snapshots AS
            SELECT
                1000::BIGINT AS timestamp,
                'NQ_NDX'::VARCHAR AS ticker,
                21000.0::DOUBLE AS spot_price,
                20950.0::DOUBLE AS zero_gamma,
                123.0::DOUBLE AS sum_gex_vol,
                456.0::DOUBLE AS sum_gex_oi,
                7.5::DOUBLE AS gex_delta_15s,
                1.0::DOUBLE AS major_pos_vol,
                2.0::DOUBLE AS major_pos_oi,
                -3.0::DOUBLE AS major_neg_vol,
                -4.0::DOUBLE AS major_neg_oi,
                5.0::DOUBLE AS delta_risk_reversal,
                1::INTEGER AS min_dte,
                2::INTEGER AS sec_min_dte,
                3::INTEGER AS max_priors,
                21100.0::DOUBLE AS pos_can1_strike,
                10.0::DOUBLE AS pos_can1_value,
                0.5::DOUBLE AS pos_can1_pct,
                21200.0::DOUBLE AS pos_can2_strike,
                8.0::DOUBLE AS pos_can2_value,
                0.4::DOUBLE AS pos_can2_pct,
                20800.0::DOUBLE AS neg_can1_strike,
                -9.0::DOUBLE AS neg_can1_value,
                0.45::DOUBLE AS neg_can1_pct,
                20700.0::DOUBLE AS neg_can2_strike,
                -6.0::DOUBLE AS neg_can2_value,
                0.3::DOUBLE AS neg_can2_pct
            """
        )

        exporter = EnrichedExporter(
            ExportConfig(gex_db=tmp_path / "unused.db", output_root=tmp_path)
        )
        exporter._load_snapshots(
            con,
            dt.datetime.fromtimestamp(0, tz=dt.timezone.utc),
            dt.datetime.fromtimestamp(2, tz=dt.timezone.utc),
        )
        row = con.execute(
            """
            SELECT sum_gex_vol, gex_delta_15s,
                   pos_can1_strike, pos_can1_value, pos_can1_pct,
                   neg_can1_strike, neg_can1_value, neg_can1_pct
            FROM snapshots
            """
        ).fetchone()
    finally:
        con.close()

    assert row == (123.0, 7.5, 21100.0, 10.0, 0.5, 20800.0, -9.0, 0.45)
