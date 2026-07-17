from pathlib import Path

import duckdb

from src.services.redis_flush_worker import RedisFlushWorker


class _FakeRedisClient:
    def __init__(self) -> None:
        self.client = self

    def exists(self, _key):
        return False

    def hgetall(self, _key):
        return {}

    def hset(self, *_args, **_kwargs):
        return None

    def delete(self, *_args):
        return None


def _create_gex_tables(db_path: Path) -> None:
    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gex_snapshots (
            timestamp BIGINT,
            ticker VARCHAR,
            spot_price DOUBLE,
            zero_gamma DOUBLE,
            net_gex DOUBLE,
            min_dte INTEGER,
            sec_min_dte INTEGER,
            major_pos_vol DOUBLE,
            major_pos_oi DOUBLE,
            major_pos_vol_gamma DOUBLE,
            major_neg_vol DOUBLE,
            major_neg_oi DOUBLE,
            major_neg_vol_gamma DOUBLE,
            sum_gex_vol DOUBLE,
            sum_gex_oi DOUBLE,
            gex_delta_15s DOUBLE,
            delta_risk_reversal DOUBLE,
            max_priors VARCHAR,
            pos_can1_strike DOUBLE,
            pos_can1_value DOUBLE,
            pos_can1_pct DOUBLE,
            pos_can2_strike DOUBLE,
            pos_can2_value DOUBLE,
            pos_can2_pct DOUBLE,
            neg_can1_strike DOUBLE,
            neg_can1_value DOUBLE,
            neg_can1_pct DOUBLE,
            neg_can2_strike DOUBLE,
            neg_can2_value DOUBLE,
            neg_can2_pct DOUBLE,
            strikes VARCHAR
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gex_strikes (
            timestamp BIGINT,
            ticker VARCHAR,
            strike DOUBLE,
            gamma DOUBLE,
            oi_gamma DOUBLE,
            priors VARCHAR
        )
        """
    )
    conn.close()


def test_write_gex_tables_dedupes_duplicate_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "gex_data.db"
    _create_gex_tables(db_path)
    worker = RedisFlushWorker(_FakeRedisClient(), object())
    worker.settings.gex_snapshot_db = db_path

    snapshot_rows = [
        {
            "timestamp": 1234,
            "ticker": "NQ_NDX",
            "spot_price": 100.0,
            "zero_gamma": 1.0,
            "net_gex": 2.0,
            "min_dte": 3,
            "sec_min_dte": 4,
            "major_pos_vol": 5.0,
            "major_pos_oi": 6.0,
            "major_pos_vol_gamma": 10.0,
            "major_neg_vol": 7.0,
            "major_neg_oi": 8.0,
            "major_neg_vol_gamma": -20.0,
            "sum_gex_vol": 9.0,
            "sum_gex_oi": 10.0,
            "gex_delta_15s": 10.5,
            "delta_risk_reversal": 11.0,
            "max_priors": "first",
            "pos_can1_strike": 20010.0,
            "pos_can1_value": 10.0,
            "pos_can1_pct": 50.0,
            "pos_can2_strike": 20011.0,
            "pos_can2_value": 5.0,
            "pos_can2_pct": 25.0,
            "neg_can1_strike": 19900.0,
            "neg_can1_value": -20.0,
            "neg_can1_pct": 40.0,
            "neg_can2_strike": 19899.0,
            "neg_can2_value": -10.0,
            "neg_can2_pct": 20.0,
        },
        {
            "timestamp": 1234,
            "ticker": "NQ_NDX",
            "spot_price": 101.0,
            "zero_gamma": 1.5,
            "net_gex": 2.5,
            "min_dte": 3,
            "sec_min_dte": 4,
            "major_pos_vol": 5.5,
            "major_pos_oi": 6.5,
            "major_pos_vol_gamma": 12.0,
            "major_neg_vol": 7.5,
            "major_neg_oi": 8.5,
            "major_neg_vol_gamma": -18.0,
            "sum_gex_vol": 9.5,
            "sum_gex_oi": 10.5,
            "gex_delta_15s": 11.5,
            "delta_risk_reversal": 11.5,
            "max_priors": "second",
            "pos_can1_strike": 20012.0,
            "pos_can1_value": 12.0,
            "pos_can1_pct": 60.0,
            "pos_can2_strike": 20013.0,
            "pos_can2_value": 6.0,
            "pos_can2_pct": 30.0,
            "neg_can1_strike": 19898.0,
            "neg_can1_value": -18.0,
            "neg_can1_pct": 45.0,
            "neg_can2_strike": 19897.0,
            "neg_can2_value": -9.0,
            "neg_can2_pct": 22.5,
        },
    ]
    strike_rows = [
        {
            "timestamp": 1234,
            "ticker": "NQ_NDX",
            "strike": 28974.92,
            "gamma": 1.0,
            "oi_gamma": 2.0,
            "priors": "first",
        },
        {
            "timestamp": 1234,
            "ticker": "NQ_NDX",
            "strike": 28974.92,
            "gamma": 3.0,
            "oi_gamma": 4.0,
            "priors": "second",
        },
    ]

    snapshot_count, strike_count = worker._write_gex_tables(snapshot_rows, strike_rows)

    assert snapshot_count == 1
    assert strike_count == 1

    conn = duckdb.connect(str(db_path), read_only=True)
    snapshot = conn.execute(
        """
        SELECT
            timestamp,
            ticker,
            spot_price,
            max_priors,
            major_pos_vol_gamma,
            major_neg_vol_gamma,
            pos_can1_strike,
            pos_can1_value,
            pos_can1_pct,
            pos_can2_strike,
            pos_can2_value,
            pos_can2_pct,
            neg_can1_strike,
            neg_can1_value,
            neg_can1_pct,
            neg_can2_strike,
            neg_can2_value,
            neg_can2_pct,
            gex_delta_15s
        FROM gex_snapshots
        """
    ).fetchone()
    strike = conn.execute(
        "SELECT timestamp, ticker, strike, gamma, oi_gamma, priors FROM gex_strikes"
    ).fetchone()
    conn.close()

    assert snapshot == (
        1234,
        "NQ_NDX",
        101.0,
        "second",
        12.0,
        -18.0,
        20012.0,
        12.0,
        60.0,
        20013.0,
        6.0,
        30.0,
        19898.0,
        -18.0,
        45.0,
        19897.0,
        -9.0,
        22.5,
        11.5,
    )
    assert strike == (1234, "NQ_NDX", 28974.92, 3.0, 4.0, "second")


def test_build_snapshot_row_projects_snapshot_candidate_values() -> None:
    worker = RedisFlushWorker(_FakeRedisClient(), object())
    snapshot = {
        "symbol": "NQ_NDX",
        "timestamp": 1234,
        "spot": 20000.0,
        "zero_gamma": 19950.0,
        "net_gex": 1.0,
        "major_pos_vol": 20010.0,
        "major_neg_vol": 19900.0,
        "sum_gex_vol": 2.0,
        "sum_gex_oi": 3.0,
        "gex_delta_15s": 0.2,
        "delta_risk_reversal": 0.1,
        "max_priors": ["x"],
        "pos_can1_strike": 20011.0,
        "pos_can1_value": 5.0,
        "pos_can1_pct": 50.0,
        "pos_can2_strike": 20012.0,
        "pos_can2_value": 2.5,
        "pos_can2_pct": 25.0,
        "neg_can1_strike": 19899.0,
        "neg_can1_value": -10.0,
        "neg_can1_pct": 50.0,
        "neg_can2_strike": 19898.0,
        "neg_can2_value": -5.0,
        "neg_can2_pct": 25.0,
        "strikes": [
            [20010.0, 10.0],
            [20011.0, 5.0],
            [20012.0, 2.5],
            [19900.0, -20.0],
            [19899.0, -10.0],
            [19898.0, -5.0],
        ],
    }

    row = worker._build_snapshot_row(snapshot)

    assert row is not None
    assert row["pos_can1_pct"] == 50.0
    assert row["pos_can2_pct"] == 25.0
    assert row["neg_can1_pct"] == 50.0
    assert row["neg_can2_pct"] == 25.0
