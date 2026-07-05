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
            major_neg_vol DOUBLE,
            major_neg_oi DOUBLE,
            sum_gex_vol DOUBLE,
            sum_gex_oi DOUBLE,
            delta_risk_reversal DOUBLE,
            max_priors VARCHAR,
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
            "major_neg_vol": 7.0,
            "major_neg_oi": 8.0,
            "sum_gex_vol": 9.0,
            "sum_gex_oi": 10.0,
            "delta_risk_reversal": 11.0,
            "max_priors": "first",
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
            "major_neg_vol": 7.5,
            "major_neg_oi": 8.5,
            "sum_gex_vol": 9.5,
            "sum_gex_oi": 10.5,
            "delta_risk_reversal": 11.5,
            "max_priors": "second",
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
        "SELECT timestamp, ticker, spot_price, max_priors FROM gex_snapshots"
    ).fetchone()
    strike = conn.execute(
        "SELECT timestamp, ticker, strike, gamma, oi_gamma, priors FROM gex_strikes"
    ).fetchone()
    conn.close()

    assert snapshot == (1234, "NQ_NDX", 101.0, "second")
    assert strike == (1234, "NQ_NDX", 28974.92, 3.0, 4.0, "second")
