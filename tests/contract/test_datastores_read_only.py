import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from src.api.app import app
from src.api import gex_api
from src.api.routes import datastores


def _create_temp_databases(base_dir: Path) -> Path:
    data_dir = base_dir / "data"
    data_dir.mkdir()

    gex_db = data_dir / "gex_data.db"
    with duckdb.connect(str(gex_db)) as conn:
        conn.execute(
            """
            CREATE TABLE gex_snapshots (
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
                call_wall_candidate1_pct DOUBLE,
                call_wall_candidate2_pct DOUBLE,
                put_wall_candidate1_pct DOUBLE,
                put_wall_candidate2_pct DOUBLE,
                strikes VARCHAR
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE gex_strikes (
                timestamp BIGINT,
                ticker VARCHAR,
                strike DOUBLE,
                gamma DOUBLE,
                oi_gamma DOUBLE,
                priors VARCHAR
            )
            """
        )
        conn.execute(
            """
            INSERT INTO gex_snapshots VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                1720008000000,
                "NQ_NDX",
                20000.0,
                123.4,
                456.7,
                1,
                2,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                0.5,
                json.dumps([[1.0, 2.0]]),
                12.5,
                7.5,
                10.0,
                5.0,
                json.dumps([{"strike": 20000.0, "gamma": 1.2}]),
            ],
        )
        conn.execute(
            """
            INSERT INTO gex_strikes VALUES
            (?, ?, ?, ?, ?, ?)
            """,
            [
                1720008000000,
                "NQ_NDX",
                20000.0,
                1.23,
                4.56,
                json.dumps([19900.0, 20000.0]),
            ],
        )

    uw_db = data_dir / "uw_messages.db"
    with duckdb.connect(str(uw_db)) as conn:
        conn.execute(
            """
            CREATE TABLE market_agg_state (
                received_at TIMESTAMP,
                date VARCHAR,
                call_premium DOUBLE,
                put_premium DOUBLE,
                call_premium_otm_only DOUBLE,
                put_premium_otm_only DOUBLE,
                delta DOUBLE,
                gamma DOUBLE,
                theta DOUBLE,
                vega DOUBLE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE option_trades (
                received_at TIMESTAMP,
                topic VARCHAR,
                topic_symbol VARCHAR,
                is_index_option BOOLEAN,
                ticker VARCHAR,
                option_chain_id VARCHAR,
                type VARCHAR,
                strike DOUBLE,
                expiry TIMESTAMP,
                dte INTEGER,
                cost_basis DOUBLE,
                volume BIGINT,
                price DOUBLE,
                tags VARCHAR,
                implied_volatility DOUBLE,
                delta DOUBLE,
                gamma DOUBLE,
                theta DOUBLE,
                vega DOUBLE,
                rho DOUBLE,
                premium DOUBLE,
                size BIGINT,
                open_interest BIGINT,
                underlying_price DOUBLE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE iv_history (
                symbol VARCHAR,
                trade_date DATE,
                avg_iv DOUBLE,
                min_iv DOUBLE,
                max_iv DOUBLE,
                trade_count INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO market_agg_state VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                datetime(2026, 7, 3, 12, 0, 20, tzinfo=timezone.utc),
                "2026-07-03",
                101.1,
                202.2,
                303.3,
                404.4,
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        )

    return data_dir


def test_router_paths_are_registered():
    paths = {route.path for route in app.routes}
    assert "/api/gex/snapshots" in paths
    assert "/api/gex/strikes" in paths
    assert "/api/uw/market-agg/history" in paths
    assert "/api/uw/option-trades" in paths
    assert "/api/uw/iv-history" in paths


def test_new_datastore_handlers_are_read_only(monkeypatch, tmp_path):
    data_dir = _create_temp_databases(tmp_path)
    monkeypatch.setattr("src.api.routes.datastores.settings.data_dir", str(data_dir))

    read_only_flags: list[bool | None] = []
    real_connect = duckdb.connect

    def recording_connect(*args, **kwargs):
        read_only_flags.append(kwargs.get("read_only"))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr("src.api.routes.datastores.duckdb.connect", recording_connect)

    async def run_checks():
        snapshots = await datastores.read_gex_snapshots(
            symbol="NQ_NDX", start=None, end=None, limit=1000
        )
        strikes = await datastores.read_gex_strikes(
            symbol="NQ_NDX", start=None, end=None, limit=1000
        )
        market_agg = await datastores.read_market_agg_history(limit=100)
        option_trades = await datastores.read_option_trades(
            ticker=None, topic_symbol=None, limit=100
        )
        iv_history = await datastores.read_iv_history(symbol=None, limit=100)
        return snapshots, strikes, market_agg, option_trades, iv_history

    snapshots, strikes, market_agg, option_trades, iv_history = asyncio.run(run_checks())

    assert snapshots["count"] == 1
    assert snapshots["data"][0]["ticker"] == "NQ_NDX"
    assert snapshots["data"][0]["strikes"][0]["strike"] == 20000.0
    assert strikes["data"][0]["priors"] == [19900.0, 20000.0]
    assert market_agg["data"][0]["call_premium"] == 101.1
    assert option_trades["count"] == 0
    assert iv_history["count"] == 0
    assert read_only_flags and all(flag is True for flag in read_only_flags)


def test_legacy_gex_handler_uses_read_only_connection(monkeypatch, tmp_path):
    data_dir = _create_temp_databases(tmp_path)
    monkeypatch.setattr("src.api.gex_api.settings.data_dir", str(data_dir))

    read_only_flags: list[bool | None] = []
    real_connect = duckdb.connect

    def recording_connect(*args, **kwargs):
        read_only_flags.append(kwargs.get("read_only"))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr("src.api.gex_api.duckdb.connect", recording_connect)

    async def run_check():
        return await gex_api.get_gex_data(
            symbol="NQ_NDX", start=None, end=None, limit=1000
        )

    snapshots = asyncio.run(run_check())

    assert snapshots[0].ticker == "NQ_NDX"
    assert read_only_flags and all(flag is True for flag in read_only_flags)
