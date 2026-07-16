from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import duckdb
from fastapi import APIRouter, HTTPException, Query

from src.config import settings

router = APIRouter()

GEX_DB_NAME = "gex_data.db"
UW_DB_NAME = "uw_messages.db"


def _db_path(db_name: str) -> Path:
    return settings.data_path / db_name


def _run_read_only_query(
    db_name: str, query: str, params: Iterable[Any] = ()
) -> list[dict[str, Any]]:
    db_path = _db_path(db_name)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"{db_name} is not available")

    try:
        with closing(duckdb.connect(str(db_path), read_only=True)) as conn:
            result = conn.execute(query, tuple(params))
            columns = [desc[0] for desc in result.description or ()]
            rows: list[dict[str, Any]] = []
            for row in result.fetchall():
                rows.append(dict(zip(columns, row)))
            return rows
    except duckdb.CatalogException as exc:
        raise HTTPException(status_code=404, detail="Requested table is not available") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to read DuckDB data") from exc


def _parse_json_field(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _normalize_row(
    row: dict[str, Any],
    *,
    json_fields: tuple[str, ...] = (),
    epoch_ms_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        if key in json_fields:
            normalized[key] = _parse_json_field(value)
            continue
        if key in epoch_ms_fields and isinstance(value, (int, float)):
            normalized[key] = datetime.fromtimestamp(value / 1000, tz=timezone.utc)
            continue
        normalized[key] = value
    return normalized


@router.get("/gex/snapshots")
async def read_gex_snapshots(
    symbol: str = Query(..., description="Ticker symbol, e.g. NQ_NDX"),
    start: Optional[datetime] = Query(None, description="Inclusive start timestamp"),
    end: Optional[datetime] = Query(None, description="Inclusive end timestamp"),
    limit: int = Query(1000, ge=1, le=10_000),
) -> dict[str, Any]:
    clauses = ["ticker = ?"]
    params: list[Any] = [symbol]
    if start is not None:
        clauses.append("timestamp >= ?")
        params.append(_to_epoch_ms(start))
    if end is not None:
        clauses.append("timestamp <= ?")
        params.append(_to_epoch_ms(end))

    rows = _run_read_only_query(
        GEX_DB_NAME,
        f"""
        SELECT
            timestamp,
            ticker,
            spot_price,
            zero_gamma,
            net_gex,
            min_dte,
            sec_min_dte,
            major_pos_vol,
            major_pos_oi,
            major_pos_vol_gamma,
            major_neg_vol,
            major_neg_oi,
            major_neg_vol_gamma,
            sum_gex_vol,
            sum_gex_oi,
            delta_risk_reversal,
            max_priors,
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
            strikes
        FROM gex_snapshots
        WHERE {" AND ".join(clauses)}
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        [*params, limit],
    )

    data = [
        _normalize_row(
            row,
            json_fields=("max_priors", "strikes"),
            epoch_ms_fields=("timestamp",),
        )
        for row in rows
    ]
    return {"status": "ok", "data": data, "count": len(data)}


@router.get("/gex/strikes")
async def read_gex_strikes(
    symbol: str = Query(..., description="Ticker symbol, e.g. NQ_NDX"),
    start: Optional[datetime] = Query(None, description="Inclusive start timestamp"),
    end: Optional[datetime] = Query(None, description="Inclusive end timestamp"),
    limit: int = Query(1000, ge=1, le=10_000),
) -> dict[str, Any]:
    clauses = ["ticker = ?"]
    params: list[Any] = [symbol]
    if start is not None:
        clauses.append("timestamp >= ?")
        params.append(_to_epoch_ms(start))
    if end is not None:
        clauses.append("timestamp <= ?")
        params.append(_to_epoch_ms(end))

    rows = _run_read_only_query(
        GEX_DB_NAME,
        f"""
        SELECT
            timestamp,
            ticker,
            strike,
            gamma,
            oi_gamma,
            priors
        FROM gex_strikes
        WHERE {" AND ".join(clauses)}
        ORDER BY timestamp DESC, strike ASC
        LIMIT ?
        """,
        [*params, limit],
    )

    data = [
        _normalize_row(
            row,
            json_fields=("priors",),
            epoch_ms_fields=("timestamp",),
        )
        for row in rows
    ]
    return {"status": "ok", "data": data, "count": len(data)}


@router.get("/uw/market-agg/history")
async def read_market_agg_history(
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    rows = _run_read_only_query(
        UW_DB_NAME,
        """
        SELECT
            received_at,
            date,
            call_premium,
            put_premium,
            call_premium_otm_only,
            put_premium_otm_only,
            delta,
            gamma,
            theta,
            vega
        FROM market_agg_state
        ORDER BY received_at DESC
        LIMIT ?
        """,
        [limit],
    )
    return {"status": "ok", "data": rows, "count": len(rows)}


@router.get("/uw/option-trades")
async def read_option_trades(
    ticker: Optional[str] = Query(None, description="Filter by ticker"),
    topic_symbol: Optional[str] = Query(None, description="Filter by topic symbol"),
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    clauses = ["1=1"]
    params: list[Any] = []
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker)
    if topic_symbol:
        clauses.append("topic_symbol = ?")
        params.append(topic_symbol)

    rows = _run_read_only_query(
        UW_DB_NAME,
        f"""
        SELECT
            received_at,
            topic,
            topic_symbol,
            is_index_option,
            ticker,
            option_chain_id,
            type,
            strike,
            expiry,
            dte,
            cost_basis,
            volume,
            price,
            tags,
            implied_volatility,
            delta,
            gamma,
            theta,
            vega,
            rho,
            premium,
            size,
            open_interest,
            underlying_price
        FROM option_trades
        WHERE {" AND ".join(clauses)}
        ORDER BY received_at DESC
        LIMIT ?
        """,
        [*params, limit],
    )
    return {"status": "ok", "data": rows, "count": len(rows)}


@router.get("/uw/iv-history")
async def read_iv_history(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    clauses = ["1=1"]
    params: list[Any] = []
    if symbol:
        clauses.append("symbol = ?")
        params.append(symbol)

    rows = _run_read_only_query(
        UW_DB_NAME,
        f"""
        SELECT
            symbol,
            trade_date,
            avg_iv,
            min_iv,
            max_iv,
            trade_count
        FROM iv_history
        WHERE {" AND ".join(clauses)}
        ORDER BY trade_date DESC
        LIMIT ?
        """,
        [*params, limit],
    )
    return {"status": "ok", "data": rows, "count": len(rows)}


def _to_epoch_ms(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.astimezone(timezone.utc).timestamp() * 1000)
