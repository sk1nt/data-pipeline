from datetime import datetime, timezone
import json
from typing import List, Optional
from zoneinfo import ZoneInfo

import duckdb
from fastapi import APIRouter, HTTPException, Query

from src.config import settings
from src.models.gex_snapshot import GEXSnapshot

NY_TZ = ZoneInfo("America/New_York")

router = APIRouter()


@router.get("/gex", response_model=List[GEXSnapshot])
async def get_gex_data(
    symbol: str = Query(..., description="Symbol, e.g., NQ_NDX"),
    start: Optional[datetime] = Query(None, description="Start timestamp"),
    end: Optional[datetime] = Query(None, description="End timestamp"),
    limit: Optional[int] = Query(
        1000, description="Maximum number of records to return", ge=1, le=10000
    ),
):
    """Query GEX snapshots."""
    try:
        query = "SELECT * FROM gex_snapshots WHERE ticker = ?"
        params = [symbol]

        if start:
            query += " AND timestamp >= ?"
            params.append(_to_epoch_ms(start))
        if end:
            query += " AND timestamp <= ?"
            params.append(_to_epoch_ms(end))

        query += " ORDER BY timestamp LIMIT ?"
        params.append(limit)

        db_path = settings.data_path / "gex_data.db"
        with duckdb.connect(str(db_path), read_only=True) as conn:
            result = conn.execute(query, tuple(params))
            columns = [desc[0] for desc in result.description]

            snapshots = []
            for row in result.fetchall():
                data = dict(zip(columns, row))
                if "strike_data" in data and data["strike_data"]:
                    data["strike_data"] = json.loads(data["strike_data"])
                if (
                    "max_priors" in data
                    and data["max_priors"]
                    and isinstance(data["max_priors"], str)
                ):
                    data["max_priors"] = json.loads(data["max_priors"])
                ts_value = data.get("timestamp")
                if isinstance(ts_value, int):
                    data["timestamp"] = _format_epoch_ms(ts_value)
                snapshots.append(GEXSnapshot.from_dict(data))

            return snapshots
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _to_epoch_ms(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=NY_TZ)
    else:
        value = value.astimezone(NY_TZ)
    return int(value.timestamp() * 1000)


def _format_epoch_ms(epoch_ms: int) -> datetime:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).astimezone(NY_TZ)
