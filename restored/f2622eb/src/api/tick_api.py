from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from src.db.duckdb_utils import DuckDBUtils
from src.models.tick_record import TickRecord

router = APIRouter()

@router.get("/ticks", response_model=List[TickRecord])
async def get_tick_data(
    symbol: str = Query(..., enum=["MNQ", "NQ"], description="Symbol"),
    start: Optional[datetime] = Query(None, description="Start timestamp"),
    end: Optional[datetime] = Query(None, description="End timestamp")
):
    """Query tick records."""
    try:
        db = DuckDBUtils()
        with db:
            query = "SELECT * FROM tick_records WHERE symbol = ?"
            params = [symbol]

            if start:
                query += " AND timestamp >= ?"
                params.append(start)
            if end:
                query += " AND timestamp <= ?"
                params.append(end)

            query += " ORDER BY timestamp"

            result = db.execute_query(query, params)
            rows = result.fetchall()

            # Convert to TickRecord objects
            ticks = []
            for row in rows:
                data = dict(zip(result.description, row))
                ticks.append(TickRecord.from_dict(data))

            return ticks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))