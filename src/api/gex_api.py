from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from src.db.duckdb_utils import DuckDBUtils
from src.models.gex_snapshot import GEXSnapshot
import json

router = APIRouter()

@router.get("/gex", response_model=List[GEXSnapshot])
async def get_gex_data(
    symbol: str = Query(..., description="Symbol, e.g., NQ_NDX"),
    start: Optional[datetime] = Query(None, description="Start timestamp"),
    end: Optional[datetime] = Query(None, description="End timestamp"),
    limit: Optional[int] = Query(1000, description="Maximum number of records to return", ge=1, le=10000)
):
    """Query GEX snapshots."""
    try:
        db = DuckDBUtils()
        with db:
            query = "SELECT * FROM gex_snapshots WHERE ticker = ?"
            params = [symbol]

            if start:
                query += " AND timestamp >= ?"
                params.append(start)
            if end:
                query += " AND timestamp <= ?"
                params.append(end)

            query += " ORDER BY timestamp LIMIT ?"
            params.append(limit)

            results = db.execute_query(query, tuple(params))

            # Convert to GEXSnapshot objects
            snapshots = []
            for row in results:
                data = dict(row)
                # Parse strike_data JSON
                if 'strike_data' in data and data['strike_data']:
                    data['strike_data'] = json.loads(data['strike_data'])
                # Parse max_priors JSON if present
                if 'max_priors' in data and data['max_priors'] and isinstance(data['max_priors'], str):
                    data['max_priors'] = json.loads(data['max_priors'])
                snapshots.append(GEXSnapshot.from_dict(data))

            return snapshots
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))