from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from src.db.duckdb_utils import DuckDBUtils
from src.models.market_depth import MarketDepth
import ast

router = APIRouter()


@router.get("/depth", response_model=List[MarketDepth])
async def get_depth_data(
    symbol: str = Query(..., enum=["MNQ"], description="Symbol"),
    start: Optional[datetime] = Query(None, description="Start timestamp"),
    end: Optional[datetime] = Query(None, description="End timestamp"),
):
    """Query market depth data."""
    try:
        db = DuckDBUtils()
        with db:
            query = "SELECT * FROM market_depth WHERE symbol = ?"
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

            # Convert to MarketDepth objects
            depths = []
            for row in rows:
                data = dict(zip(result.description, row))
                # Parse bids and asks strings back to lists
                if "bids" in data and data["bids"]:
                    data["bids"] = ast.literal_eval(data["bids"])
                if "asks" in data and data["asks"]:
                    data["asks"] = ast.literal_eval(data["asks"])
                depths.append(MarketDepth.from_dict(data))

            return depths
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
