from typing import List
import duckdb
import os
from backend.src.models.tick_data import TickData
from backend.src.services.redis_service import redis_manager

class TickService:
    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(__file__), '../../../data/tick_data.db')

    def get_realtime_ticks(self, symbols: List[str], limit: int = 100) -> List[TickData]:
        """Get real-time tick data from Redis cache."""
        ticks = []
        for symbol in symbols:
            # Try Redis first
            redis_key = f"tick:{symbol}:latest"
            cached_data = redis_manager.get_tick_data(redis_key)
            if cached_data:
                # Convert to TickData
                tick = TickData(**cached_data)
                ticks.append(tick)
                if len(ticks) >= limit:
                    break

        return ticks[:limit]

    def store_tick(self, tick: TickData, ttl_seconds: int = 3600):
        """Store tick in Redis with TTL."""
        redis_key = f"tick:{tick.symbol}:latest"
        data = tick.model_dump()
        redis_manager.set_tick_data(redis_key, data, ttl_seconds)

    def get_ticks_from_db(self, symbols: List[str], limit: int = 100) -> List[TickData]:
        """Fallback to database if not in cache."""
        conn = duckdb.connect(self.db_path)
        try:
            # Query recent ticks from database
            placeholders = ','.join(['?'] * len(symbols))
            result = conn.execute(f"""
                SELECT symbol, timestamp, price, volume, tick_type, source
                FROM tick_data
                WHERE symbol IN ({placeholders})
                ORDER BY timestamp DESC
                LIMIT ?
            """, symbols + [limit]).fetchall()

            ticks = []
            for row in result:
                tick = TickData(
                    symbol=row[0],
                    timestamp=row[1],
                    price=row[2],
                    volume=row[3],
                    tick_type=row[4],
                    source=row[5]
                )
                ticks.append(tick)
            return ticks
        finally:
            conn.close()

tick_service = TickService()
