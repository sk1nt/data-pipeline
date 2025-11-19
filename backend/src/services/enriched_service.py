from typing import List
import duckdb
import os
from datetime import datetime
from backend.src.models.enriched_data import EnrichedData

class EnrichedDataService:
    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(__file__), '../../../data/tick_data.db')

    def get_historical_data(
        self,
        symbols: List[str],
        start_time: datetime,
        end_time: datetime,
        interval: str = "1h"
    ) -> List[EnrichedData]:
        """Get historical enriched data for backtesting."""
        conn = duckdb.connect(self.db_path)
        try:
            # Query enriched data from database
            placeholders = ','.join(['?'] * len(symbols))
            result = conn.execute(f"""
                SELECT symbol, interval_start, interval_end, open_price, high_price,
                       low_price, close_price, total_volume, vwap
                FROM enriched_data
                WHERE symbol IN ({placeholders})
                AND interval_start >= ?
                AND interval_end <= ?
                ORDER BY interval_start
            """, symbols + [start_time, end_time]).fetchall()

            enriched_data = []
            for row in result:
                data = EnrichedData(
                    symbol=row[0],
                    interval_start=row[1],
                    interval_end=row[2],
                    open_price=row[3],
                    high_price=row[4],
                    low_price=row[5],
                    close_price=row[6],
                    total_volume=row[7],
                    vwap=row[8]
                )
                enriched_data.append(data)
            return enriched_data
        finally:
            conn.close()

    def aggregate_ticks_to_enriched(self, symbol: str, interval_minutes: int = 60):
        """Aggregate raw ticks into enriched data intervals."""
        # This would be called periodically to create enriched data
        # For now, placeholder implementation
        pass

enriched_service = EnrichedDataService()
