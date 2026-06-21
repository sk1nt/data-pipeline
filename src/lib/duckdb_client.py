import duckdb
from ..config.tastytrade_config import config

class DuckDBClient:
    def __init__(self):
        self.connection = duckdb.connect(config.database_url)
        self._init_tables()
    
    def _init_tables(self):
        """Initialize database tables."""
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id VARCHAR PRIMARY KEY,
                symbol VARCHAR,
                quantity DOUBLE,
                order_type VARCHAR,
                price DOUBLE,
                status VARCHAR,
                environment VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                channel_id VARCHAR,
                user_id VARCHAR
            )
        """)
        
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS traders (
                discord_id VARCHAR PRIMARY KEY,
                permissions VARCHAR[],
                account_id VARCHAR,
                allocation_percentage DOUBLE
            )
        """)
        
        # Add other tables as needed
    
    def execute(self, query: str, params: tuple = None):
        if params:
            return self.connection.execute(query, params)
        return self.connection.execute(query)
    
    def fetch_one(self, query: str, params: tuple = None):
        result = self.execute(query, params)
        return result.fetchone()
    
    def fetch_all(self, query: str, params: tuple = None):
        result = self.execute(query, params)
        return result.fetchall()

# Global instance
duckdb_client = DuckDBClient()