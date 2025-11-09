import duckdb
from typing import List, Dict, Any, Optional
import os

class DuckDBUtils:
    """Utility class for DuckDB operations."""

    def __init__(self, db_path: str = "data_pipeline.db"):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def connect(self):
        """Connect to DuckDB database."""
        if self.conn is None:
            # Ensure directory exists (only if there's a directory path)
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            self.conn = duckdb.connect(self.db_path)

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results as list of dicts."""
        if self.conn is None:
            self.connect()

        if params:
            result = self.conn.execute(query, params)
        else:
            result = self.conn.execute(query)

        # Get column names
        columns = [desc[0] for desc in result.description]

        # Convert to list of dicts
        rows = []
        for row in result.fetchall():
            rows.append(dict(zip(columns, row)))

        return rows

    def execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows."""
        if self.conn is None:
            self.connect()

        if params:
            result = self.conn.execute(query, params)
        else:
            result = self.conn.execute(query)

        return result.rowcount

    def create_table_if_not_exists(self, table_name: str, schema: str):
        """Create a table if it doesn't exist."""
        if self.conn is None:
            self.connect()

        # Clean up schema string
        schema = schema.strip()

        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})"
        self.conn.execute(query)

    def insert_data(self, table_name: str, data: List[Dict[str, Any]]):
        """Insert data into a table."""
        if self.conn is None:
            self.connect()

        if not data:
            return

        # Get column names from first row
        columns = list(data[0].keys())
        placeholders = ', '.join(['?' for _ in columns])
        column_names = ', '.join(columns)

        query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"

        # Prepare data as tuples
        values = []
        for row in data:
            values.append(tuple(row[col] for col in columns))

        self.conn.executemany(query, values)

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        if self.conn is None:
            self.connect()

        result = self.conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name=?
        """, (table_name,))

        return len(result.fetchall()) > 0

    def export_query_to_parquet(self, query: str, output_path: str,
                                partition_by: Optional[List[str]] = None):
        """Export query results to Parquet files via DuckDB COPY."""
        if self.conn is None:
            self.connect()

        options = ["FORMAT PARQUET"]
        if partition_by:
            parts = ', '.join(partition_by)
            options.append(f"PARTITION_BY ({parts})")

        options_clause = ', '.join(options)
        copy_sql = f"COPY ({query}) TO '{output_path}' ({options_clause})"
        self.conn.execute(copy_sql)
