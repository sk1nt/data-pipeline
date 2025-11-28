import duckdb
from typing import List, Dict, Any, Optional
import os
from contextlib import closing


class DuckDBUtils:
    """Utility helpers that use short-lived DuckDB connections.

    Avoid holding an open connection on the process for long periods; all
    operations here open a connection, perform the work, and close it.
    """

    def __init__(self, db_path: str = "data_pipeline.db"):
        self.db_path = db_path

    def _ensure_dir(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    def execute_query(
        self, query: str, params: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results as list of dicts."""
        self._ensure_dir()
        with closing(duckdb.connect(self.db_path)) as conn:
            if params:
                result = conn.execute(query, params)
            else:
                result = conn.execute(query)

            # Get column names (duckdb returns description)
            try:
                columns = [desc[0] for desc in result.description]
            except Exception:
                columns = []

            rows = []
            for row in result.fetchall():
                if columns:
                    rows.append(dict(zip(columns, row)))
                else:
                    rows.append(row)
            return rows

    def execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows."""
        self._ensure_dir()
        with closing(duckdb.connect(self.db_path)) as conn:
            if params:
                result = conn.execute(query, params)
            else:
                result = conn.execute(query)
            return getattr(result, "rowcount", 0)

    def create_table_if_not_exists(self, table_name: str, schema: str):
        """Create a table if it doesn't exist (short-lived connection)."""
        self._ensure_dir()
        schema = schema.strip()
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})"
        with closing(duckdb.connect(self.db_path)) as conn:
            conn.execute(query)

    def insert_data(self, table_name: str, data: List[Dict[str, Any]]):
        """Insert data into a table using a short-lived connection."""
        if not data:
            return
        self._ensure_dir()
        columns = list(data[0].keys())
        placeholders = ", ".join(["?" for _ in columns])
        column_names = ", ".join(columns)
        query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"
        values = [tuple(row[col] for col in columns) for row in data]
        with closing(duckdb.connect(self.db_path)) as conn:
            conn.executemany(query, values)

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in DuckDB using information_schema."""
        self._ensure_dir()
        q = "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?"
        with closing(duckdb.connect(self.db_path)) as conn:
            try:
                cnt = conn.execute(q, (table_name,)).fetchone()[0]
                return int(cnt) > 0
            except Exception:
                return False

    def export_query_to_parquet(
        self, query: str, output_path: str, partition_by: Optional[List[str]] = None
    ):
        """Export query results to Parquet files via DuckDB COPY (short-lived conn)."""
        self._ensure_dir()
        options = ["FORMAT PARQUET"]
        if partition_by:
            parts = ", ".join(partition_by)
            options.append(f"PARTITION_BY ({parts})")

        options_clause = ", ".join(options)
        copy_sql = f"COPY ({query}) TO '{output_path}' ({options_clause})"
        with closing(duckdb.connect(self.db_path)) as conn:
            conn.execute(copy_sql)
