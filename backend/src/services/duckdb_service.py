import duckdb
import os
from typing import Any, Dict, List
import logging
import json

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DUCKDB_PATH", "data/tick_data.db")

def get_db_connection():
    """Get DuckDB connection, creating database if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = duckdb.connect(DB_PATH)
    create_tables(conn)
    return conn

def create_tables(conn: duckdb.DuckDBPyConnection):
    """Create all necessary tables if they don't exist."""
    # AI Model table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_model (
            model_id VARCHAR PRIMARY KEY,
            name VARCHAR,
            access_permissions JSON,
            api_key_hash VARCHAR NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_access TIMESTAMP,
            query_count INTEGER DEFAULT 0
        )
    """)

    # Data Source table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS data_source (
            source_id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            type VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            last_update TIMESTAMP,
            error_count INTEGER DEFAULT 0
        )
    """)

    # Tick Data table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tick_data (
            id INTEGER PRIMARY KEY,
            symbol VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            price DECIMAL(10,4) NOT NULL,
            volume INTEGER,
            tick_type VARCHAR NOT NULL,
            source VARCHAR NOT NULL
        )
    """)

    # Enriched Data table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS enriched_data (
            id INTEGER PRIMARY KEY,
            symbol VARCHAR NOT NULL,
            interval_start TIMESTAMP NOT NULL,
            interval_end TIMESTAMP NOT NULL,
            open_price DECIMAL(10,4) NOT NULL,
            high_price DECIMAL(10,4) NOT NULL,
            low_price DECIMAL(10,4) NOT NULL,
            close_price DECIMAL(10,4) NOT NULL,
            total_volume INTEGER NOT NULL,
            vwap DECIMAL(10,4)
        )
    """)

    # Query History table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_history (
            query_id VARCHAR PRIMARY KEY,
            model_id VARCHAR NOT NULL,
            query_type VARCHAR NOT NULL,
            parameters JSON NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            response_time_ms INTEGER NOT NULL,
            data_points_returned INTEGER NOT NULL
        )
    """)

    # Service Status table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS service_status (
            service_name VARCHAR PRIMARY KEY,
            current_status VARCHAR NOT NULL,
            last_update_time TIMESTAMP NOT NULL,
            uptime_percentage DECIMAL(5,2),
            error_message VARCHAR
        )
    """)

    logger.info("Database tables created or verified")

def insert_ai_model(model_id: str, name: str, permissions: Dict[str, Any], api_key_hash: str):
    """Insert a new AI model."""
    conn = get_db_connection()
    try:
        conn.execute("""
            INSERT OR IGNORE INTO ai_model (model_id, name, access_permissions, api_key_hash)
            VALUES (?, ?, ?, ?)
        """, [model_id, name, json.dumps(permissions), api_key_hash])
        conn.commit()
    finally:
        conn.close()

def get_historical_ticks(symbol: str, start_time, end_time, interval: str = "1h") -> List[Dict[str, Any]]:
    """Get historical enriched data."""
    conn = get_db_connection()
    try:
        # For simplicity, return raw ticks aggregated
        result = conn.execute("""
            SELECT symbol, timestamp, price, volume
            FROM tick_data
            WHERE symbol = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """, [symbol, start_time, end_time]).fetchall()
        
        return [
            {
                "symbol": row[0],
                "timestamp": row[1],
                "price": float(row[2]),
                "volume": row[3]
            }
            for row in result
        ]
    finally:
        conn.close()

def store_tick_data(ticks: List[Dict[str, Any]]):
    """Store tick data."""
    conn = get_db_connection()
    try:
        for tick in ticks:
            conn.execute("""
                INSERT INTO tick_data (symbol, timestamp, price, volume, tick_type, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                tick["symbol"],
                tick["timestamp"],
                tick["price"],
                tick.get("volume"),
                tick.get("tick_type", "trade"),
                tick.get("source", "unknown")
            ])
        conn.commit()
    finally:
        conn.close()