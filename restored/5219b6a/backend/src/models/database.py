import duckdb
import os

def init_database():
    """Initialize DuckDB database with schema."""
    db_path = os.path.join(os.path.dirname(__file__), '../../../data/tick_data.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = duckdb.connect(db_path)

    # Create tables
    conn.execute("""
    CREATE TABLE IF NOT EXISTS tick_data (
        symbol VARCHAR NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        price DECIMAL(10,4),
        volume BIGINT,
        tick_type VARCHAR NOT NULL,
        source VARCHAR NOT NULL,
        PRIMARY KEY (symbol, timestamp, source)
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS enriched_data (
        symbol VARCHAR NOT NULL,
        interval_start TIMESTAMP NOT NULL,
        interval_end TIMESTAMP NOT NULL,
        open_price DECIMAL(10,4),
        high_price DECIMAL(10,4),
        low_price DECIMAL(10,4),
        close_price DECIMAL(10,4),
        total_volume BIGINT,
        vwap DECIMAL(10,4),
        PRIMARY KEY (symbol, interval_start, interval_end)
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS ai_model (
        model_id VARCHAR PRIMARY KEY,
        name VARCHAR,
        access_permissions JSON,
        api_key_hash VARCHAR NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_access TIMESTAMP,
        query_count BIGINT DEFAULT 0
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS query_history (
        query_id UUID PRIMARY KEY,
        model_id VARCHAR NOT NULL,
        query_type VARCHAR NOT NULL,
        parameters JSON,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        response_time_ms BIGINT,
        data_points_returned BIGINT,
        FOREIGN KEY (model_id) REFERENCES ai_model(model_id)
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS data_source (
        source_id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        type VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        last_update TIMESTAMP,
        error_count BIGINT DEFAULT 0
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS service_status (
        service_name VARCHAR PRIMARY KEY,
        current_status VARCHAR NOT NULL,
        last_update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        uptime_percentage DECIMAL(5,2),
        error_message VARCHAR
    )
    """)

    conn.close()
    print("Database initialized successfully.")

if __name__ == "__main__":
    init_database()