import pytest
import os
import tempfile
from backend.src.services.duckdb_service import get_db_connection, insert_ai_model

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Set up test database with required tables and test data."""
    # Use a temporary DB for tests
    test_db_path = tempfile.mktemp(suffix=".db")
    os.environ["DUCKDB_PATH"] = test_db_path
    
    # Create tables
    conn = get_db_connection()
    conn.close()
    
    # Insert test AI model
    insert_ai_model(
        model_id="test-model",
        name="Test AI Model",
        permissions={"symbols": ["AAPL"], "query_types": ["realtime", "historical"]},
        api_key_hash="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"  # hash of 'test-api-key'
    )
    
    yield
    
    # Cleanup
    if os.path.exists(test_db_path):
        os.unlink(test_db_path)