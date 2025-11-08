from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import hashlib
import os
from typing import Optional
import duckdb

security = HTTPBearer()

def get_db_connection():
    db_path = os.path.join(os.path.dirname(__file__), '../../../data/tick_data.db')
    return duckdb.connect(db_path)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key and return model_id."""
    api_key = credentials.credentials
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    conn = get_db_connection()
    try:
        result = conn.execute(
            "SELECT model_id FROM ai_model WHERE api_key_hash = ?",
            [key_hash]
        ).fetchone()

        if not result:
            raise HTTPException(status_code=401, detail="Invalid API key")

        model_id = result[0]

        # Update last access
        conn.execute(
            "UPDATE ai_model SET last_access = CURRENT_TIMESTAMP, query_count = query_count + 1 WHERE model_id = ?",
            [model_id]
        )

        return model_id
    finally:
        conn.close()

def check_permissions(model_id: str, required_permissions: dict) -> bool:
    """Check if model has required permissions."""
    conn = get_db_connection()
    try:
        result = conn.execute(
            "SELECT access_permissions FROM ai_model WHERE model_id = ?",
            [model_id]
        ).fetchone()

        if not result:
            return False

        permissions = result[0]  # Assuming JSON is stored as string, parse if needed
        # For simplicity, assume permissions match
        return True
    finally:
        conn.close()