import redis
import os
import json
from typing import Optional, Dict, Any

class RedisManager:
    def __init__(self):
        self.host = os.getenv('REDIS_HOST', 'localhost')
        self.port = int(os.getenv('REDIS_PORT', 6379))
        self.db = int(os.getenv('REDIS_DB', 0))
        self.password = os.getenv('REDIS_PASSWORD')
        self._connection: Optional[redis.Redis] = None

    @property
    def connection(self) -> redis.Redis:
        if self._connection is None:
            self._connection = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
        return self._connection

    def ping(self) -> bool:
        try:
            return self.connection.ping()
        except redis.ConnectionError:
            return False

    def set_tick_data(self, key: str, data: dict, ttl_seconds: int = 3600):
        """Store tick data with TTL."""
        self.connection.hset(key, mapping=data)
        self.connection.expire(key, ttl_seconds)

    def get_tick_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve tick data stored either as a hash (our writers) or a JSON string (GEX poller)."""
        try:
            data = self.connection.hgetall(key)
            if data:
                return data
        except redis.ResponseError:
            # Key exists but is not a hash; fall through to GET path
            pass

        raw = self.connection.get(key)
        if not raw:
            return None
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, dict):
                return decoded
        except Exception:
            # Fallback to a simple dict with the raw payload
            return {"value": raw}
        return None

redis_manager = RedisManager()
