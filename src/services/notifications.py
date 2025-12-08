from __future__ import annotations

import logging
from src.lib.redis_client import get_redis_client

logger = logging.getLogger(__name__)


def notify_operator(msg: str) -> bool:
    """Send a simple operator notification (best-effort)."""
    try:
        rc = get_redis_client().client
        rc.lpush("notifications:operators", msg)
        return True
    except Exception as exc:
        logger.warning("Failed to send operator notification: %s", exc)
        return False
