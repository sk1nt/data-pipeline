"""
Publish critical service errors to the errors:critical:stream Redis pub/sub channel.
The Discord bot subscribes to this channel and DMs the configured alert user.

Usage:
    from src.services.error_publisher import publish_critical_error

    try:
        ...
    except Exception as exc:
        publish_critical_error("redis_flush_worker", "Flush failed", exc=exc)
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

ERROR_ALERT_CHANNEL = os.getenv("ERROR_ALERT_CHANNEL", "errors:critical:stream")


def publish_critical_error(
    service: str,
    message: str,
    exc: BaseException | None = None,
    **context: Any,
) -> bool:
    """
    Publish a critical error to the Redis pub/sub channel consumed by the Discord bot.

    Args:
        service:  Name of the service or module that encountered the error.
        message:  Short human-readable description of the error.
        exc:      Optional exception to include traceback from.
        **context: Additional key/value pairs included in the payload.

    Returns:
        True if the message was published, False on failure (never raises).
    """
    try:
        from src.lib.redis_client import get_redis_client

        payload: dict[str, Any] = {
            "service": service,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **context,
        }
        if exc is not None:
            payload["exception"] = type(exc).__name__
            payload["traceback"] = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )[-1500:]  # cap at 1500 chars to stay within Discord message limits

        rc = get_redis_client()
        rc.client.publish(ERROR_ALERT_CHANNEL, json.dumps(payload))
        return True
    except Exception as publish_exc:
        logger.warning("error_publisher: failed to publish error: %s", publish_exc)
        return False
