from __future__ import annotations

import json
import logging
from typing import Optional

from src.lib.redis_client import get_redis_client
from src.models.audit import AuditRecord

logger = logging.getLogger(__name__)


def persist_audit_record(record: AuditRecord) -> bool:
    """Persist audit record to Redis list for quick ingestion.

    Args:
        record: AuditRecord instance

    Returns:
        bool: True if the write succeeded (best-effort), False otherwise
    """
    try:
        client = get_redis_client()
        payload = record.json()
        # LPush the audit payload onto the 'audit:events' list
        client.client.lpush("audit:events", payload)
        return True
    except Exception as e:
        logger.exception("Failed to persist audit record: %s", e)
        return False
