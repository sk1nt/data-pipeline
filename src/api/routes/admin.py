from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any
import os
import json

from lib.logging import get_logger
from services.automated_options_service import AutomatedOptionsService
from services.tastytrade_client import tastytrade_client
from lib.redis_client import get_redis_client

logger = get_logger(__name__)
router = APIRouter(prefix="", tags=["admin"])


def _require_admin_key(request: Request):
    admin_key = os.getenv("ADMIN_API_KEY", "")
    if not admin_key:
        return True
    header_key = request.headers.get("X-ADMIN-KEY")
    return header_key == admin_key


@router.post("/alerts/process")
async def process_alert_admin(request: Request, payload: Dict[str, Any]):
    if not _require_admin_key(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    message = payload.get("message")
    channel_id = payload.get("channel_id")
    user_id = payload.get("user_id")
    dry_run = payload.get("dry_run", True)
    if not message or not channel_id or not user_id:
        raise HTTPException(status_code=400, detail="Missing required fields")
    svc = AutomatedOptionsService(tastytrade_client=tastytrade_client)
    try:
        result = await svc.process_alert(message, channel_id, user_id)
    except Exception as exc:
        logger.warning("Admin alert processing failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "ok", "result": result}


@router.get("/audit/recent")
async def get_recent_audit(request: Request, limit: int = 100, cursor: int = 0):
    if not _require_admin_key(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    rc = get_redis_client().client
    try:
        start = cursor
        end = cursor + limit - 1
        entries = rc.lrange("audit:automated_alerts", start, end)
        events = [json.loads(e) for e in entries or []]
        next_cursor = end + 1 if len(entries) == limit else None
        return {"events": events, "cursor": next_cursor}
    except Exception as exc:
        logger.warning("Failed to read recent audit events: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
