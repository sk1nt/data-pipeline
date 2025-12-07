from fastapi import APIRouter, HTTPException, Request
import os
from typing import Dict

from lib.logging import get_logger
from services.auth_service import AuthService

logger = get_logger(__name__)
router = APIRouter(prefix="/allowlist", tags=["allowlist"])


def _require_admin_key(request: Request):
    admin_key = os.getenv("ADMIN_API_KEY", "")
    if not admin_key:
        return True
    header_key = request.headers.get("X-ADMIN-KEY")
    return header_key == admin_key


@router.get("/users")
async def list_users(request: Request):
    if not _require_admin_key(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"users": AuthService.list_users_allowlist()}


@router.post("/users")
async def add_user(request: Request, payload: Dict[str, str]):
    if not _require_admin_key(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user_id")
    success = AuthService.add_user_to_allowlist(user_id)
    if not success:
        raise HTTPException(status_code=500, detail="Unable to add user to allowlist")
    return {"success": True, "user_id": user_id}


@router.delete("/users/{user_id}")
async def remove_user(request: Request, user_id: str):
    if not _require_admin_key(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    success = AuthService.remove_user_from_allowlist(user_id)
    if not success:
        raise HTTPException(status_code=500, detail="Unable to remove user from allowlist")
    return {"success": True, "user_id": user_id}


@router.get("/channels")
async def list_channels(request: Request):
    if not _require_admin_key(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"channels": AuthService.list_channels_allowlist()}


@router.post("/channels")
async def add_channel(request: Request, payload: Dict[str, str]):
    if not _require_admin_key(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    channel_id = payload.get("channel_id")
    if not channel_id:
        raise HTTPException(status_code=400, detail="Missing channel_id")
    success = AuthService.add_channel_to_allowlist(channel_id)
    if not success:
        raise HTTPException(status_code=500, detail="Unable to add channel to allowlist")
    return {"success": True, "channel_id": channel_id}


@router.delete("/channels/{channel_id}")
async def remove_channel(request: Request, channel_id: str):
    if not _require_admin_key(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    success = AuthService.remove_channel_from_allowlist(channel_id)
    if not success:
        raise HTTPException(status_code=500, detail="Unable to remove channel from allowlist")
    return {"success": True, "channel_id": channel_id}
