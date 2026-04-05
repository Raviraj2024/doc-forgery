from __future__ import annotations

from fastapi import APIRouter, Request
from app.schemas.responses import AnalystOverrideHistoryItem

router = APIRouter(tags=["analyst"])

@router.get("/analyst/overrides")
def get_overrides(request: Request) -> list[AnalystOverrideHistoryItem]:
    return request.app.state.storage_service.get_analyst_overrides(limit=100)
