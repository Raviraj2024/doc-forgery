from __future__ import annotations

from fastapi import APIRouter, Request
from app.schemas.responses import AuditLogEntry, GovernancePolicy

router = APIRouter(tags=["compliance"])

@router.get("/compliance/policies")
def get_policies(request: Request) -> list[GovernancePolicy]:
    return request.app.state.storage_service.get_governance_policies()

@router.get("/compliance/audit-log")
def get_audit_log(request: Request) -> list[AuditLogEntry]:
    return request.app.state.storage_service.get_audit_log(limit=100)
