from __future__ import annotations

from fastapi import APIRouter, Request
from app.schemas.responses import DevOpsMonitoringSummaryResponse, DevOpsTelemetryEntry

router = APIRouter(tags=["devops"])

@router.get("/devops/telemetry")
def get_telemetry(request: Request) -> list[DevOpsTelemetryEntry]:
    return request.app.state.storage_service.get_devops_telemetry(limit=100)


@router.get("/devops/monitoring", response_model=DevOpsMonitoringSummaryResponse)
def get_monitoring(request: Request) -> DevOpsMonitoringSummaryResponse:
    payload = request.app.state.storage_service.get_serving_monitoring_summary(
        recent_limit=20
    )
    return DevOpsMonitoringSummaryResponse.model_validate(payload)
