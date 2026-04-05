from __future__ import annotations

from fastapi import APIRouter, File, Form, Header, Query, Request, UploadFile

from app.schemas.responses import (
    AnalysisHistoryResponse,
    AnalysisResponse,
    DeleteAnalysisResponse,
)

router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_document(
    request: Request,
    file: UploadFile = File(...),
    document_type: str | None = Form(default=None),
    submitter_id: str | None = Form(default=None),
    tenant_id: str | None = Form(default=None),
    user_agent: str | None = Header(default=None, alias="User-Agent"),
    x_forwarded_for: str | None = Header(default=None, alias="X-Forwarded-For"),
    x_geo_country: str | None = Header(default=None, alias="X-Geo-Country"),
) -> AnalysisResponse:
    ip_address = x_forwarded_for.split(",")[0].strip() if x_forwarded_for else request.client.host if request.client else None
    return await request.app.state.report_service.analyze_upload(
        upload_file=file,
        document_type=document_type,
        submitter_id=submitter_id,
        tenant_id=tenant_id,
        session_ip_address=ip_address,
        session_geolocation=x_geo_country,
        user_agent=user_agent,
    )


@router.get("/analyze/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(request: Request, analysis_id: str) -> AnalysisResponse:
    return request.app.state.report_service.get_analysis(analysis_id)


@router.get("/analyze", response_model=AnalysisHistoryResponse)
def list_analyses(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
) -> AnalysisHistoryResponse:
    return request.app.state.report_service.list_analyses(page=page, page_size=page_size)


@router.delete("/analyze/{analysis_id}", response_model=DeleteAnalysisResponse)
def delete_analysis(request: Request, analysis_id: str) -> DeleteAnalysisResponse:
    return request.app.state.report_service.delete_analysis(analysis_id)
