from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_analysis import router as analysis_router
from app.api.routes_artifacts import router as artifacts_router
from app.api.routes_dashboard import router as dashboard_router
from app.api.routes_health import router as health_router
from app.api.routes_model import router as model_router
from app.api.routes_analyst import router as analyst_router
from app.api.routes_compliance import router as compliance_router
from app.api.routes_devops import router as devops_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.model_loader import ModelLoader
from app.services.artifact_service import ArtifactService
from app.services.duplicate_service import DuplicateService
from app.services.engine_service import EngineService
from app.services.ocr_service import OCRService
from app.services.pdf_service import PDFService
from app.services.preprocess_service import PreprocessService
from app.services.report_service import ReportService
from app.services.segmentation_service import SegmentationService
from app.services.storage_service import StorageService


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    storage_service = StorageService(settings)
    artifact_service = ArtifactService(settings)
    model_loader = ModelLoader(settings)
    model_loader.load()

    pdf_service = PDFService(settings)
    preprocess_service = PreprocessService(settings)
    engine_service = EngineService(settings)
    ocr_service = OCRService(settings)
    duplicate_service = DuplicateService(settings, storage_service)
    segmentation_service = SegmentationService(settings, model_loader, artifact_service)
    report_service = ReportService(
        settings=settings,
        model_loader=model_loader,
        storage_service=storage_service,
        artifact_service=artifact_service,
        pdf_service=pdf_service,
        preprocess_service=preprocess_service,
        engine_service=engine_service,
        ocr_service=ocr_service,
        duplicate_service=duplicate_service,
        segmentation_service=segmentation_service,
    )

    app = FastAPI(title=settings.project_name, version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.storage_service = storage_service
    app.state.model_loader = model_loader
    app.state.report_service = report_service

    app.include_router(health_router, prefix=settings.api_v1_prefix)
    app.include_router(model_router, prefix=settings.api_v1_prefix)
    app.include_router(analysis_router, prefix=settings.api_v1_prefix)
    app.include_router(dashboard_router, prefix=settings.api_v1_prefix)
    app.include_router(artifacts_router, prefix=settings.api_v1_prefix)
    app.include_router(analyst_router, prefix=settings.api_v1_prefix)
    app.include_router(compliance_router, prefix=settings.api_v1_prefix)
    app.include_router(devops_router, prefix=settings.api_v1_prefix)
    return app


def _default_app() -> FastAPI:
    if os.getenv("DOC_FORGERY_SKIP_DEFAULT_APP_BOOTSTRAP") == "1":
        return FastAPI(title="Document Forgery Backend (bootstrap skipped)", version="1.0.0")
    return create_app()


app = _default_app()
