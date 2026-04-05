from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas.responses import CalibrationProfileResponse, ModelInfoResponse

router = APIRouter(tags=["model"])


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    settings = request.app.state.settings
    payload = request.app.state.model_loader.info()
    payload.update(
        {
            "calibration_profile_path": str(settings.calibration_profile_path),
            "calibration_loaded": settings.calibration_profile is not None,
            "calibration_generated_at": settings.calibration_profile.generated_at
            if settings.calibration_profile
            else None,
            "calibration_sample_count": settings.calibration_profile.sample_count
            if settings.calibration_profile
            else None,
        }
    )
    return ModelInfoResponse.model_validate(payload)


@router.get("/model/calibration", response_model=CalibrationProfileResponse | None)
def model_calibration(request: Request) -> CalibrationProfileResponse | None:
    calibration = request.app.state.settings.calibration_profile
    if calibration is None:
        return None
    return CalibrationProfileResponse.model_validate(calibration.model_dump())
