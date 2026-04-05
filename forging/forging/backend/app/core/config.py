from __future__ import annotations

import os
import json
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import BaseModel, Field


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _backend_root() -> Path:
    return _repo_root() / "backend"


def _env_candidate_paths() -> list[Path]:
    return [
        _repo_root() / ".env",
        _repo_root() / " - Copy.env",
        _backend_root() / ".env",
    ]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalise_database_url(value: str) -> str:
    raw = value.strip()
    if "://" not in raw:
        return raw

    scheme, remainder = raw.split("://", 1)
    if "/" in remainder:
        authority, suffix = remainder.split("/", 1)
        suffix = f"/{suffix}"
    else:
        authority = remainder
        suffix = ""

    if authority.count("@") <= 1:
        return raw

    userinfo, hostinfo = authority.rsplit("@", 1)
    if ":" not in userinfo:
        return raw

    username, password = userinfo.split(":", 1)
    encoded_username = quote(username, safe="")
    encoded_password = quote(password, safe="")
    return f"{scheme}://{encoded_username}:{encoded_password}@{hostinfo}{suffix}"


def _build_database_url() -> str | None:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return _normalise_database_url(database_url)

    host = os.getenv("DATABASE_HOST")
    if not host:
        return None

    port = os.getenv("DATABASE_PORT", "5432").strip()
    name = os.getenv("DATABASE_NAME", "postgres").strip()
    user = quote(os.getenv("DATABASE_USER", "postgres").strip(), safe="")
    password = quote(os.getenv("DATABASE_PASSWORD", "").strip(), safe="")
    return f"postgresql://{user}:{password}@{host.strip()}:{port}/{name}"


class ScoreWeights(BaseModel):
    ela: float = 0.18
    srm: float = 0.18
    noiseprint: float = 0.18
    dino_vit: float = 0.14
    ocr_anomaly: float = 0.12
    phash: float = 0.10
    segmentation: float = 0.10


class VerdictThresholds(BaseModel):
    clean_upper: float = 0.40
    suspicious_upper: float = 0.85


class LayerBenchmarkMetric(BaseModel):
    layer_name: str
    auc: float | None = None
    mean_positive_score: float | None = None
    mean_negative_score: float | None = None


class CalibrationProfile(BaseModel):
    dataset_name: str = "unavailable"
    generated_at: datetime | None = None
    sample_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    clean_upper: float | None = None
    suspicious_upper: float | None = None
    target_clean_specificity: float | None = None
    target_forgery_sensitivity: float | None = None
    recommended_weights: ScoreWeights | None = None
    mean_iou: float | None = None
    mean_f1: float | None = None
    precision: float | None = None
    recall: float | None = None
    risk_auc: float | None = None
    risk_brier: float | None = None
    layer_metrics: list[LayerBenchmarkMetric] = Field(default_factory=list)
    parity_report_path: str | None = None
    parity_sample_count: int | None = None
    parity_max_mean_abs_error: float | None = None


def _load_calibration_profile(path: Path) -> CalibrationProfile | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return CalibrationProfile.model_validate(payload)
    except Exception:
        return None


class Settings(BaseModel):
    model_config = {"protected_namespaces": ()}

    project_name: str = "Document Forgery Backend"
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    model_device: Literal["auto", "cpu", "cuda"] = "auto"
    checkpoint_path: Path
    inference_size: int = 512
    mask_threshold: float = 0.50
    pdf_dpi: int = 200
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ]
    )
    data_dir: Path
    uploads_dir: Path
    outputs_dir: Path
    artifacts_dir: Path
    calibration_profile_path: Path
    parity_report_path: Path
    db_path: Path
    database_url: str | None = None
    duplicate_near_threshold: int = 8
    duplicate_exact_threshold: int = 0
    enable_segmentation_in_final_score: bool = True
    score_weights: ScoreWeights = Field(default_factory=ScoreWeights)
    verdict_thresholds: VerdictThresholds = Field(default_factory=VerdictThresholds)
    dino_model_name: str = "vit_small_patch16_224"
    min_region_area_px: int = 64
    calibration_profile: CalibrationProfile | None = None

    @property
    def backend_root(self) -> Path:
        return _backend_root()

    @property
    def allowed_upload_suffixes(self) -> set[str]:
        return {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def build_settings() -> Settings:
    for path in _env_candidate_paths():
        _load_env_file(path)

    backend_root = _backend_root()
    data_dir = backend_root / "data"
    uploads_dir = data_dir / "uploads"
    outputs_dir = data_dir / "outputs"
    artifacts_dir = data_dir / "artifacts"
    calibration_profile_path = data_dir / "calibration" / "latest.json"
    parity_report_path = data_dir / "parity" / "latest.json"
    db_path = data_dir / "db" / "analysis.db"
    checkpoint_default = _repo_root() / "forgery_best.pth"
    calibration_path = Path(
        _env_str("CALIBRATION_PROFILE_PATH", str(calibration_profile_path))
    )
    loaded_calibration = _load_calibration_profile(calibration_path)

    return Settings(
        project_name=_env_str("PROJECT_NAME", "Document Forgery Backend"),
        api_v1_prefix=_env_str("API_V1_PREFIX", "/api/v1"),
        log_level=_env_str("LOG_LEVEL", "INFO"),
        model_device=_env_str("MODEL_DEVICE", "auto"),
        checkpoint_path=Path(_env_str("CHECKPOINT_PATH", str(checkpoint_default))),
        inference_size=_env_int("INFERENCE_SIZE", 512),
        mask_threshold=_env_float("MASK_THRESHOLD", 0.50),
        pdf_dpi=_env_int("PDF_DPI", 200),
        cors_origins=_env_list(
            "CORS_ORIGINS",
            [
                "http://localhost:3000",
                "http://localhost:3001",
                "http://localhost:5173",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173",
            ],
        ),
        data_dir=Path(_env_str("DATA_DIR", str(data_dir))),
        uploads_dir=Path(_env_str("UPLOADS_DIR", str(uploads_dir))),
        outputs_dir=Path(_env_str("OUTPUTS_DIR", str(outputs_dir))),
        artifacts_dir=Path(_env_str("ARTIFACTS_DIR", str(artifacts_dir))),
        calibration_profile_path=calibration_path,
        parity_report_path=Path(_env_str("PARITY_REPORT_PATH", str(parity_report_path))),
        db_path=Path(_env_str("DB_PATH", str(db_path))),
        database_url=_build_database_url(),
        duplicate_near_threshold=_env_int("DUPLICATE_NEAR_THRESHOLD", 8),
        duplicate_exact_threshold=_env_int("DUPLICATE_EXACT_THRESHOLD", 0),
        enable_segmentation_in_final_score=_env_bool(
            "ENABLE_SEGMENTATION_IN_FINAL_SCORE", True
        ),
        score_weights=loaded_calibration.recommended_weights
        if loaded_calibration and loaded_calibration.recommended_weights is not None
        else ScoreWeights(
            ela=_env_float("WEIGHT_ELA", 0.18),
            srm=_env_float("WEIGHT_SRM", 0.18),
            noiseprint=_env_float("WEIGHT_NOISEPRINT", 0.18),
            dino_vit=_env_float("WEIGHT_DINO_VIT", 0.14),
            ocr_anomaly=_env_float("WEIGHT_OCR_ANOMALY", 0.12),
            phash=_env_float("WEIGHT_PHASH", 0.10),
            segmentation=_env_float("WEIGHT_SEGMENTATION", 0.10),
        ),
        verdict_thresholds=VerdictThresholds(
            clean_upper=loaded_calibration.clean_upper
            if loaded_calibration and loaded_calibration.clean_upper is not None
            else _env_float("THRESHOLD_CLEAN_UPPER", 0.40),
            suspicious_upper=loaded_calibration.suspicious_upper
            if loaded_calibration and loaded_calibration.suspicious_upper is not None
            else _env_float("THRESHOLD_SUSPICIOUS_UPPER", 0.85),
        ),
        dino_model_name=_env_str("DINO_MODEL_NAME", "vit_small_patch16_224"),
        min_region_area_px=_env_int("MIN_REGION_AREA_PX", 64),
        calibration_profile=loaded_calibration,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return build_settings()
