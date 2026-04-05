from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    CLEAN = "CLEAN"
    SUSPICIOUS = "SUSPICIOUS"
    CONFIRMED_FORGERY = "CONFIRMED_FORGERY"


class DuplicateStatus(str, Enum):
    NO_MATCH = "NO_MATCH"
    NEAR_DUPLICATE = "NEAR_DUPLICATE"
    EXACT_DUPLICATE = "EXACT_DUPLICATE"


class OCRAnomalyType(str, Enum):
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    DUPLICATE_REFERENCE = "DUPLICATE_REFERENCE"
    SUSPICIOUS_KEYWORD = "SUSPICIOUS_KEYWORD"
    INVALID_DATE = "INVALID_DATE"
    OCR_WARNING = "OCR_WARNING"


class EngineScores(BaseModel):
    ela_score: float = Field(ge=0.0, le=1.0)
    srm_score: float = Field(ge=0.0, le=1.0)
    noiseprint_score: float = Field(ge=0.0, le=1.0)
    dino_vit_score: float = Field(ge=0.0, le=1.0)
    ocr_anomaly_score: float = Field(ge=0.0, le=1.0)
    phash_score: float = Field(ge=0.0, le=1.0)
    segmentation_score: float = Field(ge=0.0, le=1.0)


class OCRAnomaly(BaseModel):
    type: OCRAnomalyType
    description: str
    page_index: int | None = Field(default=None, ge=1)


class DuplicateCheck(BaseModel):
    md5_hash: str
    phash: str
    duplicate_status: DuplicateStatus
    nearest_match_analysis_id: str | None = None
    hamming_distance: int | None = None


class PageArtifacts(BaseModel):
    original_url: str
    mask_url: str
    overlay_url: str
    ela_heatmap_url: str
    srm_heatmap_url: str
    noiseprint_heatmap_url: str
    dino_heatmap_url: str
    combined_heatmap_url: str
    contours_url: str


class TamperedRegion(BaseModel):
    region_id: str
    page_index: int = Field(ge=1)
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    area_px: int = Field(ge=0)
    mean_mask_score: float = Field(ge=0.0, le=1.0)
    max_mask_score: float = Field(ge=0.0, le=1.0)


class PageResult(BaseModel):
    page_index: int = Field(ge=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    artifacts: PageArtifacts
    tampered_regions: list[TamperedRegion]


class ExtractedMetadata(BaseModel):
    metadata_type: str
    software_signature: str | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    modification_date_raw: str | None = None
    gps_data: dict | None = Field(default_factory=dict)
    raw_dump: dict = Field(default_factory=dict)


class DeviceFingerprint(BaseModel):
    device_hash: str | None = None
    user_agent: str | None = None
    browser: str | None = None
    os: str | None = None
    is_known_fraud_device: bool = False


class ForensicLayer(BaseModel):
    layer_name: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    processing_ms: int = Field(ge=0)


class AnalystReview(BaseModel):
    review_id: int
    analyst_user_id: str
    previous_verdict: str
    new_verdict: str
    override_reason: str
    reviewed_at: datetime


class AnalystOverrideHistoryItem(AnalystReview):
    analysis_id: str
    filename: str


class GovernancePolicy(BaseModel):
    policy_id: str
    description: str
    threshold_value: float
    is_active: bool
    updated_at: datetime


class RuleTrigger(BaseModel):
    policy_id: str
    severity: str
    triggered_at: datetime


class AuditLogEntry(BaseModel):
    id: int
    analysis_id: str
    filename: str
    verdict: Verdict
    forensic_risk_score: float = Field(ge=0.0, le=1.0)
    policy_id: str
    severity: str
    triggered_at: datetime


class DevOpsTelemetryEntry(BaseModel):
    layer_name: str
    execution_count: int = Field(ge=0)
    avg_processing_ms: float = Field(ge=0.0)
    avg_confidence_score: float = Field(ge=0.0, le=1.0)


class LayerBenchmarkMetricResponse(BaseModel):
    layer_name: str
    auc: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_positive_score: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_negative_score: float | None = Field(default=None, ge=0.0, le=1.0)


class CalibrationWeightsResponse(BaseModel):
    ela: float = Field(ge=0.0, le=1.0)
    srm: float = Field(ge=0.0, le=1.0)
    noiseprint: float = Field(ge=0.0, le=1.0)
    dino_vit: float = Field(ge=0.0, le=1.0)
    ocr_anomaly: float = Field(ge=0.0, le=1.0)
    phash: float = Field(ge=0.0, le=1.0)
    segmentation: float = Field(ge=0.0, le=1.0)


class CalibrationProfileResponse(BaseModel):
    dataset_name: str
    generated_at: datetime | None = None
    sample_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    clean_upper: float | None = Field(default=None, ge=0.0, le=1.0)
    suspicious_upper: float | None = Field(default=None, ge=0.0, le=1.0)
    target_clean_specificity: float | None = Field(default=None, ge=0.0, le=1.0)
    target_forgery_sensitivity: float | None = Field(default=None, ge=0.0, le=1.0)
    recommended_weights: CalibrationWeightsResponse | None = None
    mean_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_auc: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_brier: float | None = Field(default=None, ge=0.0)
    layer_metrics: list[LayerBenchmarkMetricResponse] = Field(default_factory=list)
    parity_report_path: str | None = None
    parity_sample_count: int | None = Field(default=None, ge=0)
    parity_max_mean_abs_error: float | None = Field(default=None, ge=0.0)


class MonitoringWarningEvent(BaseModel):
    analysis_id: str
    filename: str
    warning: str
    created_at: datetime


class DevOpsMonitoringSummaryResponse(BaseModel):
    total_analyses: int = Field(ge=0)
    analyses_with_warnings: int = Field(ge=0)
    analyses_with_segmentation_fallback: int = Field(ge=0)
    average_processing_time_ms: float = Field(ge=0.0)
    p50_processing_time_ms: float = Field(ge=0.0)
    p95_processing_time_ms: float = Field(ge=0.0)
    warning_rate: float = Field(ge=0.0, le=1.0)
    latest_analysis_at: datetime | None = None
    calibration_loaded: bool = False
    calibration_generated_at: datetime | None = None
    calibration_sample_count: int | None = Field(default=None, ge=0)
    calibration_mean_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_mean_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    recent_warning_events: list[MonitoringWarningEvent] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    analysis_id: str
    filename: str
    document_type: str | None = None
    submitter_id: str | None = None
    tenant_id: str | None = None
    session_ip_address: str | None = None
    session_geolocation: str | None = None
    page_count: int = Field(ge=1)
    device: str
    verdict: Verdict
    forensic_risk_score: float = Field(ge=0.0, le=1.0)
    is_human_reviewed: bool = False
    engine_scores: EngineScores
    forensic_layers: list[ForensicLayer] = Field(default_factory=list)
    ocr_anomalies: list[OCRAnomaly]
    duplicate_check: DuplicateCheck
    extracted_metadata: list[ExtractedMetadata] = Field(default_factory=list)
    device_fingerprint: DeviceFingerprint | None = None
    pages: list[PageResult]
    rule_triggers: list[RuleTrigger] = Field(default_factory=list)
    analyst_reviews: list[AnalystReview] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    processing_time_ms: int = Field(ge=0)
    created_at: datetime


class AnalysisHistoryItem(BaseModel):
    analysis_id: str
    filename: str
    document_type: str | None = None
    submitter_id: str | None = None
    tenant_id: str | None = None
    session_geolocation: str | None = None
    page_count: int
    verdict: Verdict
    forensic_risk_score: float
    duplicate_status: DuplicateStatus
    is_human_reviewed: bool = False
    ocr_anomaly_count: int = Field(default=0, ge=0)
    warning_count: int = Field(default=0, ge=0)
    tampered_region_count: int = Field(default=0, ge=0)
    processing_time_ms: int = Field(ge=0)
    created_at: datetime


class AnalysisHistoryResponse(BaseModel):
    page: int
    page_size: int
    total: int
    items: list[AnalysisHistoryItem]


class DashboardSummaryResponse(BaseModel):
    total_analyses: int = Field(ge=0)
    clean_count: int = Field(ge=0)
    suspicious_count: int = Field(ge=0)
    confirmed_forgery_count: int = Field(ge=0)
    exact_duplicate_count: int = Field(ge=0)
    near_duplicate_count: int = Field(ge=0)
    total_ocr_anomalies: int = Field(ge=0)
    average_risk_score: float = Field(ge=0.0, le=1.0)
    average_processing_time_ms: float = Field(ge=0.0)
    engine_averages: EngineScores
    recent_analyses: list[AnalysisHistoryItem]
    flagged_analyses: list[AnalysisHistoryItem]


class DeleteAnalysisResponse(BaseModel):
    analysis_id: str
    deleted: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint_exists: bool
    database_ready: bool


class ModelInfoResponse(BaseModel):
    checkpoint_path: str
    selected_encoder: str | None
    input_channels: int | None
    checkpoint_input_channels: int | None
    device: str
    model_parameter_count: int | None
    model_loaded: bool
    load_error: str | None = None
    tried_architectures: list[str] = Field(default_factory=list)
    checkpoint_sha256: str | None = None
    checkpoint_size_bytes: int | None = None
    calibration_profile_path: str | None = None
    calibration_loaded: bool = False
    calibration_generated_at: datetime | None = None
    calibration_sample_count: int | None = Field(default=None, ge=0)
