from __future__ import annotations

import os
import json
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import ScoreWeights, Settings, VerdictThresholds
from app.schemas.responses import DuplicateStatus, OCRAnomalyType, Verdict


class InMemoryStorageService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.analyses: dict[str, dict[str, Any]] = {}
        self.fingerprints: dict[str, dict[str, Any]] = {}
        self.policies = [
            {
                "policy_id": "MULTI_ENGINE_CONSENSUS",
                "description": "Synthetic policy for integration coverage.",
                "threshold_value": 0.85,
                "is_active": True,
                "updated_at": datetime.now(timezone.utc),
            }
        ]
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def database_ready(self) -> bool:
        return True

    def sanitise_filename(self, filename: str) -> str:
        return filename.replace(" ", "_")

    def save_upload(self, analysis_id: str, filename: str, payload: bytes) -> Path:
        safe_name = self.sanitise_filename(filename)
        path = self.settings.uploads_dir / f"{analysis_id}_{safe_name}"
        path.write_bytes(payload)
        return path

    def output_dir(self, analysis_id: str) -> Path:
        path = self.settings.outputs_dir / analysis_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_analysis_json(self, analysis_id: str, payload: dict[str, Any]) -> Path:
        path = self.output_dir(analysis_id) / "analysis.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def store_analysis(self, payload: dict[str, Any], upload_path: Path) -> None:
        output_path = self.save_analysis_json(payload["analysis_id"], payload)
        payload = dict(payload)
        payload["_upload_path"] = str(upload_path)
        payload["_output_json_path"] = str(output_path)
        self.analyses[payload["analysis_id"]] = payload

    def get_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        payload = self.analyses.get(analysis_id)
        return dict(payload) if payload else None

    def list_analyses(self, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        items = sorted(
            self.analyses.values(),
            key=lambda item: item["created_at"],
            reverse=True,
        )
        offset = (page - 1) * page_size
        page_items = items[offset : offset + page_size]
        return (
            [
                {
                    "analysis_id": item["analysis_id"],
                    "filename": item["filename"],
                    "document_type": item.get("document_type"),
                    "submitter_id": item.get("submitter_id"),
                    "tenant_id": item.get("tenant_id"),
                    "session_geolocation": item.get("session_geolocation"),
                    "page_count": item["page_count"],
                    "verdict": item["verdict"],
                    "forensic_risk_score": item["forensic_risk_score"],
                    "duplicate_status": item["duplicate_check"]["duplicate_status"],
                    "is_human_reviewed": item.get("is_human_reviewed", False),
                    "ocr_anomaly_count": len(item.get("ocr_anomalies", [])),
                    "warning_count": len(item.get("warnings", [])),
                    "tampered_region_count": sum(
                        len(page["tampered_regions"]) for page in item.get("pages", [])
                    ),
                    "processing_time_ms": item["processing_time_ms"],
                    "created_at": item["created_at"],
                }
                for item in page_items
            ],
            len(items),
        )

    def get_dashboard_summary(self, recent_limit: int = 8, flagged_limit: int = 8) -> dict[str, Any]:
        items, total = self.list_analyses(page=1, page_size=max(recent_limit, len(self.analyses)))
        flagged = [item for item in items if item["verdict"] != Verdict.CLEAN]
        analyses = list(self.analyses.values())
        if analyses:
            avg_risk = float(np.mean([item["forensic_risk_score"] for item in analyses]))
            avg_runtime = float(np.mean([item["processing_time_ms"] for item in analyses]))
            engine_keys = analyses[0]["engine_scores"].keys()
            engine_averages = {
                key: float(np.mean([item["engine_scores"][key] for item in analyses]))
                for key in engine_keys
            }
        else:
            avg_risk = 0.0
            avg_runtime = 0.0
            engine_averages = {
                "ela_score": 0.0,
                "srm_score": 0.0,
                "noiseprint_score": 0.0,
                "dino_vit_score": 0.0,
                "ocr_anomaly_score": 0.0,
                "phash_score": 0.0,
                "segmentation_score": 0.0,
            }
        return {
            "total_analyses": total,
            "clean_count": sum(1 for item in items if item["verdict"] == Verdict.CLEAN),
            "suspicious_count": sum(1 for item in items if item["verdict"] == Verdict.SUSPICIOUS),
            "confirmed_forgery_count": sum(1 for item in items if item["verdict"] == Verdict.CONFIRMED_FORGERY),
            "exact_duplicate_count": sum(1 for item in items if item["duplicate_status"] == DuplicateStatus.EXACT_DUPLICATE),
            "near_duplicate_count": sum(1 for item in items if item["duplicate_status"] == DuplicateStatus.NEAR_DUPLICATE),
            "total_ocr_anomalies": sum(len(item.get("ocr_anomalies", [])) for item in analyses),
            "average_risk_score": avg_risk,
            "average_processing_time_ms": avg_runtime,
            "engine_averages": engine_averages,
            "recent_analyses": items[:recent_limit],
            "flagged_analyses": flagged[:flagged_limit],
        }

    def delete_analysis(self, analysis_id: str) -> bool:
        payload = self.analyses.pop(analysis_id, None)
        if payload is None:
            return False
        upload_path = Path(payload["_upload_path"])
        if upload_path.exists():
            upload_path.unlink()
        return True

    def list_fingerprints(self) -> list[dict[str, Any]]:
        return list(self.fingerprints.values())

    def upsert_fingerprint(
        self,
        analysis_id: str,
        filename: str,
        md5_hash: str,
        phash: str,
        created_at: str,
    ) -> None:
        self.fingerprints[analysis_id] = {
            "analysis_id": analysis_id,
            "filename": filename,
            "md5_hash": md5_hash,
            "phash": phash,
            "created_at": created_at,
        }

    def get_analyst_overrides(self, limit: int = 100) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in self.analyses.values():
            for review in item.get("analyst_reviews", []):
                rows.append(
                    {
                        "review_id": review["review_id"],
                        "analysis_id": item["analysis_id"],
                        "filename": item["filename"],
                        "analyst_user_id": review["analyst_user_id"],
                        "previous_verdict": review["previous_verdict"],
                        "new_verdict": review["new_verdict"],
                        "override_reason": review["override_reason"],
                        "reviewed_at": review["reviewed_at"],
                    }
                )
        return rows[:limit]

    def get_governance_policies(self) -> list[dict[str, Any]]:
        return list(self.policies)

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for item in self.analyses.values():
            for index, trigger in enumerate(item.get("rule_triggers", []), start=1):
                entries.append(
                    {
                        "id": index,
                        "analysis_id": item["analysis_id"],
                        "filename": item["filename"],
                        "verdict": item["verdict"],
                        "forensic_risk_score": item["forensic_risk_score"],
                        "policy_id": trigger["policy_id"],
                        "severity": trigger["severity"],
                        "triggered_at": trigger["triggered_at"],
                    }
                )
        return entries[:limit]

    def get_devops_telemetry(self, limit: int = 100) -> list[dict[str, Any]]:
        telemetry: dict[str, list[dict[str, Any]]] = {}
        for item in self.analyses.values():
            for layer in item.get("forensic_layers", []):
                telemetry.setdefault(layer["layer_name"], []).append(layer)
        rows = []
        for layer_name, layers in telemetry.items():
            rows.append(
                {
                    "layer_name": layer_name,
                    "execution_count": len(layers),
                    "avg_processing_ms": float(np.mean([layer["processing_ms"] for layer in layers])),
                    "avg_confidence_score": float(np.mean([layer["confidence_score"] for layer in layers])),
                }
            )
        rows.sort(key=lambda row: row["avg_processing_ms"], reverse=True)
        return rows[:limit]

    def get_serving_monitoring_summary(self, recent_limit: int = 20) -> dict[str, Any]:
        analyses = sorted(
            self.analyses.values(),
            key=lambda item: item["created_at"],
            reverse=True,
        )
        processing_times = [float(item["processing_time_ms"]) for item in analyses]
        analyses_with_warnings = [item for item in analyses if item.get("warnings")]
        recent_warning_events = []
        for item in analyses[:recent_limit]:
            for warning in item.get("warnings", []):
                recent_warning_events.append(
                    {
                        "analysis_id": item["analysis_id"],
                        "filename": item["filename"],
                        "warning": warning,
                        "created_at": item["created_at"],
                    }
                )
        return {
            "total_analyses": len(analyses),
            "analyses_with_warnings": len(analyses_with_warnings),
            "analyses_with_segmentation_fallback": 0,
            "average_processing_time_ms": float(np.mean(processing_times)) if processing_times else 0.0,
            "p50_processing_time_ms": float(np.percentile(processing_times, 50)) if processing_times else 0.0,
            "p95_processing_time_ms": float(np.percentile(processing_times, 95)) if processing_times else 0.0,
            "warning_rate": float(len(analyses_with_warnings) / len(analyses)) if analyses else 0.0,
            "latest_analysis_at": analyses[0]["created_at"] if analyses else None,
            "calibration_loaded": False,
            "calibration_generated_at": None,
            "calibration_sample_count": None,
            "calibration_mean_iou": None,
            "calibration_mean_f1": None,
            "recent_warning_events": recent_warning_events[:recent_limit],
        }


class FakeModelLoader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = "cpu"
        self.model_loaded = True
        self.load_error = None
        self.selected_encoder = "mit_b4"
        self.input_channels = 13
        self.checkpoint_input_channels = 13
        self.model_parameter_count = 123456
        self.tried_architectures = ["mit_b4:13"]
        self.checkpoint_sha256 = "test-checkpoint"
        self.checkpoint_size_bytes = 1024

    def load(self) -> None:
        return None

    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        _, _, height, width = tensor.shape
        logits = torch.full((1, 1, height, width), -6.0, dtype=torch.float32)
        logits[:, :, height // 4 : (3 * height) // 4, width // 4 : (3 * width) // 4] = 6.0
        return logits

    def info(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.settings.checkpoint_path),
            "selected_encoder": self.selected_encoder,
            "input_channels": self.input_channels,
            "device": self.device,
            "model_parameter_count": self.model_parameter_count,
            "model_loaded": self.model_loaded,
            "load_error": self.load_error,
            "checkpoint_input_channels": self.checkpoint_input_channels,
            "tried_architectures": self.tried_architectures,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
        }


class FakeOCRService:
    def __init__(self, _settings: Settings) -> None:
        self.backend_name = "fakeocr"

    def analyze_document(self, pages: list[Image.Image]):  # noqa: ANN201
        return type(
            "OCRResult",
            (),
            {
                "anomalies": [
                    {
                        "type": OCRAnomalyType.SUSPICIOUS_KEYWORD,
                        "description": "Synthetic OCR anomaly for integration coverage.",
                        "page_index": 1,
                    }
                ],
                "score": 0.2,
                "warnings": [],
                "page_texts": ["integration test"],
                "backend_name": self.backend_name,
            },
        )()


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    return Settings(
        project_name="Integration Test Backend",
        checkpoint_path=tmp_path / "checkpoint.pth",
        inference_size=128,
        mask_threshold=0.5,
        pdf_dpi=72,
        cors_origins=["http://localhost:3000"],
        data_dir=data_dir,
        uploads_dir=data_dir / "uploads",
        outputs_dir=data_dir / "outputs",
        artifacts_dir=data_dir / "artifacts",
        calibration_profile_path=data_dir / "calibration" / "latest.json",
        parity_report_path=data_dir / "parity" / "latest.json",
        db_path=data_dir / "db" / "analysis.db",
        database_url=None,
        duplicate_near_threshold=8,
        duplicate_exact_threshold=0,
        enable_segmentation_in_final_score=True,
        score_weights=ScoreWeights(),
        verdict_thresholds=VerdictThresholds(clean_upper=0.2, suspicious_upper=0.5),
        dino_model_name="vit_small_patch16_224",
        min_region_area_px=8,
    )


@pytest.fixture()
def sample_image_bytes() -> bytes:
    image = Image.new("RGB", (160, 120), color=(244, 244, 244))
    draw = ImageDraw.Draw(image)
    draw.rectangle((28, 28, 116, 88), fill=(190, 30, 45))
    draw.text((36, 40), "TAMPER", fill=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def test_client(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> TestClient:
    os.environ["DOC_FORGERY_SKIP_DEFAULT_APP_BOOTSTRAP"] = "1"
    import app.main as app_main

    monkeypatch.setattr(app_main, "get_settings", lambda: settings)
    monkeypatch.setattr(app_main, "StorageService", InMemoryStorageService)
    monkeypatch.setattr(app_main, "ModelLoader", FakeModelLoader)
    monkeypatch.setattr(app_main, "OCRService", FakeOCRService)

    app = app_main.create_app()
    return TestClient(app)
