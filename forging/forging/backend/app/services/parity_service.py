from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import Settings
from app.services.engine_service import EngineService
from app.services.preprocess_service import PreprocessService
from app.utils.image_ops import pil_to_rgb_np
from app.utils.training_features import (
    compute_ela_multi,
    compute_laplacian,
    compute_noiseprint_map_training,
    compute_ocr_proxy,
    compute_srm_map_training,
)


@dataclass(slots=True)
class ParitySampleReport:
    sample_id: str
    image_path: str
    component_errors: dict[str, dict[str, float]]


class ParityService:
    def __init__(
        self,
        settings: Settings,
        preprocess_service: PreprocessService,
        engine_service: EngineService,
    ) -> None:
        self.settings = settings
        self.preprocess_service = preprocess_service
        self.engine_service = engine_service

    def review_image(self, image_path: Path) -> ParitySampleReport:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            original_rgb = pil_to_rgb_np(rgb_image)
            features = self.preprocess_service.extract_cpu_features(rgb_image)

        component_errors = {
            "ela_rgb": self._compare_arrays(
                features.ela_rgb.astype(np.float32),
                compute_ela_multi(rgb_image).astype(np.float32),
            ),
            "laplacian_rgb": self._compare_arrays(
                features.laplacian_rgb.astype(np.float32),
                compute_laplacian(original_rgb).astype(np.float32),
            ),
            "ocr_proxy": self._compare_arrays(
                features.ocr_proxy.astype(np.float32),
                compute_ocr_proxy(original_rgb).astype(np.float32),
            ),
            "srm_map": self._compare_arrays(
                self.engine_service._compute_srm_map(original_rgb).astype(np.float32),
                compute_srm_map_training(original_rgb).astype(np.float32),
            ),
            "noiseprint_map": self._compare_arrays(
                self.engine_service._compute_noiseprint_map(original_rgb).astype(np.float32),
                compute_noiseprint_map_training(original_rgb).astype(np.float32),
            ),
        }

        return ParitySampleReport(
            sample_id=image_path.stem,
            image_path=str(image_path),
            component_errors=component_errors,
        )

    def review_directory(self, dataset_dir: Path, sample_limit: int | None = None) -> dict[str, object]:
        image_paths = sorted(
            path
            for path in dataset_dir.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        )
        if sample_limit is not None:
            image_paths = image_paths[:sample_limit]

        reports = [self.review_image(path) for path in image_paths]
        max_mean_abs_error = max(
            (
                component["mean_abs_error"]
                for report in reports
                for component in report.component_errors.values()
            ),
            default=0.0,
        )
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sample_count": len(reports),
            "max_mean_abs_error": max_mean_abs_error,
            "samples": [
                {
                    "sample_id": report.sample_id,
                    "image_path": report.image_path,
                    "component_errors": report.component_errors,
                }
                for report in reports
            ],
        }
        self.settings.parity_report_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.parity_report_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        return payload

    @staticmethod
    def _compare_arrays(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
        delta = np.abs(left.astype(np.float32) - right.astype(np.float32))
        return {
            "mean_abs_error": float(delta.mean()) if delta.size else 0.0,
            "max_abs_error": float(delta.max()) if delta.size else 0.0,
        }
