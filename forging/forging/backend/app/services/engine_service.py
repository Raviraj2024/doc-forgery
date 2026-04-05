from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from app.core.config import Settings
from app.services.artifact_service import ArtifactService
from app.services.preprocess_service import CPUFeatureBundle
from app.utils.image_ops import apply_heatmap, normalize_map
from app.utils.scoring import clamp01, map_score
from app.utils.training_features import (
    compute_noiseprint_map_training,
    compute_srm_map_training,
    extract_dino_distance_map,
)

try:
    import timm
except ImportError:  # pragma: no cover - depends on optional install
    timm = None


@dataclass(slots=True)
class PageEngineResult:
    ela_map: np.ndarray
    ela_score: float
    srm_map: np.ndarray
    srm_score: float
    noiseprint_map: np.ndarray
    noiseprint_score: float
    dino_map: np.ndarray
    dino_score: float
    ocr_proxy_map: np.ndarray
    ela_filename: str
    srm_filename: str
    noiseprint_filename: str
    dino_filename: str
    timings_ms: dict[str, int]


class EngineService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = "cuda" if torch.cuda.is_available() and settings.model_device != "cpu" else "cpu"
        self.dino_model = self._load_dino_model()
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _load_dino_model(self) -> torch.nn.Module | None:
        if timm is None:
            return None
        # Try preferred model first, then fall back (matches doctamper_l4.py training)
        candidates = [self.settings.dino_model_name]
        if "vit_small" in self.settings.dino_model_name:
            candidates.append("vit_tiny_patch16_224")
        elif "vit_tiny" in self.settings.dino_model_name:
            candidates.insert(0, "vit_small_patch16_224")

        for model_name in candidates:
            try:
                model = timm.create_model(
                    model_name,
                    pretrained=True,
                    num_classes=0,
                )
                model.eval()
                model.to(self.device)
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                self.logger.info("DINO backend initialised", extra={"model_name": model_name})
                return model
            except Exception as exc:
                self.logger.warning("DINO model %s failed: %s", model_name, exc)
        self.logger.warning("All DINO models failed, using fallback patch analysis")
        return None

    def analyze_page(
        self,
        features: CPUFeatureBundle,
        analysis_id: str,
        page_index: int,
        artifact_service: ArtifactService,
    ) -> PageEngineResult:
        ela_started = time.perf_counter()
        ela_map = normalize_map(features.ela_rgb.mean(axis=2))
        ela_ms = int((time.perf_counter() - ela_started) * 1000)

        srm_started = time.perf_counter()
        srm_map = self._compute_srm_map(features.original_rgb)
        srm_ms = int((time.perf_counter() - srm_started) * 1000)

        noiseprint_started = time.perf_counter()
        noiseprint_map = self._compute_noiseprint_map(features.original_rgb)
        noiseprint_ms = int((time.perf_counter() - noiseprint_started) * 1000)

        dino_started = time.perf_counter()
        dino_map = self._compute_dino_map(features.original_rgb)
        dino_ms = int((time.perf_counter() - dino_started) * 1000)
        ocr_proxy_map = normalize_map(features.ocr_proxy)

        ela_filename = f"page_{page_index}_ela.png"
        srm_filename = f"page_{page_index}_srm.png"
        noiseprint_filename = f"page_{page_index}_noiseprint.png"
        dino_filename = f"page_{page_index}_dino.png"

        artifact_service.save_array(analysis_id, ela_filename, apply_heatmap(ela_map))
        artifact_service.save_array(analysis_id, srm_filename, apply_heatmap(srm_map))
        artifact_service.save_array(analysis_id, noiseprint_filename, apply_heatmap(noiseprint_map))
        artifact_service.save_array(analysis_id, dino_filename, apply_heatmap(dino_map))

        return PageEngineResult(
            ela_map=ela_map,
            ela_score=map_score(ela_map),
            srm_map=srm_map,
            srm_score=map_score(srm_map),
            noiseprint_map=noiseprint_map,
            noiseprint_score=map_score(noiseprint_map),
            dino_map=dino_map,
            dino_score=map_score(dino_map),
            ocr_proxy_map=ocr_proxy_map,
            ela_filename=ela_filename,
            srm_filename=srm_filename,
            noiseprint_filename=noiseprint_filename,
            dino_filename=dino_filename,
            timings_ms={
                "ela": ela_ms,
                "srm": srm_ms,
                "noiseprint": noiseprint_ms,
                "dino": dino_ms,
            },
        )

    def build_combined_map(
        self,
        page_engines: PageEngineResult,
        segmentation_probability_map: np.ndarray,
    ) -> np.ndarray:
        combined = (
            0.25 * page_engines.ela_map
            + 0.25 * page_engines.srm_map
            + 0.20 * page_engines.noiseprint_map
            + 0.15 * page_engines.dino_map
            + 0.10 * normalize_map(segmentation_probability_map)
            + 0.05 * page_engines.ocr_proxy_map
        )
        return normalize_map(combined)

    def _compute_srm_map(self, image_rgb: np.ndarray) -> np.ndarray:
        return normalize_map(compute_srm_map_training(image_rgb))

    def _compute_noiseprint_map(self, image_rgb: np.ndarray) -> np.ndarray:
        return normalize_map(compute_noiseprint_map_training(image_rgb))

    def _compute_dino_map(self, image_rgb: np.ndarray) -> np.ndarray:
        if self.dino_model is not None:
            timm_map = self._compute_timm_dino_map(image_rgb)
            if timm_map is not None:
                return timm_map
        return self._compute_fallback_dino_map(image_rgb)

    def _compute_timm_dino_map(self, image_rgb: np.ndarray) -> np.ndarray | None:
        try:
            return normalize_map(
                extract_dino_distance_map(
                    dino_model=self.dino_model,
                    image_rgb=image_rgb,
                    device=self.device,
                    mean=self.imagenet_mean,
                    std=self.imagenet_std,
                )
            )
        except Exception as exc:
            self.logger.warning("DINO timm path failed, using fallback: %s", exc)
            return None

    def _compute_fallback_dino_map(self, image_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        patch_size = 16
        patch_vectors = []
        for y in range(0, 224, patch_size):
            for x in range(0, 224, patch_size):
                patch = resized[y : y + patch_size, x : x + patch_size]
                vector = np.concatenate([patch.mean(axis=(0, 1)), patch.std(axis=(0, 1))], axis=0)
                patch_vectors.append(vector)
        patch_matrix = np.stack(patch_vectors, axis=0)
        centroid = patch_matrix.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(patch_matrix - centroid, axis=1)
        distances = distances.reshape(14, 14)
        return normalize_map(cv2.resize(distances, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR))
