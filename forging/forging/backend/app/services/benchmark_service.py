from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import (
    CalibrationProfile,
    LayerBenchmarkMetric,
    ScoreWeights,
    Settings,
)
from app.core.model_loader import ModelLoader
from app.schemas.responses import EngineScores
from app.services.engine_service import EngineService
from app.services.parity_service import ParityService
from app.services.preprocess_service import PreprocessService
from app.services.segmentation_service import SegmentationService
from app.utils.scoring import clamp01, forensic_risk_score


@dataclass(slots=True)
class BenchmarkSample:
    sample_id: str
    image_path: Path
    mask_path: Path | None


class _NullArtifactService:
    def save_array(self, *_args, **_kwargs) -> Path | None:
        return None

    def save_image(self, *_args, **_kwargs) -> Path | None:
        return None

    def url_for(self, *_args, **_kwargs) -> str:
        return ""


class BenchmarkService:
    def __init__(
        self,
        settings: Settings,
        model_loader: ModelLoader,
        preprocess_service: PreprocessService,
        engine_service: EngineService,
        parity_service: ParityService | None = None,
    ) -> None:
        self.settings = settings
        self.model_loader = model_loader
        self.preprocess_service = preprocess_service
        self.engine_service = engine_service
        self.parity_service = parity_service
        self.segmentation_service = SegmentationService(
            settings=settings,
            model_loader=model_loader,
            artifact_service=_NullArtifactService(),
        )

    def discover_dataset(self, dataset_dir: Path) -> list[BenchmarkSample]:
        images_dir = dataset_dir / "images"
        masks_dir = dataset_dir / "masks"
        search_root = images_dir if images_dir.exists() else dataset_dir
        samples: list[BenchmarkSample] = []
        for image_path in sorted(search_root.rglob("*")):
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
                continue
            relative = image_path.relative_to(search_root)
            candidate_mask = (masks_dir / relative).with_suffix(".png")
            if not candidate_mask.exists():
                candidate_mask = None
            samples.append(
                BenchmarkSample(
                    sample_id=image_path.stem,
                    image_path=image_path,
                    mask_path=candidate_mask,
                )
            )
        return samples

    def evaluate_directory(
        self,
        dataset_dir: Path,
        *,
        dataset_name: str | None = None,
        sample_limit: int | None = None,
    ) -> CalibrationProfile:
        samples = self.discover_dataset(dataset_dir)
        if sample_limit is not None:
            samples = samples[:sample_limit]
        return self.evaluate_samples(
            samples=samples,
            dataset_name=dataset_name or dataset_dir.name,
            parity_dataset_dir=dataset_dir,
        )

    def evaluate_samples(
        self,
        *,
        samples: list[BenchmarkSample],
        dataset_name: str,
        parity_dataset_dir: Path | None = None,
    ) -> CalibrationProfile:
        labels: list[int] = []
        risk_scores: list[float] = []
        ious: list[float] = []
        f1_scores: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []
        layer_values: dict[str, list[float]] = {
            "ela": [],
            "srm": [],
            "noiseprint": [],
            "dino_vit": [],
            "segmentation": [],
        }

        for sample in samples:
            with Image.open(sample.image_path) as image_handle:
                image = image_handle.convert("RGB")
                features = self.preprocess_service.extract_cpu_features(image)
                page_engines = self.engine_service.analyze_page(
                    features=features,
                    analysis_id="benchmark",
                    page_index=1,
                    artifact_service=_NullArtifactService(),
                )
                segmentation_tensor = self.preprocess_service.build_segmentation_tensor(
                    features=features,
                    srm_map=page_engines.srm_map,
                    noiseprint_map=page_engines.noiseprint_map,
                    dino_map=page_engines.dino_map,
                )
                segmentation_result = self.segmentation_service.segment_page(
                    analysis_id="benchmark",
                    page_index=1,
                    original_image=image,
                    original_rgb=features.original_rgb,
                    tensor=segmentation_tensor,
                )

            ground_truth = self._load_mask(sample.mask_path, image.size) if sample.mask_path else None
            label = int(bool(ground_truth is not None and np.any(ground_truth > 0)))
            labels.append(label)

            engine_scores = EngineScores(
                ela_score=page_engines.ela_score,
                srm_score=page_engines.srm_score,
                noiseprint_score=page_engines.noiseprint_score,
                dino_vit_score=page_engines.dino_score,
                ocr_anomaly_score=0.0,
                phash_score=0.0,
                segmentation_score=segmentation_result.score,
            )
            risk_scores.append(forensic_risk_score(self.settings, engine_scores))

            layer_values["ela"].append(page_engines.ela_score)
            layer_values["srm"].append(page_engines.srm_score)
            layer_values["noiseprint"].append(page_engines.noiseprint_score)
            layer_values["dino_vit"].append(page_engines.dino_score)
            layer_values["segmentation"].append(segmentation_result.score)

            if ground_truth is not None:
                iou, precision, recall, f1 = self._binary_segmentation_metrics(
                    prediction=segmentation_result.binary_mask,
                    target=ground_truth,
                )
                ious.append(iou)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)

        layer_metrics = self._build_layer_metrics(labels=labels, layer_values=layer_values)
        recommended_weights = self._derive_weight_profile(layer_metrics)
        clean_upper, suspicious_upper = self._derive_thresholds(labels, risk_scores)

        parity_summary = None
        if self.parity_service is not None and parity_dataset_dir is not None:
            parity_root = (
                parity_dataset_dir / "images"
                if (parity_dataset_dir / "images").exists()
                else parity_dataset_dir
            )
            parity_summary = self.parity_service.review_directory(
                parity_root,
                sample_limit=min(len(samples), 5),
            )

        profile = CalibrationProfile(
            dataset_name=dataset_name,
            generated_at=datetime.now(timezone.utc),
            sample_count=len(samples),
            positive_count=sum(labels),
            negative_count=max(0, len(labels) - sum(labels)),
            clean_upper=clean_upper,
            suspicious_upper=suspicious_upper,
            target_clean_specificity=0.95,
            target_forgery_sensitivity=0.90,
            recommended_weights=recommended_weights,
            mean_iou=self._mean_or_none(ious),
            mean_f1=self._mean_or_none(f1_scores),
            precision=self._mean_or_none(precisions),
            recall=self._mean_or_none(recalls),
            risk_auc=self._binary_auc(labels, risk_scores),
            risk_brier=self._brier_score(labels, risk_scores),
            layer_metrics=layer_metrics,
            parity_report_path=str(self.settings.parity_report_path) if parity_summary else None,
            parity_sample_count=parity_summary["sample_count"] if parity_summary else None,
            parity_max_mean_abs_error=parity_summary["max_mean_abs_error"] if parity_summary else None,
        )
        self.settings.calibration_profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.calibration_profile_path.write_text(
            json.dumps(profile.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        return profile

    @staticmethod
    def _load_mask(mask_path: Path | None, image_size: tuple[int, int]) -> np.ndarray | None:
        if mask_path is None or not mask_path.exists():
            return None
        with Image.open(mask_path) as mask_handle:
            mask = mask_handle.convert("L").resize(image_size, Image.Resampling.NEAREST)
        return (np.array(mask, dtype=np.uint8) > 127).astype(np.uint8) * 255

    @staticmethod
    def _binary_segmentation_metrics(
        *,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> tuple[float, float, float, float]:
        pred = prediction > 0
        truth = target > 0
        tp = float(np.logical_and(pred, truth).sum())
        fp = float(np.logical_and(pred, ~truth).sum())
        fn = float(np.logical_and(~pred, truth).sum())

        if tp == 0.0 and fp == 0.0 and fn == 0.0:
            return 1.0, 1.0, 1.0, 1.0

        iou = tp / (tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)
        return (float(iou), float(precision), float(recall), float(f1))

    def _build_layer_metrics(
        self,
        *,
        labels: list[int],
        layer_values: dict[str, list[float]],
    ) -> list[LayerBenchmarkMetric]:
        metrics: list[LayerBenchmarkMetric] = []
        for layer_name, scores in layer_values.items():
            positive_scores = [score for score, label in zip(scores, labels, strict=True) if label == 1]
            negative_scores = [score for score, label in zip(scores, labels, strict=True) if label == 0]
            metrics.append(
                LayerBenchmarkMetric(
                    layer_name=layer_name,
                    auc=self._binary_auc(labels, scores),
                    mean_positive_score=self._mean_or_none(positive_scores),
                    mean_negative_score=self._mean_or_none(negative_scores),
                )
            )
        return metrics

    def _derive_weight_profile(self, layer_metrics: list[LayerBenchmarkMetric]) -> ScoreWeights:
        baseline = self.settings.score_weights
        raw_weights = {
            "ela": max(0.01, next((m.auc for m in layer_metrics if m.layer_name == "ela" and m.auc is not None), baseline.ela)),
            "srm": max(0.01, next((m.auc for m in layer_metrics if m.layer_name == "srm" and m.auc is not None), baseline.srm)),
            "noiseprint": max(
                0.01,
                next((m.auc for m in layer_metrics if m.layer_name == "noiseprint" and m.auc is not None), baseline.noiseprint),
            ),
            "dino_vit": max(
                0.01,
                next((m.auc for m in layer_metrics if m.layer_name == "dino_vit" and m.auc is not None), baseline.dino_vit),
            ),
            "ocr_anomaly": baseline.ocr_anomaly,
            "phash": baseline.phash,
            "segmentation": max(
                0.01,
                next((m.auc for m in layer_metrics if m.layer_name == "segmentation" and m.auc is not None), baseline.segmentation),
            ),
        }
        total = sum(raw_weights.values())
        return ScoreWeights(**{key: float(value / total) for key, value in raw_weights.items()})

    @staticmethod
    def _derive_thresholds(labels: list[int], scores: list[float]) -> tuple[float, float]:
        if not labels or len(set(labels)) < 2:
            return (0.40, 0.85)

        negatives = np.array([score for score, label in zip(scores, labels, strict=True) if label == 0], dtype=np.float32)
        positives = np.array([score for score, label in zip(scores, labels, strict=True) if label == 1], dtype=np.float32)

        clean_upper = float(np.quantile(negatives, 0.95)) if negatives.size else 0.40
        suspicious_upper = float(np.quantile(positives, 0.10)) if positives.size else 0.85
        clean_upper = clamp01(clean_upper)
        suspicious_upper = clamp01(max(clean_upper, suspicious_upper))
        return (clean_upper, suspicious_upper)

    @staticmethod
    def _binary_auc(labels: list[int], scores: list[float]) -> float | None:
        positives = np.array([score for score, label in zip(scores, labels, strict=True) if label == 1], dtype=np.float32)
        negatives = np.array([score for score, label in zip(scores, labels, strict=True) if label == 0], dtype=np.float32)
        if positives.size == 0 or negatives.size == 0:
            return None
        comparisons = positives[:, None] - negatives[None, :]
        auc = (np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size
        return float(auc)

    @staticmethod
    def _brier_score(labels: list[int], scores: list[float]) -> float | None:
        if not labels:
            return None
        label_array = np.array(labels, dtype=np.float32)
        score_array = np.array(scores, dtype=np.float32)
        return float(np.mean((score_array - label_array) ** 2))

    @staticmethod
    def _mean_or_none(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None
