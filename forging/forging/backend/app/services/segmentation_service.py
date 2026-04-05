from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from PIL import Image

from app.core.config import Settings
from app.core.model_loader import ModelLoader
from app.services.artifact_service import ArtifactService
from app.utils.image_ops import blend_heatmap, grayscale_uint8, normalize_map
from app.utils.mask_ops import draw_contours, extract_regions, overlay_mask, threshold_mask
from app.utils.scoring import segmentation_score


@dataclass(slots=True)
class SegmentationResult:
    probability_map: np.ndarray
    binary_mask: np.ndarray
    regions: list[dict[str, int | float | str]]
    score: float
    mask_filename: str
    overlay_filename: str
    contours_filename: str
    warnings: list[str]
    processing_ms: int


class SegmentationService:
    def __init__(
        self,
        settings: Settings,
        model_loader: ModelLoader,
        artifact_service: ArtifactService,
    ) -> None:
        self.settings = settings
        self.model_loader = model_loader
        self.artifact_service = artifact_service

    def segment_page(
        self,
        analysis_id: str,
        page_index: int,
        original_image: Image.Image,
        original_rgb: np.ndarray,
        tensor,
    ) -> SegmentationResult:
        started = time.perf_counter()
        warnings: list[str] = []
        if self.model_loader.model_loaded:
            logits = self.model_loader.predict(tensor)
            probabilities = logits.sigmoid().squeeze().detach().cpu().numpy().astype(np.float32)
            probability_map = normalize_map(
                np.array(
                    Image.fromarray(grayscale_uint8(probabilities), mode="L").resize(
                        original_image.size,
                        Image.Resampling.BILINEAR,
                    ),
                    dtype=np.float32,
                )
            )
        else:
            probability_map = np.zeros((original_image.height, original_image.width), dtype=np.float32)
            warnings.append(
                self.model_loader.load_error or "Segmentation model unavailable; generated blank masks."
            )

        binary_mask = threshold_mask(probability_map, self.settings.mask_threshold)
        regions = extract_regions(
            probability_map=probability_map,
            binary_mask=binary_mask,
            page_index=page_index,
            min_area_px=self.settings.min_region_area_px,
        )
        score = segmentation_score(probability_map, binary_mask)

        probability_filename = f"page_{page_index}_probability.png"
        mask_filename = f"page_{page_index}_mask.png"
        overlay_filename = f"page_{page_index}_overlay.png"
        contours_filename = f"page_{page_index}_contours.png"

        self.artifact_service.save_array(analysis_id, probability_filename, grayscale_uint8(probability_map))
        self.artifact_service.save_array(analysis_id, mask_filename, binary_mask)
        self.artifact_service.save_array(
            analysis_id,
            overlay_filename,
            overlay_mask(original_rgb, probability_map),
        )
        self.artifact_service.save_array(
            analysis_id,
            contours_filename,
            draw_contours(original_rgb, binary_mask),
        )

        return SegmentationResult(
            probability_map=probability_map,
            binary_mask=binary_mask,
            regions=regions,
            score=score,
            mask_filename=mask_filename,
            overlay_filename=overlay_filename,
            contours_filename=contours_filename,
            warnings=warnings,
            processing_ms=int((time.perf_counter() - started) * 1000),
        )
