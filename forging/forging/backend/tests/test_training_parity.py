from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from app.services.engine_service import EngineService
from app.services.preprocess_service import PreprocessService
from app.utils.training_features import (
    compute_ela_multi,
    compute_laplacian,
    compute_noiseprint_map_training,
    compute_ocr_proxy,
    compute_srm_map_training,
)


def _sample_image() -> Image.Image:
    image = Image.new("RGB", (96, 96), color=(235, 238, 244))
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 22, 72, 70), fill=(40, 88, 166))
    draw.text((24, 36), "AI", fill=(255, 255, 255))
    return image


def test_preprocess_matches_training_feature_stack(settings) -> None:
    image = _sample_image()
    preprocess = PreprocessService(settings)
    features = preprocess.extract_cpu_features(image)
    original_rgb = np.array(image.convert("RGB"))

    assert np.array_equal(features.ela_rgb, compute_ela_multi(image))
    assert np.array_equal(features.laplacian_rgb, compute_laplacian(original_rgb))
    assert np.array_equal(features.ocr_proxy, compute_ocr_proxy(original_rgb))


def test_engine_residual_maps_match_training_implementations(settings) -> None:
    image = _sample_image()
    engine = EngineService(settings)
    original_rgb = np.array(image.convert("RGB"))

    srm = engine._compute_srm_map(original_rgb)
    noiseprint = engine._compute_noiseprint_map(original_rgb)

    assert np.allclose(srm, compute_srm_map_training(original_rgb), atol=1e-6)
    assert np.allclose(noiseprint, compute_noiseprint_map_training(original_rgb), atol=1e-6)

