from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from app.core.config import Settings
from app.utils.image_ops import pil_to_rgb_np, resize_map, resize_rgb
from app.utils.training_features import (
    CHANNEL_MEAN,
    CHANNEL_STD,
    compute_ela_multi,
    compute_laplacian,
    compute_ocr_proxy,
)


@dataclass(slots=True)
class CPUFeatureBundle:
    original_rgb: np.ndarray
    inference_rgb: np.ndarray
    ela_rgb: np.ndarray
    inference_ela_rgb: np.ndarray
    laplacian_rgb: np.ndarray
    inference_laplacian_rgb: np.ndarray
    ocr_proxy: np.ndarray
    inference_ocr_proxy: np.ndarray
    width: int
    height: int


class PreprocessService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._channel_mean = CHANNEL_MEAN.clone()
        self._channel_std = CHANNEL_STD.clone()

    def extract_cpu_features(self, image: Image.Image) -> CPUFeatureBundle:
        original_rgb = pil_to_rgb_np(image)
        width, height = image.width, image.height
        inference_size = (self.settings.inference_size, self.settings.inference_size)

        ela_rgb = compute_ela_multi(image)
        laplacian_rgb = compute_laplacian(original_rgb)
        ocr_proxy = compute_ocr_proxy(original_rgb)

        return CPUFeatureBundle(
            original_rgb=original_rgb,
            inference_rgb=resize_rgb(original_rgb, inference_size),
            ela_rgb=ela_rgb,
            inference_ela_rgb=resize_rgb(ela_rgb, inference_size),
            laplacian_rgb=laplacian_rgb,
            inference_laplacian_rgb=resize_rgb(laplacian_rgb, inference_size),
            ocr_proxy=ocr_proxy,
            inference_ocr_proxy=resize_map(ocr_proxy, inference_size),
            width=width,
            height=height,
        )

    def build_segmentation_tensor(
        self,
        features: CPUFeatureBundle,
        srm_map: np.ndarray,
        noiseprint_map: np.ndarray,
        dino_map: np.ndarray,
    ) -> torch.Tensor:
        inference_size = (self.settings.inference_size, self.settings.inference_size)
        srm_resized = resize_map(srm_map, inference_size) * 255.0
        noiseprint_resized = resize_map(noiseprint_map, inference_size) * 255.0
        dino_resized = resize_map(dino_map, inference_size) * 255.0

        channels = [
            features.inference_rgb[..., channel].astype(np.float32) for channel in range(3)
        ]
        channels.extend(
            features.inference_ela_rgb[..., channel].astype(np.float32) for channel in range(3)
        )
        channels.extend(
            features.inference_laplacian_rgb[..., channel].astype(np.float32)
            for channel in range(3)
        )
        channels.append(features.inference_ocr_proxy.astype(np.float32))
        channels.append(srm_resized.astype(np.float32))
        channels.append(noiseprint_resized.astype(np.float32))
        channels.append(dino_resized.astype(np.float32))

        tensor = torch.from_numpy(np.stack(channels, axis=0)).unsqueeze(0)
        mean = self._channel_mean.to(tensor.device)
        std = self._channel_std.to(tensor.device)
        return (tensor - mean) / (std + 1e-6)
