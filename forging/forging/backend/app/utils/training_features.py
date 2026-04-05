from __future__ import annotations

import io

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Exact channel statistics copied from doctamper_l4.py.
CHANNEL_MEAN = torch.tensor(
    [
        123.675,
        116.28,
        103.53,
        127.5,
        127.5,
        127.5,
        127.5,
        127.5,
        127.5,
        127.5,
        127.5,
        127.5,
        127.5,
    ],
    dtype=torch.float32,
).view(1, 13, 1, 1)

CHANNEL_STD = torch.tensor(
    [
        58.395,
        57.12,
        57.375,
        64.0,
        64.0,
        64.0,
        64.0,
        64.0,
        64.0,
        64.0,
        64.0,
        64.0,
        64.0,
    ],
    dtype=torch.float32,
).view(1, 13, 1, 1)

_SRM_KERNELS = np.stack(
    [
        np.array(
            [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
            dtype=np.float32,
        )
        / 4.0,
        np.array(
            [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
            dtype=np.float32,
        )
        / 12.0,
        np.array(
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            dtype=np.float32,
        )
        / 2.0,
        np.array(
            [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, -4, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
            dtype=np.float32,
        ),
        np.array(
            [[1, -2, 1, 0, 0], [-2, 4, -2, 0, 0], [1, -2, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            dtype=np.float32,
        )
        / 4.0,
    ],
    axis=0,
)


def compute_ela_multi(image: Image.Image, quality: int = 90) -> np.ndarray:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    compressed = np.array(Image.open(buffer).convert("RGB"), dtype=np.float32)
    original = np.array(image.convert("RGB"), dtype=np.float32)
    diff = np.abs(original - compressed)
    return (diff * 255.0 / (diff.max() + 1e-6)).astype(np.uint8)


def compute_laplacian(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fine = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=1))
    blur3 = cv2.GaussianBlur(gray, (3, 3), 0)
    medium = np.abs(cv2.Laplacian(blur3, cv2.CV_32F, ksize=3))
    blur5 = cv2.GaussianBlur(gray, (5, 5), 0)
    coarse = np.abs(cv2.Laplacian(blur5, cv2.CV_32F, ksize=5))
    lap = np.stack([fine, medium, coarse], axis=2)
    return (lap / (lap.max() + 1e-6) * 255.0).astype(np.uint8)


def compute_ocr_proxy(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gradient_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, gradient_kernel)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    return cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, horizontal_kernel).astype(np.float32)


def compute_srm_map_training(image_rgb: np.ndarray) -> np.ndarray:
    gray = (
        0.299 * image_rgb[:, :, 0].astype(np.float32)
        + 0.587 * image_rgb[:, :, 1].astype(np.float32)
        + 0.114 * image_rgb[:, :, 2].astype(np.float32)
    ) / 255.0
    responses = [
        np.abs(cv2.filter2D(gray, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT))
        for kernel in _SRM_KERNELS
    ]
    srm = np.mean(np.stack(responses, axis=0), axis=0)
    return srm / (float(srm.max()) + 1e-6)


def compute_noiseprint_map_training(image_rgb: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    gray = (
        0.299 * image_rgb[:, :, 0].astype(np.float32)
        + 0.587 * image_rgb[:, :, 1].astype(np.float32)
        + 0.114 * image_rgb[:, :, 2].astype(np.float32)
    ) / 255.0
    kernel = gaussian_kernel_2d(size=5, sigma=sigma)
    smooth = cv2.filter2D(gray, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)
    residual = np.abs(gray - smooth)
    return residual / (float(residual.max()) + 1e-6)


def gaussian_kernel_2d(size: int, sigma: float) -> np.ndarray:
    axis = np.arange(size, dtype=np.float32) - (size // 2)
    kernel_1d = np.exp(-(axis**2) / (2 * sigma**2))
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel_2d /= kernel_2d.sum()
    return kernel_2d.astype(np.float32)


def extract_dino_distance_map(
    *,
    dino_model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: str,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> np.ndarray:
    tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.0
    tensor = (tensor - mean.to(device)) / (std.to(device) + 1e-6)
    tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
    tensor = tensor.to(next(dino_model.parameters()).dtype)
    with torch.no_grad():
        features = dino_model.forward_features(tensor)

    patch_tokens: torch.Tensor | None = None
    cls_token: torch.Tensor | None = None
    if isinstance(features, dict):
        patch_tokens = features.get("x_norm_patchtokens")
        cls_token = features.get("x_norm_clstoken")
        if patch_tokens is None:
            x_value = features.get("x")
            if isinstance(x_value, torch.Tensor) and x_value.ndim == 3 and x_value.shape[1] > 1:
                patch_tokens = x_value[:, 1:]
                cls_token = x_value[:, :1]
    elif isinstance(features, torch.Tensor) and features.ndim == 3 and features.shape[1] > 1:
        patch_tokens = features[:, 1:]
        cls_token = features[:, :1]

    if patch_tokens is None or cls_token is None:
        raise RuntimeError("Unsupported DINO feature format for patch-distance extraction.")

    distance = torch.norm(patch_tokens - cls_token, dim=-1)
    distance_min = distance.amin(dim=1, keepdim=True)
    distance_max = distance.amax(dim=1, keepdim=True)
    distance = (distance - distance_min) / (distance_max - distance_min + 1e-6)
    side = int(distance.shape[1] ** 0.5)
    distance = distance.reshape(-1, 1, side, side)
    distance = F.interpolate(
        distance,
        size=(image_rgb.shape[0], image_rgb.shape[1]),
        mode="bilinear",
        align_corners=False,
    )
    return distance.squeeze().detach().cpu().numpy().astype(np.float32)

