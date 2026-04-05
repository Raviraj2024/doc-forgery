from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw
import torch

from app.services.benchmark_service import BenchmarkService
from app.services.parity_service import ParityService
from app.services.engine_service import EngineService
from app.services.preprocess_service import PreprocessService


def _write_sample(image_path: Path, mask_path: Path, *, tampered: bool) -> None:
    image = Image.new("RGB", (128, 128), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 18, 110, 110), fill=(60, 90, 150))
    if tampered:
        draw.rectangle((42, 44, 92, 88), fill=(190, 34, 48))
    image.save(image_path)

    mask = Image.new("L", (128, 128), color=0)
    if tampered:
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle((42, 44, 92, 88), fill=255)
    mask.save(mask_path)


class _FakeModelLoader:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.device = "cpu"
        self.model_loaded = True

    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        _, _, height, width = tensor.shape
        logits = torch.full((1, 1, height, width), -6.0, dtype=torch.float32)
        logits[:, :, height // 4 : (3 * height) // 4, width // 4 : (3 * width) // 4] = 6.0
        return logits


def test_benchmark_service_generates_calibration_profile(settings, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    _write_sample(images_dir / "clean.png", masks_dir / "clean.png", tampered=False)
    _write_sample(images_dir / "tampered.png", masks_dir / "tampered.png", tampered=True)

    preprocess = PreprocessService(settings)
    engine = EngineService(settings)
    parity = ParityService(settings, preprocess, engine)

    benchmark_service = BenchmarkService(
        settings=settings,
        model_loader=_FakeModelLoader(settings),
        preprocess_service=preprocess,
        engine_service=engine,
        parity_service=parity,
    )

    profile = benchmark_service.evaluate_directory(dataset_dir, dataset_name="synthetic-docs")

    assert profile.sample_count == 2
    assert profile.clean_upper is not None
    assert profile.suspicious_upper is not None
    assert profile.recommended_weights is not None
    assert abs(sum(profile.recommended_weights.model_dump().values()) - 1.0) < 1e-6
    assert settings.calibration_profile_path.exists()
    assert settings.parity_report_path.exists()
