from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings
from app.core.model_loader import ModelLoader
from app.services.benchmark_service import BenchmarkService
from app.services.engine_service import EngineService
from app.services.parity_service import ParityService
from app.services.preprocess_service import PreprocessService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the forgery pipeline on a labeled dataset and write a calibration profile.",
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Dataset root containing images/ and optionally masks/ subdirectories.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional logical name stored in the generated calibration profile.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit the number of discovered samples for a faster evaluation pass.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Override MODEL_DEVICE for this evaluation run.",
    )
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="Skip training/inference parity review during the benchmark run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1

    if args.device:
        os.environ["MODEL_DEVICE"] = args.device
    get_settings.cache_clear()
    settings = get_settings()

    model_loader = ModelLoader(settings)
    model_loader.load()
    preprocess_service = PreprocessService(settings)
    engine_service = EngineService(settings)
    parity_service = None
    if not args.skip_parity:
        parity_service = ParityService(
            settings=settings,
            preprocess_service=preprocess_service,
            engine_service=engine_service,
        )

    benchmark_service = BenchmarkService(
        settings=settings,
        model_loader=model_loader,
        preprocess_service=preprocess_service,
        engine_service=engine_service,
        parity_service=parity_service,
    )
    profile = benchmark_service.evaluate_directory(
        dataset_dir,
        dataset_name=args.dataset_name,
        sample_limit=args.sample_limit,
    )

    payload = {
        "dataset_dir": str(dataset_dir),
        "checkpoint_path": str(settings.checkpoint_path),
        "checkpoint_sha256": model_loader.checkpoint_sha256,
        "model_loaded": model_loader.model_loaded,
        "load_error": model_loader.load_error,
        "calibration_profile_path": str(settings.calibration_profile_path),
        "parity_report_path": str(settings.parity_report_path),
        "profile": profile.model_dump(mode="json"),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
