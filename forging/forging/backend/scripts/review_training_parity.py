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
from app.services.engine_service import EngineService
from app.services.parity_service import ParityService
from app.services.preprocess_service import PreprocessService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review training/inference feature parity for one image or a directory of images.",
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Single image path or a directory containing review samples.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional cap when the target is a directory.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Override MODEL_DEVICE for DINO-backed feature extraction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = args.target.resolve()
    if not target.exists():
        print(f"Target not found: {target}", file=sys.stderr)
        return 1

    if args.device:
        os.environ["MODEL_DEVICE"] = args.device
    get_settings.cache_clear()
    settings = get_settings()

    preprocess_service = PreprocessService(settings)
    engine_service = EngineService(settings)
    parity_service = ParityService(
        settings=settings,
        preprocess_service=preprocess_service,
        engine_service=engine_service,
    )

    if target.is_dir():
        payload = parity_service.review_directory(target, sample_limit=args.sample_limit)
    else:
        report = parity_service.review_image(target)
        payload = {
            "generated_at": None,
            "sample_count": 1,
            "max_mean_abs_error": max(
                (
                    component["mean_abs_error"]
                    for component in report.component_errors.values()
                ),
                default=0.0,
            ),
            "samples": [
                {
                    "sample_id": report.sample_id,
                    "image_path": report.image_path,
                    "component_errors": report.component_errors,
                }
            ],
        }
        settings.parity_report_path.parent.mkdir(parents=True, exist_ok=True)
        settings.parity_report_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "target": str(target),
                "parity_report_path": str(settings.parity_report_path),
                "report": payload,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
