from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from PIL import Image

from app.core.config import Settings
from app.schemas.responses import DuplicateStatus
from app.services.storage_service import StorageService
from app.utils.hashing import hamming_distance, md5_bytes, phash_from_image
from app.utils.image_ops import document_collage
from app.utils.scoring import clamp01


@dataclass(slots=True)
class DuplicateResult:
    duplicate_check: dict[str, object]
    phash_score: float
    warnings: list[str]


class DuplicateService:
    def __init__(self, settings: Settings, storage_service: StorageService) -> None:
        self.settings = settings
        self.storage_service = storage_service

    def check_document(self, payload: bytes, pages: list[Image.Image]) -> DuplicateResult:
        document_md5 = md5_bytes(payload)
        document_phash = phash_from_image(document_collage(pages))
        nearest_match = None
        nearest_distance = None

        for fingerprint in self.storage_service.list_fingerprints():
            if fingerprint["md5_hash"] == document_md5:
                nearest_match = fingerprint
                nearest_distance = 0
                break

            distance = hamming_distance(document_phash, fingerprint["phash"])
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_match = fingerprint

        status = DuplicateStatus.NO_MATCH
        phash_score = 0.0
        if nearest_match is not None and nearest_distance is not None:
            if nearest_match["md5_hash"] == document_md5 or nearest_distance <= self.settings.duplicate_exact_threshold:
                status = DuplicateStatus.EXACT_DUPLICATE
                phash_score = 1.0
            elif nearest_distance <= self.settings.duplicate_near_threshold:
                status = DuplicateStatus.NEAR_DUPLICATE
                phash_score = clamp01(
                    1.0 - (nearest_distance / max(1, self.settings.duplicate_near_threshold))
                )

        return DuplicateResult(
            duplicate_check={
                "md5_hash": document_md5,
                "phash": document_phash,
                "duplicate_status": status,
                "nearest_match_analysis_id": nearest_match["analysis_id"] if status != DuplicateStatus.NO_MATCH and nearest_match else None,
                "hamming_distance": nearest_distance if status != DuplicateStatus.NO_MATCH else None,
            },
            phash_score=phash_score,
            warnings=[],
        )

    def register_analysis(
        self,
        analysis_id: str,
        filename: str,
        md5_hash: str,
        phash: str,
        created_at: datetime,
    ) -> None:
        self.storage_service.upsert_fingerprint(
            analysis_id=analysis_id,
            filename=filename,
            md5_hash=md5_hash,
            phash=phash,
            created_at=created_at.astimezone(timezone.utc).isoformat(),
        )
