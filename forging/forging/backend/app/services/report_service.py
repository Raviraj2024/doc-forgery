from __future__ import annotations

import hashlib
import io
import logging
import mimetypes
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from PIL import Image
from fastapi import HTTPException, UploadFile

from app.core.config import Settings
from app.core.model_loader import ModelLoader
from app.schemas.responses import (
    AnalysisHistoryItem,
    AnalysisHistoryResponse,
    AnalysisResponse,
    DeleteAnalysisResponse,
    DuplicateCheck,
    DuplicateStatus,
    EngineScores,
    OCRAnomaly,
    PageArtifacts,
    PageResult,
    TamperedRegion,
)
from app.services.artifact_service import ArtifactService
from app.services.duplicate_service import DuplicateService
from app.services.engine_service import EngineService
from app.services.ocr_service import OCRAnalysisResult, OCRService
from app.services.pdf_service import PDFService, RenderedPage
from app.services.preprocess_service import PreprocessService
from app.services.segmentation_service import SegmentationService
from app.services.storage_service import StorageService
from app.utils.image_ops import apply_heatmap, rgb_np_to_pil
from app.utils.scoring import forensic_risk_score, verdict_for_score


class ReportService:
    def __init__(
        self,
        settings: Settings,
        model_loader: ModelLoader,
        storage_service: StorageService,
        artifact_service: ArtifactService,
        pdf_service: PDFService,
        preprocess_service: PreprocessService,
        engine_service: EngineService,
        ocr_service: OCRService,
        duplicate_service: DuplicateService,
        segmentation_service: SegmentationService,
    ) -> None:
        self.settings = settings
        self.model_loader = model_loader
        self.storage_service = storage_service
        self.artifact_service = artifact_service
        self.pdf_service = pdf_service
        self.preprocess_service = preprocess_service
        self.engine_service = engine_service
        self.ocr_service = ocr_service
        self.duplicate_service = duplicate_service
        self.segmentation_service = segmentation_service
        self.logger = logging.getLogger(self.__class__.__name__)

    async def analyze_upload(
        self,
        upload_file: UploadFile,
        document_type: str | None = None,
        submitter_id: str | None = None,
        tenant_id: str | None = None,
        session_ip_address: str | None = None,
        session_geolocation: str | None = None,
        user_agent: str | None = None,
    ) -> AnalysisResponse:
        started = time.perf_counter()
        payload = await upload_file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        analysis_id = str(uuid.uuid4())
        filename = upload_file.filename or "upload.bin"
        upload_path = self.storage_service.save_upload(analysis_id, filename, payload)

        render_started = time.perf_counter()
        try:
            rendered_pages = self.pdf_service.render_document(upload_path)
        except ValueError as exc:
            if upload_path.exists():
                upload_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            if upload_path.exists():
                upload_path.unlink(missing_ok=True)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        render_ms = int((time.perf_counter() - render_started) * 1000)

        self.logger.info(
            "Document rendered",
            extra={"analysis_id": analysis_id, "page_count": len(rendered_pages)},
        )

        duplicate_started = time.perf_counter()
        duplicate_result = self.duplicate_service.check_document(
            payload=payload,
            pages=[page.image for page in rendered_pages],
        )
        duplicate_ms = int((time.perf_counter() - duplicate_started) * 1000)

        ocr_started = time.perf_counter()
        ocr_result = self.ocr_service.analyze_document([page.image for page in rendered_pages])
        ocr_ms = int((time.perf_counter() - ocr_started) * 1000)

        page_scores = {
            "ela": [],
            "srm": [],
            "noiseprint": [],
            "dino": [],
            "segmentation": [],
        }
        layer_timings_ms = {
            "ELA": 0,
            "SRM": 0,
            "Noiseprint": 0,
            "DINO_ViT": 0,
            "OCR_Anomaly": ocr_ms,
            "pHash_Duplicate": duplicate_ms,
            self._segmentation_layer_name(): 0,
        }
        page_results: list[PageResult] = []
        warnings = self._dedupe_warnings(duplicate_result.warnings + ocr_result.warnings)

        for rendered_page in rendered_pages:
            page_index = rendered_page.page_index
            features = self.preprocess_service.extract_cpu_features(rendered_page.image)

            original_filename = f"page_{page_index}_original.png"
            inference_filename = f"page_{page_index}_inference.png"
            self.artifact_service.save_image(
                analysis_id,
                original_filename,
                rgb_np_to_pil(features.original_rgb),
            )
            self.artifact_service.save_image(
                analysis_id,
                inference_filename,
                rgb_np_to_pil(features.inference_rgb),
            )

            page_engines = self.engine_service.analyze_page(
                features=features,
                analysis_id=analysis_id,
                page_index=page_index,
                artifact_service=self.artifact_service,
            )
            layer_timings_ms["ELA"] += page_engines.timings_ms["ela"]
            layer_timings_ms["SRM"] += page_engines.timings_ms["srm"]
            layer_timings_ms["Noiseprint"] += page_engines.timings_ms["noiseprint"]
            layer_timings_ms["DINO_ViT"] += page_engines.timings_ms["dino"]

            segmentation_tensor = self.preprocess_service.build_segmentation_tensor(
                features=features,
                srm_map=page_engines.srm_map,
                noiseprint_map=page_engines.noiseprint_map,
                dino_map=page_engines.dino_map,
            )
            segmentation_result = self.segmentation_service.segment_page(
                analysis_id=analysis_id,
                page_index=page_index,
                original_image=rendered_page.image,
                original_rgb=features.original_rgb,
                tensor=segmentation_tensor,
            )
            layer_timings_ms[self._segmentation_layer_name()] += segmentation_result.processing_ms

            combined_map = self.engine_service.build_combined_map(
                page_engines=page_engines,
                segmentation_probability_map=segmentation_result.probability_map,
            )
            combined_filename = f"page_{page_index}_combined.png"
            self.artifact_service.save_array(
                analysis_id,
                combined_filename,
                apply_heatmap(combined_map),
            )

            warnings = self._dedupe_warnings(warnings + segmentation_result.warnings)
            page_scores["ela"].append(page_engines.ela_score)
            page_scores["srm"].append(page_engines.srm_score)
            page_scores["noiseprint"].append(page_engines.noiseprint_score)
            page_scores["dino"].append(page_engines.dino_score)
            page_scores["segmentation"].append(segmentation_result.score)

            page_results.append(
                PageResult(
                    page_index=page_index,
                    width=rendered_page.width,
                    height=rendered_page.height,
                    artifacts=PageArtifacts(
                        original_url=self.artifact_service.url_for(analysis_id, original_filename),
                        mask_url=self.artifact_service.url_for(
                            analysis_id, segmentation_result.mask_filename
                        ),
                        overlay_url=self.artifact_service.url_for(
                            analysis_id, segmentation_result.overlay_filename
                        ),
                        ela_heatmap_url=self.artifact_service.url_for(
                            analysis_id, page_engines.ela_filename
                        ),
                        srm_heatmap_url=self.artifact_service.url_for(
                            analysis_id, page_engines.srm_filename
                        ),
                        noiseprint_heatmap_url=self.artifact_service.url_for(
                            analysis_id, page_engines.noiseprint_filename
                        ),
                        dino_heatmap_url=self.artifact_service.url_for(
                            analysis_id, page_engines.dino_filename
                        ),
                        combined_heatmap_url=self.artifact_service.url_for(
                            analysis_id, combined_filename
                        ),
                        contours_url=self.artifact_service.url_for(
                            analysis_id, segmentation_result.contours_filename
                        ),
                    ),
                    tampered_regions=[
                        TamperedRegion(**region) for region in segmentation_result.regions
                    ],
                )
            )

        engine_scores = EngineScores(
            ela_score=self._average(page_scores["ela"]),
            srm_score=self._average(page_scores["srm"]),
            noiseprint_score=self._average(page_scores["noiseprint"]),
            dino_vit_score=self._average(page_scores["dino"]),
            ocr_anomaly_score=ocr_result.score,
            phash_score=duplicate_result.phash_score,
            segmentation_score=self._average(page_scores["segmentation"]),
        )
        overall_score = forensic_risk_score(self.settings, engine_scores)
        verdict = verdict_for_score(self.settings, overall_score)
        created_at = datetime.now(timezone.utc)

        response = AnalysisResponse(
            analysis_id=analysis_id,
            filename=filename,
            document_type=document_type,
            submitter_id=submitter_id,
            tenant_id=tenant_id,
            session_ip_address=session_ip_address,
            session_geolocation=session_geolocation,
            is_human_reviewed=False,
            page_count=len(rendered_pages),
            device=self.model_loader.device,
            verdict=verdict,
            forensic_risk_score=overall_score,
            engine_scores=engine_scores,
            forensic_layers=self._build_forensic_layers(engine_scores, layer_timings_ms),
            extracted_metadata=self._build_extracted_metadata(
                upload_file=upload_file,
                upload_path=upload_path,
                payload=payload,
                rendered_pages=rendered_pages,
                render_ms=render_ms,
                ocr_result=ocr_result,
            ),
            device_fingerprint=self._build_device_fingerprint(
                user_agent=user_agent,
                session_ip_address=session_ip_address,
                tenant_id=tenant_id,
                submitter_id=submitter_id,
            ),
            rule_triggers=self._build_rule_triggers(
                created_at=created_at,
                engine_scores=engine_scores,
                overall_score=overall_score,
                duplicate_status=duplicate_result.duplicate_check["duplicate_status"],
                ocr_result=ocr_result,
                page_results=page_results,
                warnings=warnings,
            ),
            ocr_anomalies=[OCRAnomaly(**anomaly) for anomaly in ocr_result.anomalies],
            duplicate_check=DuplicateCheck(**duplicate_result.duplicate_check),
            pages=page_results,
            warnings=warnings,
            processing_time_ms=int((time.perf_counter() - started) * 1000),
            created_at=created_at,
        )

        payload_to_store = response.model_dump(mode="json")
        self.storage_service.store_analysis(payload_to_store, upload_path)
        self.duplicate_service.register_analysis(
            analysis_id=analysis_id,
            filename=filename,
            md5_hash=response.duplicate_check.md5_hash,
            phash=response.duplicate_check.phash,
            created_at=created_at,
        )
        return response

    def get_analysis(self, analysis_id: str) -> AnalysisResponse:
        payload = self.storage_service.get_analysis(analysis_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        return AnalysisResponse.model_validate(payload)

    def list_analyses(self, page: int, page_size: int) -> AnalysisHistoryResponse:
        items, total = self.storage_service.list_analyses(page=page, page_size=page_size)
        return AnalysisHistoryResponse(
            page=page,
            page_size=page_size,
            total=total,
            items=[AnalysisHistoryItem.model_validate(item) for item in items],
        )

    def delete_analysis(self, analysis_id: str) -> DeleteAnalysisResponse:
        deleted = self.storage_service.delete_analysis(analysis_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        return DeleteAnalysisResponse(analysis_id=analysis_id, deleted=True)

    def _build_forensic_layers(
        self,
        engine_scores: EngineScores,
        layer_timings_ms: dict[str, int],
    ) -> list[dict[str, Any]]:
        segmentation_name = self._segmentation_layer_name()
        layers = [
            ("ELA", engine_scores.ela_score),
            ("SRM", engine_scores.srm_score),
            ("Noiseprint", engine_scores.noiseprint_score),
            ("DINO_ViT", engine_scores.dino_vit_score),
            ("OCR_Anomaly", engine_scores.ocr_anomaly_score),
            ("pHash_Duplicate", engine_scores.phash_score),
            (segmentation_name, engine_scores.segmentation_score),
        ]
        return [
            {
                "layer_name": layer_name,
                "confidence_score": score,
                "processing_ms": int(layer_timings_ms.get(layer_name, 0)),
            }
            for layer_name, score in layers
        ]

    def _build_extracted_metadata(
        self,
        upload_file: UploadFile,
        upload_path,
        payload: bytes,
        rendered_pages: list[RenderedPage],
        render_ms: int,
        ocr_result: OCRAnalysisResult,
    ) -> list[dict[str, Any]]:
        guessed_mime = mimetypes.guess_type(upload_path.name)[0]
        mime_type = upload_file.content_type or guessed_mime or "application/octet-stream"
        filesystem_entry = {
            "metadata_type": "FILE_SYSTEM",
            "software_signature": None,
            "modification_date_raw": datetime.fromtimestamp(
                upload_path.stat().st_mtime,
                tz=timezone.utc,
            ).isoformat(),
            "raw_dump": {
                "mime_type": mime_type,
                "size_bytes": len(payload),
                "suffix": upload_path.suffix.lower(),
                "page_count": len(rendered_pages),
                "render_ms": render_ms,
                "page_dimensions": [
                    {"page_index": page.page_index, "width": page.width, "height": page.height}
                    for page in rendered_pages
                ],
            },
        }

        details: list[dict[str, Any]] = [filesystem_entry]
        if upload_path.suffix.lower() == ".pdf":
            pdf_metadata = self._extract_pdf_metadata(payload)
            filesystem_entry["software_signature"] = pdf_metadata.get("producer")
            filesystem_entry["modification_date_raw"] = (
                pdf_metadata.get("modification_date_raw")
                or filesystem_entry["modification_date_raw"]
            )
            details.append(
                {
                    "metadata_type": "PDF_TRAILER",
                    "software_signature": pdf_metadata.get("creator"),
                    "modification_date_raw": pdf_metadata.get("modification_date_raw"),
                    "raw_dump": pdf_metadata,
                }
            )
        else:
            image_metadata = self._extract_image_metadata(payload)
            filesystem_entry["software_signature"] = image_metadata.get("software_signature")
            filesystem_entry["camera_make"] = image_metadata.get("camera_make")
            filesystem_entry["camera_model"] = image_metadata.get("camera_model")
            filesystem_entry["modification_date_raw"] = (
                image_metadata.get("modification_date_raw")
                or filesystem_entry["modification_date_raw"]
            )
            filesystem_entry["gps_data"] = image_metadata.get("gps_data", {})

        details.append(
            {
                "metadata_type": "PIPELINE_CONTEXT",
                "software_signature": self.model_loader.selected_encoder,
                "raw_dump": {
                    "ocr_backend": ocr_result.backend_name,
                    "model_device": self.model_loader.device,
                    "checkpoint_input_channels": self.model_loader.checkpoint_input_channels,
                    "loaded_input_channels": self.model_loader.input_channels,
                    "dino_model": self.settings.dino_model_name,
                    "checkpoint_sha256": self.model_loader.checkpoint_sha256,
                    "calibration_loaded": self.settings.calibration_profile is not None,
                    "calibration_generated_at": self.settings.calibration_profile.generated_at.isoformat()
                    if self.settings.calibration_profile and self.settings.calibration_profile.generated_at
                    else None,
                    "calibration_sample_count": self.settings.calibration_profile.sample_count
                    if self.settings.calibration_profile
                    else None,
                },
            }
        )
        return details

    def _extract_pdf_metadata(self, payload: bytes) -> dict[str, Any]:
        text = payload[:500000].decode("latin-1", errors="ignore")
        patterns = {
            "producer": r"/Producer\s*\((.*?)\)",
            "creator": r"/Creator\s*\((.*?)\)",
            "title": r"/Title\s*\((.*?)\)",
            "modification_date_raw": r"/ModDate\s*\((.*?)\)",
        }
        metadata: dict[str, Any] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, flags=re.DOTALL)
            if match:
                metadata[key] = self._clean_pdf_literal(match.group(1))
        return metadata

    def _extract_image_metadata(self, payload: bytes) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "software_signature": None,
            "camera_make": None,
            "camera_model": None,
            "modification_date_raw": None,
            "gps_data": {},
        }
        try:
            with Image.open(io.BytesIO(payload)) as image:
                exif = image.getexif()
                metadata["software_signature"] = (
                    image.info.get("software")
                    or image.info.get("creator")
                    or exif.get(305)
                )
                metadata["camera_make"] = exif.get(271)
                metadata["camera_model"] = exif.get(272)
                metadata["modification_date_raw"] = exif.get(306) or exif.get(36867)
                gps_data = exif.get(34853)
                if isinstance(gps_data, dict):
                    metadata["gps_data"] = {
                        str(key): str(value) for key, value in gps_data.items()
                    }
        except Exception:
            return metadata
        return metadata

    def _build_device_fingerprint(
        self,
        user_agent: str | None,
        session_ip_address: str | None,
        tenant_id: str | None,
        submitter_id: str | None,
    ) -> dict[str, Any] | None:
        seed_parts = [
            tenant_id or "",
            submitter_id or "",
            session_ip_address or "",
            user_agent or "",
        ]
        if not any(seed_parts):
            return None

        normalized_user_agent = user_agent or None
        lower_user_agent = (user_agent or "").lower()
        device_hash = hashlib.sha256("|".join(seed_parts).encode("utf-8")).hexdigest()[:24]
        return {
            "device_hash": device_hash,
            "user_agent": normalized_user_agent,
            "browser": self._parse_browser(lower_user_agent),
            "os": self._parse_os(lower_user_agent),
            "is_known_fraud_device": any(
                marker in lower_user_agent
                for marker in ("headless", "selenium", "playwright", "puppeteer", "phantomjs")
            ),
        }

    def _build_rule_triggers(
        self,
        created_at: datetime,
        engine_scores: EngineScores,
        overall_score: float,
        duplicate_status: DuplicateStatus,
        ocr_result: OCRAnalysisResult,
        page_results: list[PageResult],
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        triggers: list[dict[str, Any]] = []
        total_regions = sum(len(page.tampered_regions) for page in page_results)
        high_signal_count = sum(
            score >= 0.55
            for score in (
                engine_scores.ela_score,
                engine_scores.srm_score,
                engine_scores.noiseprint_score,
                engine_scores.dino_vit_score,
                engine_scores.ocr_anomaly_score,
                engine_scores.segmentation_score,
            )
        )

        if duplicate_status == DuplicateStatus.EXACT_DUPLICATE:
            triggers.append(
                {
                    "policy_id": "EXACT_DUPLICATE_DOCUMENT",
                    "severity": "CRITICAL",
                    "triggered_at": created_at.isoformat(),
                }
            )
        elif duplicate_status == DuplicateStatus.NEAR_DUPLICATE:
            triggers.append(
                {
                    "policy_id": "NEAR_DUPLICATE_DOCUMENT",
                    "severity": "HIGH",
                    "triggered_at": created_at.isoformat(),
                }
            )

        if ocr_result.anomalies and engine_scores.ocr_anomaly_score >= 0.20:
            triggers.append(
                {
                    "policy_id": "OCR_CONTENT_INCONSISTENCY",
                    "severity": "HIGH" if engine_scores.ocr_anomaly_score >= 0.45 else "MEDIUM",
                    "triggered_at": created_at.isoformat(),
                }
            )

        if total_regions > 0:
            triggers.append(
                {
                    "policy_id": "LOCALIZED_PIXEL_TAMPER",
                    "severity": "HIGH" if engine_scores.segmentation_score >= 0.60 else "MEDIUM",
                    "triggered_at": created_at.isoformat(),
                }
            )

        if overall_score >= 0.85 or high_signal_count >= 3:
            triggers.append(
                {
                    "policy_id": "MULTI_ENGINE_CONSENSUS",
                    "severity": "CRITICAL" if overall_score >= 0.92 else "HIGH",
                    "triggered_at": created_at.isoformat(),
                }
            )

        if warnings:
            triggers.append(
                {
                    "policy_id": "PIPELINE_WARNING",
                    "severity": "MEDIUM",
                    "triggered_at": created_at.isoformat(),
                }
            )

        return triggers

    def _segmentation_layer_name(self) -> str:
        encoder = self.model_loader.selected_encoder or "unavailable"
        return f"Segmentation_{encoder}"

    @staticmethod
    def _parse_browser(user_agent: str) -> str | None:
        if not user_agent:
            return None
        if "edg/" in user_agent:
            return "Edge"
        if "opr/" in user_agent or "opera" in user_agent:
            return "Opera"
        if "chrome/" in user_agent and "edg/" not in user_agent:
            return "Chrome"
        if "firefox/" in user_agent:
            return "Firefox"
        if "safari/" in user_agent and "chrome/" not in user_agent:
            return "Safari"
        return None

    @staticmethod
    def _parse_os(user_agent: str) -> str | None:
        if not user_agent:
            return None
        if "windows" in user_agent:
            return "Windows"
        if "android" in user_agent:
            return "Android"
        if "iphone" in user_agent or "ipad" in user_agent or "ios" in user_agent:
            return "iOS"
        if "mac os x" in user_agent or "macintosh" in user_agent:
            return "macOS"
        if "linux" in user_agent:
            return "Linux"
        return None

    @staticmethod
    def _clean_pdf_literal(value: str) -> str:
        return value.replace("\\(", "(").replace("\\)", ")").replace("\\n", " ").strip()

    @staticmethod
    def _average(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _dedupe_warnings(values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value not in seen:
                seen.add(value)
                deduped.append(value)
        return deduped
