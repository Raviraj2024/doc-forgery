from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from PIL import Image

from app.core.config import Settings
from app.services.pdf_service import RenderedPage

SUPPORTED_DOCUMENT_TYPES = {
    "invoice",
    "receipt",
    "bank_statement",
    "legal_filing",
    "affidavit",
    "agreement",
    "certificate",
    "id_document",
    "payslip",
    "medical_record",
    "other",
}

_FILENAME_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("invoice", ("invoice", "inv", "bill", "gst")),
    ("receipt", ("receipt", "rcpt")),
    ("bank_statement", ("bank statement", "statement", "passbook")),
    ("affidavit", ("affidavit",)),
    ("agreement", ("agreement", "contract", "deed", "mou")),
    ("legal_filing", ("petition", "notice", "application", "court", "judgment", "order", "case")),
    ("certificate", ("certificate", "licence", "license")),
    ("id_document", ("aadhaar", "aadhar", "pan", "passport", "voter", "license", "licence", "dl")),
    ("payslip", ("payslip", "salary slip", "salary")),
    ("medical_record", ("medical", "prescription", "lab report", "diagnostic")),
]


@dataclass(slots=True)
class DocumentRoutingDecision:
    document_type: str
    confidence: float
    provider: str
    source: str
    language_code: str
    warnings: list[str] = field(default_factory=list)
    page_texts: list[str] | None = None
    ocr_backend_name: str | None = None


class DocumentRoutingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)

    def inspect_document(
        self,
        *,
        upload_path: Path,
        filename: str,
        rendered_pages: list[RenderedPage],
        requested_document_type: str | None = None,
    ) -> DocumentRoutingDecision:
        warnings: list[str] = []
        requested_type: str | None = None
        heuristic_type, heuristic_confidence = self.classify_from_filename(filename)
        language_code = self.detect_language_code(filename)

        if requested_document_type:
            candidate = self.normalise_document_type(requested_document_type)
            if candidate in SUPPORTED_DOCUMENT_TYPES:
                requested_type = candidate

        nemotron_type: str | None = None
        nemotron_confidence = 0.0
        if rendered_pages and self.settings.openrouter_api_key:
            try:
                nemotron_type, nemotron_confidence, model_language = self.classify_with_nemotron(
                    rendered_pages[0].image
                )
                if model_language:
                    language_code = self.merge_language_hint(language_code, model_language)
            except Exception as exc:
                warning = f"Nemotron classification unavailable; falling back to filename routing ({exc})."
                warnings.append(warning)
                self.logger.warning(warning)

        document_type = requested_type or nemotron_type or heuristic_type
        confidence = 1.0 if requested_type else nemotron_confidence if nemotron_type else heuristic_confidence
        provider = self.select_provider(
            filename=filename,
            page_count=len(rendered_pages),
            language_code=language_code,
            document_type=document_type,
        )
        source = "requested" if requested_type else "nemotron" if nemotron_type else "filename"

        page_texts: list[str] | None = None
        ocr_backend_name: str | None = None
        if provider == "sarvam" and self.settings.sarvam_api_key:
            try:
                page_texts = self.extract_with_sarvam(
                    upload_path=upload_path,
                    language_code=language_code,
                )
                ocr_backend_name = "sarvam_document_intelligence"
                text_type, text_confidence = self.classify_from_text("\n".join(page_texts))
                if text_confidence >= confidence:
                    document_type = text_type
                    confidence = text_confidence
                    source = "sarvam_text"
            except Exception as exc:
                warning = f"Sarvam extraction unavailable; falling back to local OCR ({exc})."
                warnings.append(warning)
                self.logger.warning(warning)
                provider = "local"
        elif provider == "sarvam":
            warnings.append("Sarvam key not configured; using local OCR fallback.")
            provider = "local"
        elif (
            provider == "nemotron"
            and self.settings.openrouter_api_key
            and (
                self.settings.document_router_provider == "nemotron"
                or self.should_use_nemotron_text_extraction(
                    filename=filename,
                    page_count=len(rendered_pages),
                    document_type=document_type,
                )
            )
        ):
            try:
                page_texts = self.extract_with_nemotron_pages(rendered_pages)
                ocr_backend_name = "nemotron_vision"
                text_type, text_confidence = self.classify_from_text("\n".join(page_texts))
                if text_confidence >= confidence:
                    document_type = text_type
                    confidence = text_confidence
                    source = "nemotron_text"
            except Exception as exc:
                warning = f"Nemotron text extraction unavailable; falling back to local OCR ({exc})."
                warnings.append(warning)
                self.logger.warning(warning)
                provider = "local"

        return DocumentRoutingDecision(
            document_type=document_type,
            confidence=confidence,
            provider=provider,
            source=source,
            language_code=language_code,
            warnings=warnings,
            page_texts=page_texts,
            ocr_backend_name=ocr_backend_name,
        )

    def classify_with_nemotron(
        self,
        image: Image.Image,
    ) -> tuple[str, float, str | None]:
        if not self.settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")

        payload = {
            "model": self.settings.openrouter_model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Classify this document. Return strict JSON with keys document_type, "
                        "language_hint, confidence. document_type must be one of: "
                        "invoice, receipt, bank_statement, legal_filing, affidavit, "
                        "agreement, certificate, id_document, payslip, medical_record, other."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply only with JSON."},
                        {
                            "type": "image_url",
                            "image_url": {"url": self.image_to_data_url(image)},
                        },
                    ],
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.settings.document_router_timeout_seconds) as client:
            response = client.post(
                self.settings.openrouter_base_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

        body = response.json()
        content = body["choices"][0]["message"]["content"]
        parsed = self.parse_json_payload(content)
        document_type = self.normalise_document_type(parsed.get("document_type"))
        if document_type not in SUPPORTED_DOCUMENT_TYPES:
            raise RuntimeError(f"Unsupported Nemotron document type: {parsed.get('document_type')}")

        confidence = self.coerce_confidence(parsed.get("confidence"))
        language_hint = parsed.get("language_hint")
        return document_type, max(0.0, min(1.0, confidence)), language_hint

    def extract_with_sarvam(self, *, upload_path: Path, language_code: str) -> list[str]:
        if not self.settings.sarvam_api_key:
            raise RuntimeError("SARVAM_API_KEY is not configured.")

        headers = {
            "api-subscription-key": self.settings.sarvam_api_key,
            "Content-Type": "application/json",
        }
        filename = upload_path.name
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        payload = upload_path.read_bytes()

        with httpx.Client(
            timeout=self.settings.document_router_timeout_seconds,
            follow_redirects=True,
        ) as client:
            create_response = client.post(
                f"{self.settings.sarvam_base_url}/doc-digitization/job/v1",
                headers=headers,
                json={
                    "job_parameters": {
                        "language": language_code,
                        "output_format": "md",
                    }
                },
            )
            create_response.raise_for_status()
            job_id = create_response.json()["job_id"]

            upload_response = client.post(
                f"{self.settings.sarvam_base_url}/doc-digitization/job/v1/upload-files",
                headers=headers,
                json={"job_id": job_id, "files": [filename]},
            )
            upload_response.raise_for_status()
            upload_url = upload_response.json()["upload_urls"][filename]["file_url"]

            file_response = client.put(
                upload_url,
                content=payload,
                headers={
                    "x-ms-blob-type": "BlockBlob",
                    "Content-Type": mime_type,
                },
            )
            file_response.raise_for_status()

            start_response = client.post(
                f"{self.settings.sarvam_base_url}/doc-digitization/job/v1/{job_id}/start",
                headers={"api-subscription-key": self.settings.sarvam_api_key},
            )
            start_response.raise_for_status()

            deadline = time.monotonic() + self.settings.sarvam_poll_timeout_seconds
            while True:
                status_response = client.get(
                    f"{self.settings.sarvam_base_url}/doc-digitization/job/v1/{job_id}/status",
                    headers={"api-subscription-key": self.settings.sarvam_api_key},
                )
                status_response.raise_for_status()
                state = status_response.json().get("job_state")
                if state in {"Completed", "PartiallyCompleted"}:
                    break
                if state == "Failed":
                    raise RuntimeError("Sarvam document intelligence job failed.")
                if time.monotonic() >= deadline:
                    raise RuntimeError("Sarvam document intelligence timed out.")
                time.sleep(self.settings.sarvam_poll_interval_seconds)

            download_response = client.post(
                f"{self.settings.sarvam_base_url}/doc-digitization/job/v1/{job_id}/download-files",
                headers={"api-subscription-key": self.settings.sarvam_api_key},
            )
            download_response.raise_for_status()
            entry = next(iter(download_response.json()["download_urls"].values()))
            archive_response = client.get(entry["file_url"])
            archive_response.raise_for_status()

        return self.page_texts_from_sarvam_archive(archive_response.content)

    def extract_with_nemotron_pages(self, rendered_pages: list[RenderedPage]) -> list[str]:
        if not self.settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        page_texts: list[str] = []
        max_pages = len(rendered_pages)

        with httpx.Client(timeout=self.settings.document_router_timeout_seconds) as client:
            for rendered_page in rendered_pages[:max_pages]:
                payload = {
                    "model": self.settings.openrouter_model,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Extract only the visible text from this document page. "
                                "Preserve line breaks. Do not explain anything."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Return only the text from the page."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": self.image_to_data_url(rendered_page.image)},
                                },
                            ],
                        },
                    ],
                }
                response = client.post(
                    self.settings.openrouter_base_url,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                page_texts.append(content.strip())

        if len(rendered_pages) > max_pages:
            page_texts.extend(["" for _ in rendered_pages[max_pages:]])

        return page_texts

    def page_texts_from_sarvam_archive(self, archive_bytes: bytes) -> list[str]:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
        page_texts: dict[int, str] = {}
        markdown_text = ""

        for name in archive.namelist():
            lower_name = name.lower()
            if lower_name.endswith(".md"):
                markdown_text = archive.read(name).decode("utf-8", errors="ignore").strip()
                continue

            if lower_name.startswith("metadata/page_") and lower_name.endswith(".json"):
                payload = json.loads(archive.read(name).decode("utf-8"))
                page_num = int(payload.get("page_num") or 1)
                blocks = sorted(
                    payload.get("blocks", []),
                    key=lambda block: block.get("reading_order", 0),
                )
                text = "\n".join(
                    str(block.get("text", "")).strip()
                    for block in blocks
                    if str(block.get("text", "")).strip()
                ).strip()
                page_texts[page_num] = text

        if page_texts:
            return [page_texts[index] for index in sorted(page_texts)]
        if markdown_text:
            return [markdown_text]
        return [""]

    def select_provider(
        self,
        *,
        filename: str,
        page_count: int,
        language_code: str,
        document_type: str,
    ) -> str:
        if self.settings.document_router_provider == "sarvam":
            return "sarvam"
        if self.settings.document_router_provider == "nemotron":
            return "nemotron"

        lower_name = filename.lower()
        is_pdf = lower_name.endswith(".pdf")
        is_multi_page_pdf = is_pdf and page_count > 1
        has_non_latin = self.contains_non_latin(filename)
        if has_non_latin or language_code != "en-IN":
            return "sarvam"
        if is_pdf:
            if page_count == 1 and document_type in {"id_document", "certificate"}:
                return "nemotron"
            return "sarvam"
        if document_type in {"legal_filing", "affidavit", "agreement", "certificate", "medical_record"}:
            return "sarvam"
        if is_multi_page_pdf:
            return "sarvam"
        return "nemotron"

    def should_use_nemotron_text_extraction(
        self,
        *,
        filename: str,
        page_count: int,
        document_type: str,
    ) -> bool:
        lower_name = filename.lower()
        if not lower_name.endswith(".pdf"):
            return True
        return page_count == 1 and document_type in {"id_document", "certificate"}

    def classify_from_filename(self, filename: str) -> tuple[str, float]:
        lowered = filename.lower().replace("_", " ")
        for document_type, keywords in _FILENAME_PATTERNS:
            if any(keyword in lowered for keyword in keywords):
                return document_type, 0.72
        return "other", 0.20

    def classify_from_text(self, text: str) -> tuple[str, float]:
        lowered = text.lower()
        if any(token in lowered for token in ("invoice no", "tax invoice", "gst", "amount due")):
            return "invoice", 0.88
        if any(token in lowered for token in ("receipt no", "receipt", "payment received")):
            return "receipt", 0.82
        if any(token in lowered for token in ("account statement", "opening balance", "closing balance")):
            return "bank_statement", 0.86
        if any(token in lowered for token in ("affidavit", "solemnly affirm")):
            return "affidavit", 0.84
        if any(token in lowered for token in ("agreement", "party of the first part", "contract")):
            return "agreement", 0.82
        if any(token in lowered for token in ("certificate", "certified that")):
            return "certificate", 0.80
        if any(token in lowered for token in ("prescription", "diagnosis", "patient name")):
            return "medical_record", 0.78
        if any(token in lowered for token in ("court", "petition", "writ", "notice", "order")):
            return "legal_filing", 0.80
        if any(token in lowered for token in ("salary", "earnings", "deductions", "payslip")):
            return "payslip", 0.82
        if any(token in lowered for token in ("aadhaar", "passport", "date of birth", "identity")):
            return "id_document", 0.76
        return "other", 0.25

    @staticmethod
    def image_to_data_url(image: Image.Image) -> str:
        canvas = image.convert("RGB")
        max_side = max(canvas.size)
        if max_side > 1280:
            scale = 1280 / max_side
            canvas = canvas.resize(
                (max(1, int(canvas.width * scale)), max(1, int(canvas.height * scale))),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def normalise_document_type(value: str | None) -> str:
        if not value:
            return "other"
        normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        return normalized or "other"

    @staticmethod
    def parse_json_payload(content: str) -> dict[str, object]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        return json.loads(cleaned)

    @staticmethod
    def coerce_confidence(value: object) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "high":
                return 0.85
            if lowered == "medium":
                return 0.65
            if lowered == "low":
                return 0.35
            try:
                return float(lowered)
            except ValueError:
                return 0.0
        return 0.0

    @staticmethod
    def contains_non_latin(value: str) -> bool:
        return any(ord(char) > 127 for char in value)

    @staticmethod
    def merge_language_hint(current: str, model_hint: str | None) -> str:
        if not model_hint:
            return current
        hint = model_hint.lower()
        if hint.startswith(("hi", "mr")):
            return "hi-IN"
        if hint.startswith("bn"):
            return "bn-IN"
        if hint.startswith("ta"):
            return "ta-IN"
        if hint.startswith("te"):
            return "te-IN"
        if hint.startswith("gu"):
            return "gu-IN"
        if hint.startswith("kn"):
            return "kn-IN"
        if hint.startswith("ml"):
            return "ml-IN"
        if hint.startswith("pa"):
            return "pa-IN"
        if hint.startswith("od"):
            return "od-IN"
        return "en-IN"

    @staticmethod
    def detect_language_code(filename: str) -> str:
        for char in filename:
            codepoint = ord(char)
            if 0x0900 <= codepoint <= 0x097F:
                return "hi-IN"
            if 0x0980 <= codepoint <= 0x09FF:
                return "bn-IN"
            if 0x0A00 <= codepoint <= 0x0A7F:
                return "pa-IN"
            if 0x0A80 <= codepoint <= 0x0AFF:
                return "gu-IN"
            if 0x0B00 <= codepoint <= 0x0B7F:
                return "od-IN"
            if 0x0B80 <= codepoint <= 0x0BFF:
                return "ta-IN"
            if 0x0C00 <= codepoint <= 0x0C7F:
                return "te-IN"
            if 0x0C80 <= codepoint <= 0x0CFF:
                return "kn-IN"
            if 0x0D00 <= codepoint <= 0x0D7F:
                return "ml-IN"
        return "en-IN"
