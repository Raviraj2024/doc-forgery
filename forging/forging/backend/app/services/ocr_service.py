from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from app.core.config import Settings
from app.schemas.responses import OCRAnomalyType
from app.utils.scoring import clamp01


@dataclass(slots=True)
class OCRAnalysisResult:
    anomalies: list[dict[str, object]]
    score: float
    warnings: list[str]
    page_texts: list[str]
    backend_name: str | None


class OCRService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)
        self.backend_name: str | None = None
        self.reader = self._initialise_backend()

    def _initialise_backend(self) -> object | None:
        try:
            from paddleocr import PaddleOCR

            self.backend_name = "paddleocr"
            return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except Exception:
            try:
                import easyocr

                self.backend_name = "easyocr"
                return easyocr.Reader(["en"], gpu=False)
            except Exception:
                self.backend_name = None
                return None

    def analyze_document(self, pages: list[Image.Image]) -> OCRAnalysisResult:
        warnings: list[str] = []
        anomalies: list[dict[str, object]] = []
        page_texts: list[str] = []

        if self.reader is None or self.backend_name is None:
            warning = "OCR backend unavailable; OCR anomaly score set to 0.0."
            warnings.append(warning)
            anomalies.append(
                {
                    "type": OCRAnomalyType.OCR_WARNING,
                    "description": warning,
                    "page_index": None,
                }
            )
            return OCRAnalysisResult(
                anomalies=anomalies,
                score=0.0,
                warnings=warnings,
                page_texts=["" for _ in pages],
                backend_name=None,
            )

        for page_number, page in enumerate(pages, start=1):
            try:
                page_texts.append(self._extract_page_text(page))
            except Exception as exc:
                warning = f"OCR extraction failed on page {page_number}: {exc}"
                warnings.append(warning)
                page_texts.append("")

        if not any(text.strip() for text in page_texts):
            warning = "OCR produced no text; OCR anomaly score set to 0.0."
            warnings.append(warning)
            anomalies.append(
                {
                    "type": OCRAnomalyType.OCR_WARNING,
                    "description": warning,
                    "page_index": None,
                }
            )
            return OCRAnalysisResult(
                anomalies=anomalies,
                score=0.0,
                warnings=warnings,
                page_texts=page_texts,
                backend_name=self.backend_name,
            )

        anomalies.extend(self._detect_amount_mismatch(page_texts))
        anomalies.extend(self._detect_duplicate_references(page_texts))
        anomalies.extend(self._detect_suspicious_keywords(page_texts))
        anomalies.extend(self._detect_invalid_dates(page_texts))

        score = self._score_anomalies(anomalies)
        return OCRAnalysisResult(
            anomalies=anomalies,
            score=score,
            warnings=warnings,
            page_texts=page_texts,
            backend_name=self.backend_name,
        )

    def _extract_page_text(self, image: Image.Image) -> str:
        image_array = np.array(image.convert("RGB"))

        if self.backend_name == "paddleocr":
            result = self.reader.ocr(image_array, cls=True)
            texts: list[str] = []
            for block in result or []:
                for line in block or []:
                    if len(line) >= 2:
                        texts.append(str(line[1][0]))
            return "\n".join(texts)

        if self.backend_name == "easyocr":
            result = self.reader.readtext(image_array)
            return "\n".join(str(item[1]) for item in result)

        return ""

    def _detect_amount_mismatch(self, page_texts: list[str]) -> list[dict[str, object]]:
        anomalies: list[dict[str, object]] = []
        currency_pattern = re.compile(
            r"(?:USD|INR|EUR|GBP|AUD|CAD|Rs\.?|[$₹€£])?\s*([0-9]{1,3}(?:[, ][0-9]{3})*(?:\.[0-9]{2})|[0-9]+\.[0-9]{2})",
            re.IGNORECASE,
        )
        total_pattern = re.compile(r"\b(total|grand total|amount due|net payable)\b", re.IGNORECASE)

        for page_index, text in enumerate(page_texts, start=1):
            totals: list[float] = []
            line_items: list[float] = []
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                matches = [float(match.group(1).replace(",", "").replace(" ", "")) for match in currency_pattern.finditer(line)]
                if not matches:
                    continue
                if total_pattern.search(line):
                    totals.extend(matches)
                else:
                    if len(matches) == 1:
                        line_items.extend(matches)
            if totals and len(line_items) >= 2:
                candidate_total = max(totals)
                candidate_sum = sum(line_items[: min(len(line_items), 20)])
                tolerance = max(1.0, candidate_total * 0.02)
                if abs(candidate_sum - candidate_total) > tolerance:
                    anomalies.append(
                        {
                            "type": OCRAnomalyType.AMOUNT_MISMATCH,
                            "description": (
                                f"Detected amount mismatch on page {page_index}: "
                                f"line items sum to {candidate_sum:.2f} while total is {candidate_total:.2f}."
                            ),
                            "page_index": page_index,
                        }
                    )
        return anomalies

    def _detect_duplicate_references(self, page_texts: list[str]) -> list[dict[str, object]]:
        anomalies: list[dict[str, object]] = []
        reference_pattern = re.compile(
            r"\b(?:ref(?:erence)?|invoice|txn|transaction|receipt|order)[\s:#-]*([A-Z0-9-]{4,})\b",
            re.IGNORECASE,
        )
        seen: dict[str, int] = {}

        for page_index, text in enumerate(page_texts, start=1):
            for match in reference_pattern.finditer(text):
                code = match.group(1).upper()
                if code in seen:
                    anomalies.append(
                        {
                            "type": OCRAnomalyType.DUPLICATE_REFERENCE,
                            "description": (
                                f"Reference code {code} appears on pages {seen[code]} and {page_index}."
                            ),
                            "page_index": page_index,
                        }
                    )
                else:
                    seen[code] = page_index
        return anomalies

    def _detect_suspicious_keywords(self, page_texts: list[str]) -> list[dict[str, object]]:
        anomalies: list[dict[str, object]] = []
        keywords = [
            "edited",
            "corrected",
            "manually adjusted",
            "revised",
            "void",
            "sample",
            "copy",
            "duplicate",
        ]
        for page_index, text in enumerate(page_texts, start=1):
            lower_text = text.lower()
            for keyword in keywords:
                if keyword in lower_text:
                    anomalies.append(
                        {
                            "type": OCRAnomalyType.SUSPICIOUS_KEYWORD,
                            "description": f"Keyword '{keyword}' detected in OCR text.",
                            "page_index": page_index,
                        }
                    )
        return anomalies

    def _detect_invalid_dates(self, page_texts: list[str]) -> list[dict[str, object]]:
        anomalies: list[dict[str, object]] = []
        date_pattern = re.compile(
            r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\b"
        )
        supported_formats = [
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%m-%d-%Y",
            "%Y/%m/%d",
            "%Y-%m-%d",
            "%b %d, %Y",
            "%B %d, %Y",
            "%d/%m/%y",
            "%m/%d/%y",
        ]
        now = datetime.now(timezone.utc)
        future_year_limit = now.year + 1

        for page_index, text in enumerate(page_texts, start=1):
            for raw_value in date_pattern.findall(text):
                parsed = None
                for fmt in supported_formats:
                    try:
                        parsed = datetime.strptime(raw_value, fmt).replace(
                            tzinfo=timezone.utc
                        )
                        break
                    except ValueError:
                        continue
                if parsed is None:
                    anomalies.append(
                        {
                            "type": OCRAnomalyType.INVALID_DATE,
                            "description": f"Unparseable date detected: {raw_value}.",
                            "page_index": page_index,
                        }
                    )
                    continue
                if parsed.year < 2000 or parsed.year > future_year_limit:
                    anomalies.append(
                        {
                            "type": OCRAnomalyType.INVALID_DATE,
                            "description": f"Implausible date detected: {raw_value}.",
                            "page_index": page_index,
                        }
                    )
        return anomalies

    def _score_anomalies(self, anomalies: list[dict[str, object]]) -> float:
        weights = {
            OCRAnomalyType.AMOUNT_MISMATCH: 0.35,
            OCRAnomalyType.DUPLICATE_REFERENCE: 0.25,
            OCRAnomalyType.SUSPICIOUS_KEYWORD: 0.20,
            OCRAnomalyType.INVALID_DATE: 0.20,
            OCRAnomalyType.OCR_WARNING: 0.0,
        }
        score = sum(weights.get(anomaly["type"], 0.0) for anomaly in anomalies)
        return clamp01(score)
