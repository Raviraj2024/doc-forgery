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

    def analyze_document(
        self,
        pages: list[Image.Image],
        document_type: str | None = None,
        page_texts_override: list[str] | None = None,
        backend_name_override: str | None = None,
    ) -> OCRAnalysisResult:
        warnings: list[str] = []
        anomalies: list[dict[str, object]] = []
        page_texts: list[str] = []
        backend_name = backend_name_override or self.backend_name
        normalized_document_type = self._normalise_document_type(document_type)

        if page_texts_override is not None:
            page_texts = [text or "" for text in page_texts_override]
        elif self.reader is None or self.backend_name is None:
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
        else:
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
                backend_name=backend_name,
            )

        if normalized_document_type in {"invoice", "receipt", "payslip", "other"}:
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
            backend_name=backend_name,
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
        total_pattern = re.compile(
            r"\b(grand\s+total|net\s+payable|amount\s+due|invoice\s+total|total\s+amount|total)\b",
            re.IGNORECASE,
        )
        grand_total_pattern = re.compile(
            r"\b(grand\s+total|net\s+payable|amount\s+due|invoice\s+total|total\s+amount)\b",
            re.IGNORECASE,
        )
        subtotal_pattern = re.compile(
            r"\b(sub\s*total|subtotal|taxable\s+value|line\s+item\s+total)\b",
            re.IGNORECASE,
        )
        header_pattern = re.compile(
            r"\b(sr\.?\s*no|description|hsn|sac|quantity|qty|unit|rate|gst|amount|buyer|seller|invoice\s+information)\b",
            re.IGNORECASE,
        )
        non_item_pattern = re.compile(
            r"\b(invoice\s+no|invoice\s+number|invoice\s+date|due\s+date|purchase\s+order|payment\s+terms|gstin|contact|email|address|phone|currency)\b",
            re.IGNORECASE,
        )

        for page_index, text in enumerate(page_texts, start=1):
            line_subtotals: list[float] = []
            line_gross_totals: list[float] = []
            declared_subtotals: list[float] = []
            declared_grand_totals: list[float] = []
            declared_generic_totals: list[float] = []

            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue

                amount_tokens = self._extract_amount_tokens(line)
                if not amount_tokens:
                    continue

                amounts = [float(token["value"]) for token in amount_tokens]
                if total_pattern.search(line) or subtotal_pattern.search(line):
                    candidate_total = max(amounts)
                    if subtotal_pattern.search(line):
                        declared_subtotals.append(candidate_total)
                    elif grand_total_pattern.search(line):
                        declared_grand_totals.append(candidate_total)
                    else:
                        declared_generic_totals.append(candidate_total)
                    continue

                if header_pattern.search(line) and len(amount_tokens) <= 1:
                    continue
                if non_item_pattern.search(line):
                    continue

                item = self._parse_invoice_line_item(line, amount_tokens)
                if item is None:
                    continue

                line_subtotals.append(item["subtotal"])
                line_gross_totals.append(item["gross_total"])

                expected_subtotal = item["quantity"] * item["unit_price"]
                tolerance = max(1.0, abs(item["subtotal"]) * 0.005)
                if abs(expected_subtotal - item["subtotal"]) > tolerance:
                    anomalies.append(
                        {
                            "type": OCRAnomalyType.AMOUNT_MISMATCH,
                            "description": (
                                f"Line item arithmetic mismatch on page {page_index}: "
                                f"{item['quantity']:.2f} x {item['unit_price']:.2f} = "
                                f"{expected_subtotal:.2f}, but the row amount is {item['subtotal']:.2f}."
                            ),
                            "page_index": page_index,
                        }
                    )

            if not line_subtotals:
                continue

            subtotal_sum = sum(line_subtotals)
            gross_sum = sum(line_gross_totals)
            candidate_subtotal = max(declared_subtotals) if declared_subtotals else None
            candidate_grand_total = (
                max(declared_grand_totals)
                if declared_grand_totals
                else max(declared_generic_totals)
                if declared_generic_totals
                else None
            )

            if candidate_subtotal is not None:
                anomalies.extend(
                    self._compare_invoice_total(
                        page_index=page_index,
                        label="subtotal",
                        expected=subtotal_sum,
                        declared=candidate_subtotal,
                    )
                )

            if candidate_grand_total is not None:
                expected_total = gross_sum if abs(gross_sum - subtotal_sum) > 0.01 else subtotal_sum
                anomalies.extend(
                    self._compare_invoice_total(
                        page_index=page_index,
                        label="grand total",
                        expected=expected_total,
                        declared=candidate_grand_total,
                    )
                )
        return anomalies

    def _extract_amount_tokens(self, line: str) -> list[dict[str, float | int | str]]:
        amount_pattern = re.compile(
            r"(?<![A-Z0-9])(?:USD|INR|EUR|GBP|AUD|CAD|Rs\.?|[$₹€£])?\s*([0-9]{1,3}(?:[, ][0-9]{3})+(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)(?![A-Z0-9])",
            re.IGNORECASE,
        )
        tokens: list[dict[str, float | int | str]] = []
        for match in amount_pattern.finditer(line):
            raw_value = match.group(1).strip()
            compact = raw_value.replace(",", "").replace(" ", "")
            if not compact or compact == ".":
                continue
            start, end = match.span(1)
            if self._is_percentage_token(line, end):
                continue
            if self._looks_like_date_fragment(line, start, end):
                continue
            try:
                value = float(compact)
            except ValueError:
                continue
            tokens.append(
                {
                    "value": value,
                    "raw": raw_value,
                    "start": start,
                    "end": end,
                }
            )
        return tokens

    def _parse_invoice_line_item(
        self,
        line: str,
        amount_tokens: list[dict[str, float | int | str]],
    ) -> dict[str, float] | None:
        values = [float(token["value"]) for token in amount_tokens]
        if len(values) < 3:
            return None

        if self._line_starts_with_serial_number(line, amount_tokens[0]):
            values = values[1:]
        if len(values) < 3:
            return None

        while len(values) > 3 and values[0].is_integer() and values[0] >= 100000:
            values = values[1:]

        quantity, unit_price, subtotal = values[-3], values[-2], values[-1]
        if quantity <= 0 or unit_price <= 0 or subtotal <= 0:
            return None
        if quantity > 100000 or unit_price < 1:
            return None

        gst_rate = self._extract_gst_rate(line)
        gross_total = subtotal * (1.0 + gst_rate / 100.0) if gst_rate is not None else subtotal
        return {
            "quantity": quantity,
            "unit_price": unit_price,
            "subtotal": subtotal,
            "gross_total": gross_total,
        }

    def _compare_invoice_total(
        self,
        *,
        page_index: int,
        label: str,
        expected: float,
        declared: float,
    ) -> list[dict[str, object]]:
        tolerance = max(1.0, abs(declared) * 0.005)
        if abs(expected - declared) <= tolerance:
            return []
        return [
            {
                "type": OCRAnomalyType.AMOUNT_MISMATCH,
                "description": (
                    f"Invoice {label} mismatch on page {page_index}: "
                    f"computed line items total {expected:.2f}, but declared {label} is {declared:.2f}."
                ),
                "page_index": page_index,
            }
        ]

    @staticmethod
    def _is_percentage_token(line: str, end: int) -> bool:
        return line[end : end + 2].lstrip().startswith("%")

    @staticmethod
    def _looks_like_date_fragment(line: str, start: int, end: int) -> bool:
        before = line[max(0, start - 1) : start]
        after = line[end : end + 1]
        return before in {"/", "-"} or after in {"/", "-"}

    @staticmethod
    def _line_starts_with_serial_number(
        line: str,
        token: dict[str, float | int | str],
    ) -> bool:
        value = float(token["value"])
        return bool(
            value.is_integer()
            and 0 < value <= 100
            and int(token["start"]) <= max(4, len(line) // 10)
        )

    @staticmethod
    def _extract_gst_rate(line: str) -> float | None:
        match = re.search(r"\b(\d{1,2}(?:\.\d{1,2})?)\s*%", line)
        if not match:
            return None
        rate = float(match.group(1))
        return rate if 0 <= rate <= 50 else None

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
            r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b"
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
            "%d %b %Y",
            "%d %B %Y",
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
                if parsed.date() > now.date():
                    anomalies.append(
                        {
                            "type": OCRAnomalyType.INVALID_DATE,
                            "description": f"Future date detected: {raw_value}.",
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

    @staticmethod
    def _normalise_document_type(value: str | None) -> str:
        if not value:
            return "other"
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "other"
