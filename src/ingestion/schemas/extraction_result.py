"""
FinSight AI — Extraction Result Schemas
Chuẩn hóa output OCR/VLM thành JSON + Markdown cho ingestion pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class OCRBlock:
    """Một block text do OCR engine phát hiện."""

    text: str
    confidence: float = 0.0
    bbox: list[list[float]] = field(default_factory=list)
    language: str | None = None


@dataclass
class ExtractedTable:
    """Bảng đã được trích xuất/chuẩn hóa."""

    name: str
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ConfidenceReport:
    """Điểm tin cậy tổng hợp của pipeline."""

    ocr_confidence: float = 0.0
    layout_confidence: float = 0.0
    table_confidence: float = 0.0
    overall: float = 0.0





@dataclass
class ExtractionResult:
    """Kết quả cuối cùng của ingestion pipeline."""

    source_file: str
    document_type: str = "unknown"
    languages: list[str] = field(default_factory=list)
    markdown: str = ""
    fields: dict[str, Any] = field(default_factory=dict)
    tables: list[ExtractedTable] = field(default_factory=list)
    raw_ocr: list[OCRBlock] = field(default_factory=list)
    uncertain_tokens: list[str] = field(default_factory=list)
    quality_class: str = "unknown"
    quality_score: float = 0.0
    issue_flags: list[str] = field(default_factory=list)
    recommended_action: str = ""
    confidence: ConfidenceReport = field(default_factory=ConfidenceReport)
    requires_human_review: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Chuyển dataclass lồng nhau thành dict để lưu JSON."""
        return asdict(self)
