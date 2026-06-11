"""
FinSight AI — PaddleOCR Extractor
Local multilingual OCR engine optimized as a companion to VLM extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os

from src.config import settings, get_logger
from src.ingestion.schemas import OCRBlock

# PaddleOCR 3.x + paddlepaddle 3.x trên Windows CPU có thể lỗi oneDNN/PIR:
# "ConvertPirAttribute2RuntimeAttribute not support ...".
# Tắt các optimization này trước khi import/khởi tạo PaddleOCR để ưu tiên đường chạy ổn định.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("PADDLE_DISABLE_MKLDNN", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

logger = get_logger(__name__)


class PaddleOCRExtractor:
    """Wrapper nhẹ quanh PaddleOCR, trả về OCRBlock chuẩn hóa."""

    def __init__(self, languages: list[str] | None = None, use_gpu: bool | None = None):
        self.languages = languages or settings.get("ingestion.ocr.languages", ["vi", "en"])
        self.use_gpu = settings.get("ingestion.ocr.use_gpu", False) if use_gpu is None else use_gpu
        self._ocr = None

    def _load_engine(self):
        """Lazy-load PaddleOCR để tránh tăng thời gian import app.

        PaddleOCR 3.x official API:
        - PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, ...)
        - ocr.predict(image_path)

        PaddleOCR 2.x fallback:
        - PaddleOCR(use_angle_cls=True, lang=...)
        - ocr.ocr(image_path, cls=True)
        """
        if self._ocr is not None:
            return self._ocr

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR chưa được cài. Hãy cài dependencies trong requirements-core.txt."
            ) from exc

        lang = self._select_primary_lang(self.languages)
        device = "gpu" if self.use_gpu else "cpu"
        logger.info(f"Khởi tạo PaddleOCR lang={lang}, device={device}")

        try:
            self._ocr = PaddleOCR(
                lang=lang,
                device=device,
                use_angle_cls=True,
                use_doc_orientation_classify=True,
                use_doc_unwarping=False,
                use_textline_orientation=True,
            )
            self._api_version = 3
            return self._ocr
        except TypeError as exc:
            logger.warning(f"PaddleOCR 3.x constructor không tương thích, fallback 2.x: {exc}")

        try:
            self._ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        except TypeError:
            self._ocr = PaddleOCR(lang=lang)
        self._api_version = 2
        return self._ocr

    @staticmethod
    def _select_primary_lang(languages: list[str]) -> str:
        """Map config language list sang PaddleOCR lang hợp lệ."""
        lowered = {lang.lower() for lang in languages}
        if "ch" in lowered or "zh" in lowered or "chinese" in lowered:
            return "ch"
        if "japan" in lowered or "ja" in lowered or "japanese" in lowered:
            return "japan"
        if "korean" in lowered or "ko" in lowered:
            return "korean"
        if "vi" in lowered or "vietnamese" in lowered:
            return "vi"
        return "en"

    def extract(self, image_path: str | Path) -> list[OCRBlock]:
        """Chạy OCR trên ảnh và trả danh sách text blocks."""
        path = Path(image_path)
        logger.info(f"Bắt đầu PaddleOCR: {path.name}")
        ocr = self._load_engine()

        if getattr(self, "_api_version", 3) == 3 and hasattr(ocr, "predict"):
            raw_result = ocr.predict(str(path))
            blocks = self._parse_paddle3_result(raw_result)
        else:
            try:
                raw_result: list[Any] = ocr.ocr(str(path), cls=True)
            except TypeError:
                raw_result = ocr.ocr(str(path))
            blocks = self._parse_legacy_ocr_result(raw_result)

        logger.info(f"PaddleOCR hoàn tất: {len(blocks)} blocks")
        return blocks

    def _parse_legacy_ocr_result(self, raw_result: list[Any]) -> list[OCRBlock]:
        """Parse output kiểu PaddleOCR 2.x."""
        blocks: list[OCRBlock] = []
        for page in raw_result or []:
            for line in page or []:
                if not line or len(line) < 2:
                    continue
                bbox = line[0]
                text_info = line[1]
                text = text_info[0] if text_info else ""
                confidence = float(text_info[1]) if len(text_info) > 1 else 0.0
                if text.strip():
                    blocks.append(OCRBlock(text=text.strip(), confidence=confidence, bbox=bbox))
        return blocks

    def _parse_paddle3_result(self, raw_result: list[Any]) -> list[OCRBlock]:
        """Parse output kiểu PaddleOCR 3.x/PaddleX."""
        blocks: list[OCRBlock] = []
        for item in raw_result or []:
            data = getattr(item, "json", None)
            if callable(data):
                data = data()
            if not isinstance(data, dict):
                data = item if isinstance(item, dict) else {}

            result = data.get("res", data)
            texts = result.get("rec_texts") or result.get("texts") or []
            scores = result.get("rec_scores") or result.get("scores") or []
            boxes = result.get("rec_boxes") or result.get("dt_polys") or result.get("boxes") or []

            for idx, text in enumerate(texts):
                confidence = float(scores[idx]) if idx < len(scores) else 0.0
                bbox = boxes[idx].tolist() if idx < len(boxes) and hasattr(boxes[idx], "tolist") else (boxes[idx] if idx < len(boxes) else [])
                if str(text).strip():
                    blocks.append(OCRBlock(text=str(text).strip(), confidence=confidence, bbox=bbox))
        return blocks
