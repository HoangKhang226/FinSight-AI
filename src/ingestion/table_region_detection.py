"""
FinSight AI — Heuristic table region detection from cleaned images.

Module này tìm vùng bảng bằng morphology OpenCV để cô lập từng bảng trước
khi dựng lại Markdown từ OCR blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import get_logger

logger = get_logger(__name__)


@dataclass
class TableRegion:
    """Vùng bảng được phát hiện trên ảnh."""

    bbox: tuple[int, int, int, int]
    score: float = 0.0
    source: str = "morphology"

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]


class TableRegionDetector:
    """Detect vùng bảng từ ảnh cleaned, ưu tiên line morphology nhẹ."""

    def detect(self, image_path: str | Path) -> list[TableRegion]:
        path = str(image_path)
        image = cv2.imread(path)
        if image is None:
            logger.warning(f"Không thể đọc ảnh để detect vùng bảng: {path}")
            return []

        regions = self.detect_from_image(image)
        logger.info(f"Detected {len(regions)} table region(s) from: {path}")
        return regions

    def detect_from_image(self, image: np.ndarray) -> list[TableRegion]:
        if image is None or image.size == 0:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < 100 or gray.shape[1] < 100:
            return []

        # Nhị phân hóa để làm nổi đường kẻ bảng.
        binary = cv2.adaptiveThreshold(
            ~gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            -2,
        )

        horizontal = self._extract_lines(binary, axis="horizontal")
        vertical = self._extract_lines(binary, axis="vertical")
        grid = cv2.bitwise_or(horizontal, vertical)
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

        regions = self._contours_to_regions(grid, image.shape[:2])
        if regions:
            return regions

        # Fallback nhẹ: dùng Canny + morphology nếu lưới bảng quá yếu.
        edges = cv2.Canny(gray, 50, 150)
        fallback = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
        return self._contours_to_regions(fallback, image.shape[:2])

    def _extract_lines(self, binary: np.ndarray, axis: str) -> np.ndarray:
        h, w = binary.shape[:2]
        if axis == "horizontal":
            size = max(20, w // 30)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))
        else:
            size = max(20, h // 30)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
        extracted = cv2.erode(binary, kernel, iterations=1)
        extracted = cv2.dilate(extracted, kernel, iterations=1)
        return extracted

    def _contours_to_regions(self, mask: np.ndarray, image_shape: tuple[int, int]) -> list[TableRegion]:
        h, w = image_shape
        area_total = float(h * w)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: list[TableRegion] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < area_total * 0.01:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 80 or bh < 80:
                continue

            aspect = bw / max(1.0, bh)
            if aspect < 0.4 and bh / max(1.0, bw) < 0.4:
                continue

            pad_x = max(8, int(bw * 0.02))
            pad_y = max(8, int(bh * 0.02))
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)
            score = min(1.0, area / max(1.0, area_total * 0.1))
            regions.append(TableRegion(bbox=(x1, y1, x2, y2), score=round(score, 3)))

        regions.sort(key=lambda region: (region.y1, region.x1))
        return self._merge_overlapping_regions(regions)

    def _merge_overlapping_regions(self, regions: list[TableRegion]) -> list[TableRegion]:
        if not regions:
            return []

        merged: list[TableRegion] = [regions[0]]
        for region in regions[1:]:
            last = merged[-1]
            if self._iou(last.bbox, region.bbox) >= 0.25:
                merged[-1] = TableRegion(
                    bbox=(
                        min(last.x1, region.x1),
                        min(last.y1, region.y1),
                        max(last.x2, region.x2),
                        max(last.y2, region.y2),
                    ),
                    score=max(last.score, region.score),
                    source=last.source,
                )
            else:
                merged.append(region)
        return merged

    def _iou(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter_area / max(1.0, float(area_a + area_b - inter_area))


def crop_ocr_blocks_to_region(ocr_blocks: list[Any], region: TableRegion) -> list[Any]:
    """Lấy các OCR blocks có tâm nằm trong bbox region."""
    selected: list[Any] = []
    x1, y1, x2, y2 = region.bbox
    for block in ocr_blocks:
        if not getattr(block, "bbox", None):
            continue
        xs = [point[0] for point in block.bbox]
        ys = [point[1] for point in block.bbox]
        center_x = (min(xs) + max(xs)) / 2.0
        center_y = (min(ys) + max(ys)) / 2.0
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            selected.append(block)
    return selected
