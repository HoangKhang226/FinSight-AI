"""
FinSight AI — Heuristic table reconstruction from OCR bounding boxes.

Module này độc lập với prompt/VLM. Nó chỉ dùng hình học bbox để phát hiện
cụm bảng và dựng Markdown table tuyến tính, phù hợp làm fallback cho RAG.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from statistics import median
from typing import Any

from src.ingestion.schemas import ExtractedTable, OCRBlock


@dataclass
class _BlockGeom:
    """OCR block kèm thông tin hình học đã chuẩn hóa."""

    text: str
    confidence: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    @property
    def width(self) -> float:
        return max(1.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(1.0, self.y_max - self.y_min)


def reconstruct_tables_from_ocr(
    ocr_blocks: list[OCRBlock],
    *,
    min_rows: int = 3,
    min_cols: int = 3,
    confidence_threshold: float = 0.7,
) -> tuple[list[ExtractedTable], str, dict[str, Any]]:
    """Dựng bảng Markdown bằng heuristic gom hàng/cột từ bbox OCR.

    Trả về `(tables, markdown, metrics)`. Nếu không đủ bằng chứng có bảng,
    `tables` và `markdown` sẽ rỗng. Nếu score thấp, trả về linear text flow.
    """
    geoms = [_to_geom(block) for block in ocr_blocks]
    geoms = [geom for geom in geoms if geom and geom.text.strip()]
    if len(geoms) < min_rows * min_cols:
        linear_markdown = _linear_text_markdown(geoms)
        return [], linear_markdown, {"detected": False, "reason": "too_few_blocks", "table_confidence": 0.0, "fallback": "linear_text"}

    rows = _group_rows(geoms)
    row_cells = [sorted(row, key=lambda item: item.center_x) for row in rows]
    useful_rows = [row for row in row_cells if len(row) >= 2]
    if len(useful_rows) < min_rows:
        linear_markdown = _linear_text_markdown(geoms)
        return [], linear_markdown, {"detected": False, "reason": "too_few_rows", "rows": len(useful_rows), "table_confidence": 0.0, "fallback": "linear_text"}

    anchor_row = _select_anchor_row(useful_rows)
    column_anchors = _extract_column_anchors(anchor_row)
    if len(column_anchors) < min_cols:
        column_anchors = _extract_column_anchors(max(useful_rows, key=len))
    if len(column_anchors) < min_cols:
        linear_markdown = _linear_text_markdown(geoms)
        return [], linear_markdown, {"detected": False, "reason": "too_few_columns", "columns": len(column_anchors), "table_confidence": 0.0, "fallback": "linear_text"}

    header_row = _build_header_row(anchor_row, column_anchors)
    grid = [_assign_row_to_columns(row, column_anchors) for row in useful_rows]
    grid = _normalize_grid_rows(grid, len(column_anchors))
    grid = _drop_duplicate_header_rows(grid, header_row)
    if len(grid) < min_rows:
        linear_markdown = _linear_text_markdown(geoms)
        return [], linear_markdown, {"detected": False, "reason": "sparse_grid", "rows": len(grid), "table_confidence": 0.0, "fallback": "linear_text"}

    header = _choose_header(grid, header_row)
    data_rows = grid[1:] if _rows_match_header(grid[0], header) else grid
    columns = _make_columns(header, len(column_anchors))
    table_rows = [_row_to_dict(columns, row) for row in data_rows]
    markdown = _grid_to_markdown(columns, data_rows)

    table_confidence, confidence_metrics = _score_table_quality(
        geoms=geoms,
        grid=grid,
        columns=columns,
        column_anchors=column_anchors,
        header_row=header_row,
        data_rows=data_rows,
    )
    confidence_metrics.update(
        {
            "detected": True,
            "rows": len(grid),
            "columns": len(columns),
            "source": "ocr_bbox_heuristic",
            "table_confidence": round(table_confidence, 3),
        }
    )

    if table_confidence < confidence_threshold:
        linear_markdown = _linear_text_markdown(geoms)
        confidence_metrics.update({"fallback": "linear_text", "fallback_reason": "low_confidence"})
        return [], linear_markdown, confidence_metrics

    table = ExtractedTable(name="reconstructed_table_1", columns=columns, rows=table_rows)
    return [table], markdown, confidence_metrics


def _to_geom(block: OCRBlock) -> _BlockGeom | None:
    if not block.bbox:
        return None
    try:
        xs = [float(point[0]) for point in block.bbox]
        ys = [float(point[1]) for point in block.bbox]
    except (TypeError, ValueError, IndexError):
        return None
    return _BlockGeom(
        text=block.text.strip(),
        confidence=float(block.confidence or 0.0),
        x_min=min(xs),
        x_max=max(xs),
        y_min=min(ys),
        y_max=max(ys),
    )


def _vertical_overlap_ratio(a: _BlockGeom, b: _BlockGeom) -> float:
    overlap = max(0.0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
    return overlap / max(1.0, min(a.height, b.height))


def _group_rows(blocks: list[_BlockGeom]) -> list[list[_BlockGeom]]:
    sorted_blocks = sorted(blocks, key=lambda item: (item.center_y, item.center_x))
    rows: list[list[_BlockGeom]] = []
    median_height = median([block.height for block in sorted_blocks]) if sorted_blocks else 12.0
    y_tolerance = max(8.0, median_height * 0.7)

    for block in sorted_blocks:
        best_index: int | None = None
        best_score = 0.0
        for idx, row in enumerate(rows):
            row_ref = row[0]
            overlap = _vertical_overlap_ratio(row_ref, block)
            center_delta = abs(row_ref.center_y - block.center_y)
            score = overlap - (center_delta / max(1.0, y_tolerance)) * 0.15
            if overlap >= 0.45 or center_delta <= y_tolerance:
                if score > best_score:
                    best_score = score
                    best_index = idx
        if best_index is None:
            rows.append([block])
        else:
            rows[best_index].append(block)

    return sorted(rows, key=lambda row: median([item.center_y for item in row]))


def _extract_column_anchors(row: list[_BlockGeom]) -> list[float]:
    if not row:
        return []

    sorted_row = sorted(row, key=lambda item: item.center_x)
    anchors: list[float] = [sorted_row[0].center_x]
    widths = [block.width for block in sorted_row]
    snap_gap = max(16.0, median(widths) * 0.6)

    for block in sorted_row[1:]:
        if abs(block.center_x - anchors[-1]) > snap_gap:
            anchors.append(block.center_x)
        else:
            anchors[-1] = (anchors[-1] + block.center_x) / 2.0

    return anchors


def _select_anchor_row(rows: list[list[_BlockGeom]]) -> list[_BlockGeom]:
    if not rows:
        return []
    return max(rows, key=lambda row: (len(row), -median([item.y_min for item in row])))


def _assign_row_to_columns(row: list[_BlockGeom], anchors: list[float]) -> list[str]:
    """Snap từng OCR block vào cột neo gần nhất."""
    cells: list[list[str]] = [[] for _ in anchors]
    for block in sorted(row, key=lambda item: item.center_x):
        nearest_idx = min(range(len(anchors)), key=lambda idx: abs(block.center_x - anchors[idx]))
        cells[nearest_idx].append(block.text)
    return [" ".join(parts).strip() for parts in cells]


def _build_header_row(anchor_row: list[_BlockGeom], anchors: list[float]) -> list[str]:
    return _assign_row_to_columns(anchor_row, anchors)


def _normalize_grid_rows(grid: list[list[str]], column_count: int) -> list[list[str]]:
    normalized: list[list[str]] = []
    for row in grid:
        cells = list(row[:column_count])
        if len(cells) < column_count:
            cells.extend([""] * (column_count - len(cells)))
        normalized.append(cells)
    return normalized


def _row_text_signature(row: list[str]) -> str:
    return " ".join(cell.strip().lower() for cell in row if cell.strip())


def _rows_match_header(row: list[str], header: list[str]) -> bool:
    row_sig = _row_text_signature(row)
    header_sig = _row_text_signature(header)
    if not row_sig or not header_sig:
        return False
    row_tokens = set(row_sig.split())
    header_tokens = set(header_sig.split())
    if not row_tokens or not header_tokens:
        return False
    overlap = len(row_tokens & header_tokens) / max(1, len(header_tokens))
    return overlap >= 0.8


def _drop_duplicate_header_rows(grid: list[list[str]], header_row: list[str]) -> list[list[str]]:
    if not header_row:
        return grid
    deduped: list[list[str]] = []
    for idx, row in enumerate(grid):
        if idx > 0 and _rows_match_header(row, header_row):
            continue
        deduped.append(row)
    return deduped


def _choose_header(grid: list[list[str]], header_row: list[str]) -> list[str]:
    """Chọn header theo cấu trúc, không dựa vào ngôn ngữ.

    Header ưu tiên là anchor row đã được chọn từ hình học bbox. Nếu anchor row
    rỗng, dùng hàng đầu tiên có mật độ ô cao nhất ở phần đầu bảng.
    """
    if header_row and any(cell.strip() for cell in header_row):
        return header_row
    if not grid:
        return []
    candidate_window = grid[: max(1, min(3, len(grid)))]
    best_row = max(candidate_window, key=lambda row: (sum(1 for cell in row if cell.strip()), -candidate_window.index(row)))
    if any(cell.strip() for cell in best_row):
        return best_row
    return [f"Column {idx}" for idx in range(1, len(grid[0]) + 1)]


def _make_columns(header: list[str], col_count: int) -> list[str]:
    columns: list[str] = []
    for idx in range(col_count):
        value = header[idx].strip() if idx < len(header) else ""
        columns.append(value or f"Column {idx + 1}")
    seen: dict[str, int] = {}
    unique_columns: list[str] = []
    for column in columns:
        count = seen.get(column, 0) + 1
        seen[column] = count
        unique_columns.append(column if count == 1 else f"{column} {count}")
    return unique_columns


def _row_to_dict(columns: list[str], row: list[str]) -> dict[str, Any]:
    return {column: row[idx] if idx < len(row) else "" for idx, column in enumerate(columns)}


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _grid_to_markdown(columns: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(_escape_cell(col) for col in columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        padded = [row[idx] if idx < len(row) else "" for idx in range(len(columns))]
        lines.append("| " + " | ".join(_escape_cell(cell) for cell in padded) + " |")
    return "\n".join(lines)


def _linear_text_markdown(geoms: list[_BlockGeom]) -> str:
    """Dựng lại văn bản tuyến tính sạch hơn khi bảng không đủ tin cậy.

    Fallback này ưu tiên dữ liệu bảng/giá trị giao dịch và chủ động loại bớt
    header/footer hành chính để giảm nhiễu khi đưa vào embedding/RAG.
    """
    if not geoms:
        return ""

    rows = _strict_group_rows_for_reading_order(geoms)
    filtered_rows = _strip_non_table_noise_rows(rows)
    if not filtered_rows:
        filtered_rows = rows

    lines: list[str] = []
    for row in filtered_rows:
        text = _row_to_linear_text(row)
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines).strip()


def _strict_group_rows_for_reading_order(blocks: list[_BlockGeom]) -> list[list[_BlockGeom]]:
    """Gom dòng theo chaining cục bộ để chịu được ảnh chụp bị nghiêng/méo.

    Thuật toán duyệt block từ trái sang phải, ưu tiên gắn block mới vào hàng có
    block bên trái gần nhất và baseline Y động gần nhất. Mỗi hàng giữ một
    frontier X và một baseline Y cập nhật dần theo các block đã gắn.
    """
    sorted_blocks = sorted(blocks, key=lambda item: (item.x_min, item.center_y))
    if not sorted_blocks:
        return []

    median_height = median([block.height for block in sorted_blocks])
    base_tolerance = max(6.0, median_height * 0.65)
    overlap_tolerance = max(0.20, min(0.45, 0.18 + median_height / max(1.0, sum(block.width for block in sorted_blocks) / len(sorted_blocks)) * 0.08))

    rows: list[dict[str, Any]] = []

    for block in sorted_blocks:
        best_idx: int | None = None
        best_score = float("inf")

        for idx, row in enumerate(rows):
            last_block = row["blocks"][-1]
            if block.x_min + 3.0 < last_block.x_min:
                continue

            y_delta = abs(block.center_y - row["baseline_y"])
            local_tolerance = max(base_tolerance, min(last_block.height, block.height) * 0.8)
            x_gap = max(0.0, block.x_min - row["frontier_x"])
            y_overlap = _vertical_overlap_ratio(last_block, block)
            same_band = y_delta <= local_tolerance
            touching_band = y_overlap >= overlap_tolerance

            if not (same_band or touching_band):
                continue

            score = (
                y_delta * 1.0
                + x_gap * 0.02
                + abs(block.center_x - row["frontier_x"]) * 0.002
                + max(0.0, len(row["blocks"]) - 1) * 0.01
            )
            if score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            rows.append(
                {
                    "blocks": [block],
                    "baseline_y": block.center_y,
                    "frontier_x": block.x_max,
                }
            )
            continue

        row = rows[best_idx]
        row["blocks"].append(block)
        row["frontier_x"] = max(row["frontier_x"], block.x_max)
        row["baseline_y"] = (
            row["baseline_y"] * 0.68
            + block.center_y * 0.32
        )

    normalized_rows = [
        sorted(row["blocks"], key=lambda item: (item.x_min, item.center_x))
        for row in rows
    ]
    return sorted(normalized_rows, key=lambda row: median([item.center_y for item in row]))


def _strip_non_table_noise_rows(rows: list[list[_BlockGeom]]) -> list[list[_BlockGeom]]:
    """Loại các dòng chắc chắn là thông tin hành chính ngoài bảng."""
    if not rows:
        return []

    bounded_rows = _apply_safe_y_boundary_filter(rows)
    if not bounded_rows:
        bounded_rows = rows

    first_table_idx = _find_first_table_like_row(bounded_rows)
    last_table_idx = _find_last_table_like_row(bounded_rows)
    candidate_rows = bounded_rows[first_table_idx : last_table_idx + 1] if first_table_idx <= last_table_idx else bounded_rows

    cleaned: list[list[_BlockGeom]] = []
    for row in candidate_rows:
        text = _row_to_linear_text(row)
        if not text:
            continue
        if _is_noise_line(text) and not _is_table_like_line(text):
            continue
        cleaned.append(row)
    return cleaned


def _apply_safe_y_boundary_filter(rows: list[list[_BlockGeom]], margin_ratio: float = 0.05) -> list[list[_BlockGeom]]:
    """Loại dòng ở 5% biên trên/dưới nếu không giống dữ liệu bảng.

    Khi table detector/crop hơi rộng, các dòng hành chính thường nằm sát biên.
    Dòng ở vùng biên chỉ được giữ nếu có dấu hiệu thuộc cấu trúc bảng.
    """
    all_blocks = [block for row in rows for block in row]
    if not all_blocks:
        return rows

    y_min = min(block.y_min for block in all_blocks)
    y_max = max(block.y_max for block in all_blocks)
    height = max(1.0, y_max - y_min)
    top_cutoff = y_min + height * margin_ratio
    bottom_cutoff = y_max - height * margin_ratio

    filtered: list[list[_BlockGeom]] = []
    for row in rows:
        row_center = median([block.center_y for block in row])
        text = _row_to_linear_text(row)
        in_boundary = row_center <= top_cutoff or row_center >= bottom_cutoff
        if in_boundary and not _is_table_like_line(text):
            continue
        filtered.append(row)
    return filtered


def _row_to_linear_text(row: list[_BlockGeom]) -> str:
    merged_parts = _merge_horizontal_fragments(row)
    return _normalize_linear_text(" ".join(merged_parts))


def _merge_horizontal_fragments(row: list[_BlockGeom]) -> list[str]:
    """Ghép các OCR block gần nhau theo chiều ngang trong cùng dòng."""
    if not row:
        return []

    sorted_row = sorted(row, key=lambda item: (item.x_min, item.center_x))
    median_height = median([block.height for block in sorted_row])
    median_width = median([block.width for block in sorted_row])
    gap_threshold = max(8.0, min(38.0, median_width * 0.55 + median_height * 0.6))

    merged: list[tuple[str, float, float]] = []
    current_text = sorted_row[0].text.strip()
    current_x_min = sorted_row[0].x_min
    current_x_max = sorted_row[0].x_max

    for block in sorted_row[1:]:
        text = block.text.strip()
        if not text:
            continue
        gap = block.x_min - current_x_max
        if gap <= gap_threshold:
            current_text = f"{current_text} {text}".strip()
            current_x_max = max(current_x_max, block.x_max)
        else:
            merged.append((current_text, current_x_min, current_x_max))
            current_text = text
            current_x_min = block.x_min
            current_x_max = block.x_max

    if current_text:
        merged.append((current_text, current_x_min, current_x_max))
    return [text for text, _, _ in merged if text]


def _find_first_table_like_row(rows: list[list[_BlockGeom]]) -> int:
    """Tìm đầu vùng bảng bằng mật độ hình học, không dùng từ khóa."""
    if not rows:
        return 0
    scores = _row_geometry_scores(rows)
    top_limit = max(1, int(len(rows) * 0.55))
    candidate_indexes = range(0, top_limit)
    return max(candidate_indexes, key=lambda idx: (scores[idx], len(rows[idx]), -idx))


def _find_last_table_like_row(rows: list[list[_BlockGeom]]) -> int:
    """Tìm cuối vùng bảng bằng mật độ hình học/content số phổ quát."""
    if not rows:
        return 0
    scores = _row_geometry_scores(rows)
    best_start = _find_first_table_like_row(rows)
    for idx in range(len(rows) - 1, best_start - 1, -1):
        if scores[idx] >= 0.35 or _row_has_universal_numeric_signal(rows[idx]):
            return idx
    return len(rows) - 1


def _row_geometry_scores(rows: list[list[_BlockGeom]]) -> list[float]:
    max_cells = max((len(row) for row in rows), default=1)
    all_blocks = [block for row in rows for block in row]
    page_width = max(1.0, max((block.x_max for block in all_blocks), default=1.0) - min((block.x_min for block in all_blocks), default=0.0))
    scores: list[float] = []
    for row in rows:
        cell_density = len(row) / max(1, max_cells)
        x_spread = _row_x_spread(row) / page_width
        uniformity = _row_x_uniformity(row)
        numeric_signal = 0.25 if _row_has_universal_numeric_signal(row) else 0.0
        scores.append(min(1.0, cell_density * 0.45 + x_spread * 0.25 + uniformity * 0.20 + numeric_signal))
    return scores


def _row_x_spread(row: list[_BlockGeom]) -> float:
    if not row:
        return 0.0
    return max(block.x_max for block in row) - min(block.x_min for block in row)


def _row_x_uniformity(row: list[_BlockGeom]) -> float:
    if len(row) < 3:
        return 0.0
    centers = sorted(block.center_x for block in row)
    gaps = [centers[idx + 1] - centers[idx] for idx in range(len(centers) - 1)]
    positive_gaps = [gap for gap in gaps if gap > 1.0]
    if len(positive_gaps) < 2:
        return 0.0
    avg_gap = sum(positive_gaps) / len(positive_gaps)
    variance = sum(abs(gap - avg_gap) for gap in positive_gaps) / max(1, len(positive_gaps))
    return max(0.0, 1.0 - min(1.0, variance / max(1.0, avg_gap)))


def _row_has_universal_numeric_signal(row: list[_BlockGeom]) -> bool:
    text = _row_to_linear_text(row)
    digit_count = sum(1 for char in text if char.isdigit())
    number_tokens = re.findall(r"\d[\d.,:/-]*", text)
    return digit_count >= 2 or len(number_tokens) >= 2


def _normalize_linear_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" -\t")


def _is_table_like_line(text: str) -> bool:
    digit_count = sum(1 for char in text if char.isdigit())
    number_tokens = re.findall(r"\d[\d.,:/-]*", text)
    return digit_count >= 2 or len(number_tokens) >= 2


def _is_noise_line(text: str) -> bool:
    """Không dùng blacklist ngôn ngữ; nhiễu được xử lý bằng geometry crop."""
    return False


def _score_table_quality(
    *,
    geoms: list[_BlockGeom],
    grid: list[list[str]],
    columns: list[str],
    column_anchors: list[float],
    header_row: list[str],
    data_rows: list[list[str]],
) -> tuple[float, dict[str, Any]]:
    """Chấm điểm chất lượng bảng bằng heuristic, không dùng model."""
    snap_distances = _compute_snap_distances(geoms, column_anchors)
    snap_distance_variance = sum(snap_distances) / max(1, len(snap_distances)) if snap_distances else 0.0

    total_cells = max(1, len(grid) * len(columns))
    empty_cells = sum(1 for row in grid for cell in row if not cell.strip())
    empty_cell_ratio = empty_cells / total_cells

    avg_ocr_confidence = sum(block.confidence for block in geoms) / max(1, len(geoms))

    numeric_columns = _guess_numeric_columns(columns, header_row)
    sanity_penalty = _sanity_check_penalty(data_rows, numeric_columns)

    snap_score = max(0.0, 1.0 - min(1.0, snap_distance_variance / 60.0))
    structure_score = max(0.0, 1.0 - min(1.0, empty_cell_ratio / 0.35))
    ocr_score = max(0.0, min(1.0, avg_ocr_confidence))
    sanity_score = max(0.0, 1.0 - sanity_penalty)

    final_score = (
        snap_score * 0.30
        + structure_score * 0.25
        + ocr_score * 0.25
        + sanity_score * 0.20
    )
    metrics = {
        "snap_distance_variance": round(snap_distance_variance, 3),
        "empty_cell_ratio": round(empty_cell_ratio, 3),
        "avg_ocr_confidence": round(avg_ocr_confidence, 3),
        "sanity_penalty": round(sanity_penalty, 3),
        "snap_score": round(snap_score, 3),
        "structure_score": round(structure_score, 3),
        "ocr_score": round(ocr_score, 3),
        "sanity_score": round(sanity_score, 3),
    }
    return final_score, metrics


def _compute_snap_distances(geoms: list[_BlockGeom], column_anchors: list[float]) -> list[float]:
    if not geoms or not column_anchors:
        return []
    distances: list[float] = []
    for block in geoms:
        nearest = min(abs(block.center_x - anchor) for anchor in column_anchors)
        distances.append(nearest)
    return distances


def _guess_numeric_columns(columns: list[str], header_row: list[str]) -> set[int]:
    """Đoán cột số bằng hình thái nội dung thay vì tên cột/ngôn ngữ."""
    numeric_indexes: set[int] = set()
    for idx, column in enumerate(columns):
        sample = f"{column} {header_row[idx] if idx < len(header_row) else ''}"
        if _looks_numeric_like(sample):
            numeric_indexes.add(idx)
    return numeric_indexes


def _looks_numeric_like(text: str) -> bool:
    compact = text.strip()
    if not compact:
        return False
    digit_count = sum(1 for char in compact if char.isdigit())
    alpha_count = sum(1 for char in compact if char.isalpha())
    numberish_count = sum(1 for char in compact if char.isdigit() or char in ".,:%/+-()OoSsoIl|")
    return digit_count >= 1 and numberish_count >= max(1, alpha_count)


def _normalize_numeric_ocr(text: str) -> str:
    """Sửa nhầm lẫn OCR phổ quát trong chuỗi số mà không phụ thuộc ngôn ngữ."""
    normalized = re.sub(r"(?<=\d)[OoS](?=\d)|(?<=\d)[OoS]|[OoS](?=\d)", "0", text)
    normalized = re.sub(r"(?<=\d)[Il|](?=\d)|(?<=\d)[Il|]|[Il|](?=\d)", "1", normalized)
    return normalized


def _sanity_check_penalty(data_rows: list[list[str]], numeric_columns: set[int]) -> float:
    if not data_rows or not numeric_columns:
        return 0.0
    penalty = 0.0
    inspected = 0
    for row in data_rows:
        for idx in numeric_columns:
            if idx >= len(row):
                continue
            cell = _normalize_numeric_ocr(row[idx].strip())
            if not cell:
                continue
            inspected += 1
            alpha_count = sum(1 for char in cell if char.isalpha())
            digit_count = sum(1 for char in cell if char.isdigit())
            if alpha_count > digit_count:
                penalty += 1.0
            elif alpha_count > 0:
                penalty += 0.5
    if inspected == 0:
        return 0.0
    return min(1.0, penalty / inspected)
