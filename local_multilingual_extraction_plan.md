# FinSight AI — Local Multilingual Extraction Pipeline

## 1. Bối cảnh

Project FinSight AI cần đọc tài liệu tài chính đa ngôn ngữ trên laptop local:

```text
Thiết bị mục tiêu: Laptop RTX 2050, 16GB RAM
Runtime: local-first
VLM: Ollama
Model hiện tại: qwen2.5vl:3b
Input: ảnh hóa đơn, ảnh scan, PDF báo cáo, chứng từ tài chính
Output: Markdown + JSON để phục vụ RAG và xử lý nghiệp vụ
```

Sau khi test thực tế với ảnh hóa đơn trong `data/raw`, có 2 kết luận quan trọng:

1. **Docling đọc ảnh hóa đơn tiếng Việt chưa ổn**:
   - OCR tiếng Việt sai nhiều.
   - Bảng bị vỡ cấu trúc.
   - Phù hợp hơn với PDF digital hoặc tài liệu có text layer.

2. **Qwen2.5VL 3B dựng layout tốt hơn**:
   - Giữ được cấu trúc bảng Markdown.
   - Hiểu bố cục hóa đơn tốt.
   - Nhưng OCR chữ nhỏ/tiếng Việt vẫn còn sai.

Vì vậy, pipeline mới không dùng một engine duy nhất. Ta chuyển sang hướng **hybrid extraction**:

```text
PaddleOCR đọc chữ chi tiết
+
Qwen2.5VL dựng layout/bảng
+
Output Markdown + JSON
```

Bước xác thực số liệu tài chính hiện được xem là **optional**, có thể để sau hoặc bỏ nếu không phù hợp với phạm vi project.

---

## 2. Lý do làm pipeline mới

### 2.1 Không thể phụ thuộc hoàn toàn vào Docling

Docling mạnh với:

- PDF digital.
- Layout parsing.
- Tài liệu sạch có text layer.

Nhưng khi dùng cho ảnh hóa đơn tiếng Việt:

- Chữ nhận sai.
- Bảng bị tách dòng sai.
- Không đủ ổn định cho ingestion đa ngôn ngữ.

### 2.2 Không thể phụ thuộc hoàn toàn vào VLM nhỏ

`qwen2.5vl:3b` chạy được trên RTX 2050, nhưng vẫn là model nhỏ:

- Hiểu layout tốt.
- Dựng bảng tốt.
- Đọc số tiền khá ổn.
- Nhưng chữ nhỏ, dấu tiếng Việt, tên hàng hóa có thể sai.

Do đó, VLM nên nhận thêm **OCR text blocks** làm bằng chứng phụ trợ.

### 2.3 PaddleOCR phù hợp làm OCR nền

PaddleOCR phù hợp vì:

- Hỗ trợ nhiều ngôn ngữ.
- Đọc text box chi tiết tốt hơn VLM nhỏ trong nhiều trường hợp.
- Chạy CPU được trên laptop 16GB RAM.
- Không bắt buộc CUDA/Paddle GPU ở giai đoạn đầu.

### 2.4 Mục tiêu của pipeline mới

Pipeline mới hướng tới:

- Quét ảnh/PDF ổn định hơn.
- Trích xuất đủ text, bảng, số tiền, ngày tháng.
- Giữ nguyên ngôn ngữ gốc, không dịch.
- Có output JSON để sau này dễ validate/index/UI.
- Chạy local được, không phụ thuộc cloud API.

---

## 3. Cấu trúc ingestion mới

Các module chính đã/đang được triển khai:

```text
src/ingestion/
  classifier.py
  path_happy.py
  path_fallback.py
  vlm_ocr.py
  pipeline.py

  extractors/
    __init__.py
    paddle_ocr_extractor.py

  prompts/
    __init__.py
    vlm_prompts.py

  schemas/
    __init__.py
    extraction_result.py

  fusion/
    __init__.py
    validator.py
```

Vai trò từng phần:

| Module | Vai trò |
|---|---|
| `classifier.py` | Phân loại input PDF/ảnh. Hiện ảnh được ưu tiên qua VLM pipeline. |
| `path_happy.py` | Dùng Docling cho PDF digital/tài liệu sạch. |
| `path_fallback.py` | Tiền xử lý ảnh nhẹ, không phá ảnh. |
| `paddle_ocr_extractor.py` | Chạy PaddleOCR để lấy text blocks. |
| `vlm_ocr.py` | Gọi VLM qua Ollama. |
| `vlm_prompts.py` | Tạo prompt JSON-first có OCR context. |
| `extraction_result.py` | Schema chuẩn cho Markdown, JSON, OCR blocks, confidence. |
| `pipeline.py` | Orchestrator chính của ingestion. |
| `validator.py` | Rule validation nhẹ, hiện là optional. |

---

## 4. Pipeline xử lý ảnh để lấy thông tin

Luồng xử lý ảnh/chứng từ scan:

```text
Ảnh đầu vào
  ↓
DocumentClassifier
  ↓
ImagePreprocessor
  ↓
PaddleOCRExtractor
  ↓
VLMOCRProcessor với prompt có OCR context
  ↓
Parse JSON từ VLM
  ↓
Lưu Markdown + JSON
  ↓
Chunk Markdown
  ↓
Index vào Qdrant
```

Chi tiết:

### Bước 1 — Classify

Ảnh `.png`, `.jpg`, `.jpeg` được đưa vào pipeline ảnh.

Lý do: ảnh hóa đơn dù rõ vẫn có khả năng bị Docling OCR sai, nên dùng hybrid OCR/VLM ổn hơn.

### Bước 2 — Preprocess nhẹ

`ImagePreprocessor` chỉ nên làm nhẹ:

- Giữ ảnh gốc/ảnh tự nhiên.
- Deskew nếu ảnh nghiêng.
- Không adaptive threshold mạnh.
- Không binarize trắng đen mặc định.
- Không remove shadow quá tay.

Lý do: VLM đọc tốt ảnh tự nhiên hơn ảnh đã bị xử lý quá mạnh.

### Bước 3 — PaddleOCR

PaddleOCR tạo danh sách text blocks:

```json
[
  {
    "text": "Hóa đơn GTGT",
    "confidence": 0.93,
    "bbox": [[...], [...], [...], [...]]
  }
]
```

Các block này không phải output cuối. Chúng là bằng chứng phụ trợ cho VLM.

### Bước 4 — VLM extraction

VLM nhận:

1. Ảnh đã preprocess nhẹ.
2. OCR text blocks.
3. Prompt yêu cầu output JSON hợp lệ.

Nhiệm vụ VLM:

- Dựng lại layout.
- Dựng bảng Markdown.
- Gom field như invoice number/date/total nếu có.
- Đánh dấu phần không chắc vào `uncertain_tokens`.

### Bước 5 — Parse output

Pipeline cố parse JSON từ response VLM.

Nếu parse được:

- `markdown` lấy từ JSON.
- `fields`, `tables`, `uncertain_tokens` được lưu vào `.json`.

Nếu parse lỗi:

- Dùng raw response làm Markdown fallback.
- Vẫn không làm crash pipeline.

### Bước 6 — Save outputs

Mỗi input tạo:

```text
data/processed/<file_stem>.md
data/processed/<file_stem>.json
```

Markdown dùng cho RAG.

JSON dùng cho:

- UI.
- Debug.
- Tìm field.
- Sau này validate nghiệp vụ nếu cần.

### Bước 7 — Index

Chỉ index Markdown cuối cùng vào Qdrant, không index raw OCR rác.

---

## 5. Pipeline PDF

### PDF digital

```text
PDF digital
  ↓
Docling
  ↓
Markdown
  ↓
JSON wrapper
  ↓
Index
```

### PDF scan

Giai đoạn sau nên thêm render PDF page sang ảnh:

```text
PDF scan
  ↓
Render từng page thành image
  ↓
Chạy image pipeline
```

Hiện task trước mắt tập trung ảnh trong `data/raw`.

---

## 6. Output chuẩn

Pipeline mới lưu `ExtractionResult` dạng JSON:

```json
{
  "source_file": "image.png",
  "document_type": "invoice",
  "languages": ["vi", "en"],
  "markdown": "# ...",
  "fields": {
    "invoice_number": null,
    "date": null,
    "seller_name": null,
    "buyer_name": null,
    "tax_code": null,
    "phone": null,
    "subtotal": null,
    "vat_amount": null,
    "total_amount": null
  },
  "tables": [],
  "raw_ocr": [],
  "uncertain_tokens": [],
  "confidence": {
    "ocr_confidence": 0.0,
    "layout_confidence": 0.0,
    "table_confidence": 0.0,
    "financial_validation_score": 0.0,
    "overall": 0.0
  },
  "requires_human_review": false,
  "metadata": {}
}
```

---

## 7. Về bước xác thực số liệu

Hiện `validator.py` chỉ là rule nhẹ để cảnh báo:

- Markdown quá ngắn.
- Không có bảng.
- Không thấy ngày tháng.
- Không thấy số tiền.

Bước xác thực số liệu tài chính sâu như:

```text
số lượng × đơn giá = thành tiền
tổng dòng = tổng tiền
VAT đúng tỷ lệ
```

được xem là **optional**.

Có thể chưa làm hoặc bỏ khỏi MVP nếu project chỉ cần OCR/RAG tốt.

Định hướng hiện tại:

```text
Ưu tiên extraction đầy đủ trước.
Validation số liệu để sau.
```

---

## 8. Dependencies mới

Cần thêm vào `requirements-core.txt`:

```text
paddleocr>=2.7
paddlepaddle>=2.6
pymupdf>=1.24
pdfplumber>=0.11
rapidfuzz>=3.9
python-dateutil>=2.9
```

Ghi chú local:

- Dùng `paddlepaddle` CPU trước để dễ cài trên Windows.
- Không cài Paddle GPU ở giai đoạn này.
- VLM vẫn chạy qua Ollama và model `qwen2.5vl:3b`.

---

## 9. Config đề xuất

Thêm vào `config/setting.yaml` về sau:

```yaml
ingestion:
  mode: "local_balanced"

  preprocessing:
    keep_original: true
    deskew: true
    aggressive_cleaning: false
    max_image_side: 1600

  ocr:
    enabled: true
    engine: "paddleocr"
    use_gpu: false
    languages:
      - "vi"
      - "en"

  vlm:
    enabled: true
    model: "qwen2.5vl:3b"
    temperature: 0.0
    timeout_seconds: 180

  extraction:
    output_json: true
    output_markdown: true
    confidence_threshold: 0.75
    retry_on_low_confidence: false
    max_retries: 1

  validation:
    financial_rules: false
```

---

## 10. Test plan hiện tại

Test với 2 ảnh trong:

```text
data/raw
```

Kỳ vọng sau test:

```text
data/processed/image.md
data/processed/image.json
data/processed/image copy.md
data/processed/image copy.json
```

Cần kiểm tra:

1. `.md` có bảng Markdown không.
2. `.json` parse được không.
3. `raw_ocr` có OCR blocks không.
4. `uncertain_tokens` có được ghi nhận không.
5. VLM có giữ nguyên ngôn ngữ gốc không.
6. Pipeline không crash nếu PaddleOCR/VLM lỗi.

---

## 11. Next steps

Thứ tự tiếp theo:

1. Cài dependencies mới trong venv.
2. Chạy test lại 2 ảnh trong `data/raw`.
3. So sánh output `.md` và `.json`.
4. Nếu PaddleOCR quá nặng/chậm, tối ưu language mode.
5. Nếu VLM output không phải JSON hợp lệ, chỉnh prompt thêm một vòng.
6. Sau khi ingestion ổn, mới tối ưu indexing/retrieval.
