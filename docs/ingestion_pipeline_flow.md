# Bức Tranh Toàn Cảnh: Luồng Xử Lý Ingestion Pipeline

Tài liệu này hệ thống hóa luồng đi của **1 tấm ảnh chụp tài liệu** (ví dụ: ảnh hóa đơn chụp bằng điện thoại) từ lúc hệ thống nhận được cho đến khi ra được kết quả Markdown sạch sẽ, sẵn sàng cho RAG.

Toàn bộ quá trình nằm trong module `src/ingestion/pipeline.py` và được chia thành **6 Giai đoạn (Phases)**.

---

## 1. Phase 1: Phân loại & Định tuyến (Classification)

**Đầu vào:** Một file ảnh (`.png`, `.jpg`).

- Hệ thống đẩy ảnh qua `DocumentClassifier`.
- Phân tích siêu dữ liệu và cấu trúc file để xác định đây là `CAMERA_PHOTO` (Ảnh chụp điện thoại) hay `SCANNED_PDF` (Ảnh scan phẳng).
- **Kết quả:** Đưa ra quyết định định tuyến. Với `CAMERA_PHOTO`, bắt buộc phải đi qua Luồng Tiền Xử Lý Hình Học (Path Fallback).

## 2. Phase 2: Tiền xử lý Hình học (Preprocessing & Flattening)

**Mục đích:** Xóa bỏ độ cong vênh, bóng râm và độ méo của ảnh chụp điện thoại.

- File `path_fallback.py` sử dụng thư viện `OpenCV` để dò tìm các đường viền (contours) của tờ giấy.
- Áp dụng phép biến đổi g## 3. Phase 3: Bóc tách Ký tự Tọa độ (PaddleOCR)
  **Mục đích:** Cung cấp "bản đồ tọa độ" (Bounding Boxes) cho toàn bộ chữ trên ảnh.
- Tấm ảnh phẳng được đưa vào `PaddleOCR` (Engine siêu nhẹ chạy CPU).
- Lấy danh sách hàng chục/hàng trăm khối chữ kèm tọa độ để phục vụ cho các thuật toán hình học phía sau (nếu cần).

## 4. Phase 4: Quét thô bằng VLM Tầng 1 (VLM Structural OCR)

**Mục đích:** Dùng Vision-Language Model để "nhìn" tấm ảnh và tự động trích xuất nội dung ra Markdown chuẩn xác cấu trúc.

- Ảnh `cleaned_image.png` cùng với gợi ý chữ của PaddleOCR được đẩy thẳng vào **`qwen2.5vl:7b` (Tầng VLM 1)** thông qua `VLMOCRProcessor`.
- VLM này chịu trách nhiệm sinh ra cấu trúc tài liệu toàn diện (Bảng biểu, đoạn văn) dưới dạng Markdown.
- Đồng thời `layout_router.py` tính toán các chỉ số phức tạp để biết vùng nào VLM có nguy cơ "đọc nát" (Ví dụ: hóa đơn nhòe nhẹt).

## 5. Phase 5: Tái tạo cấu trúc (Geometry Fallback)

**Mục đích:** Đỡ đạn cho Tầng VLM 1. Nếu VLM 1 đọc hóa đơn bị lệch cột do ảnh quá rối, hệ thống dùng tọa độ thô của PaddleOCR ráp lại thành bảng.

- Dùng `table_region_detection.py` (Morphology) tìm vùng hộp.
- Dùng **1D DBSCAN** ráp cột và **Adaptive Row Chaining** ráp hàng.
- Kết quả: Nếu VLM 1 gãy cấu trúc, hệ thống có một chuỗi Văn bản phẳng/Bảng thô (Raw Fallback Text) để chữa cháy.

## 6. Phase 6: Hậu xử lý bằng VLM Tầng 2 (VLM Post-Correction)

**Mục đích:** Sửa các lỗi chính tả ở cấp độ pixel mà hệ thống nhận diện sai.

- Dòng văn bản thô (có thể từ Tầng 1 hoặc Fallback Tầng 5) được đẩy qua **`qwen2.5vl:7b` (Tầng VLM 2)** thông qua `LocalPostCorrector`.
- **Luật Thép:** VLM này chỉ được phép sửa lỗi đánh máy, _tuyệt đối không làm toán_ và không thay đổi cấu trúc số liệu.
- **Quality Gate:** Kiểm tra regex loại bỏ filler (VD: _"Here is..."_). Nếu VLM 2 ngoan, output được ghi nhận.

---

## 🏁 Đích đến

Kiến trúc **2 Tầng VLM** kết hợp với Mạng lưới An toàn Hình học (Geometry Fallback) đảm bảo dữ liệu Markdown ra đời sạch sẽ và trọn vẹn nhất, sẵn sàng lưu vào Qdrant cho RAG. Việc dùng chung model `qwen2.5vl:7b` cho cả Tầng 1 và Tầng 2 giúp VRAM không bị reset.

---

## Phụ lục: Sơ đồ luồng xử lý (Architecture Diagram)

```mermaid
graph TD
    classDef phase style fill:#f9f,stroke:#333,stroke-width:2px;
    classDef logic style fill:#bbf,stroke:#333,stroke-width:1px;
    classDef error style fill:#ff9999,stroke:#333,stroke-width:1px;
    classDef success style fill:#99ff99,stroke:#333,stroke-width:1px;

    Start([📷 Ảnh chụp .png/.jpg]) --> P1[Phase 1 & 2: Phân loại & Tiền xử lý góc nhìn]

    P1 --> P3[Phase 3: Quét BBox bằng PaddleOCR]

    P3 --> P4[Phase 4: VLM Tầng 1 quét thô<br>qwen2.5vl:7b sinh Markdown]

    P4 --> P5{Kiểm tra chất lượng cấu trúc<br>Layout Router}

    P5 -->|PASS: Cấu trúc tốt| P6[Phase 6: VLM Tầng 2 sửa chính tả]
    P5 -->|FAIL: Cấu trúc nát| P5_Fallback[Phase 5: Fallback Dựng hình học<br>DBSCAN & Row Chaining]

    P5_Fallback -->|Chuỗi text hàng/cột thô| P6

    subgraph Phase 6: Bộ não sửa lỗi VLM Tầng 2
        P6 --> P6_1[Gọi VLM qwen2.5vl:7b<br>Sửa chữ sai, cấm tính toán]
        P6_1 --> P6_2{Quality Gate}
        P6_2 -->|Lỗi/Filler| P6_Abort:::error
        P6_2 -->|Pass| P6_Success:::success
    end

    P6_Success & P6_Abort --> Out([📦 Đích đến: Markdown sạch cho Qdrant]) P4 -->|Nếu có cấu trúc bảng biểu| P5[Phase 5: Tái tạo cấu trúc bảng]
    P4 -->|Nếu toàn chữ xuông| P5_Fallback[Tạo đoạn text thô liên tục]

    subgraph Phase 5: Thuật toán dựng hàng cột bằng OpenCV & Thống kê
        P5 --> P5_1[Dò vùng hộp bảng - Morphology]
        P5_1 --> P5_2[Gom Cột - Thuật toán 1D DBSCAN]
        P5_1 --> P5_3[Gom Hàng - Adaptive Row Chaining]
    end

    P5_2 & P5_3 & P5_Fallback -->|Raw Markdown / Text thô dính chữ| P6[Phase 6: Hậu xử lý & Vá lỗi<br>post_correction.py]

    subgraph Phase 6: Bộ não sửa lỗi VLM & Chốt chặn Quality Gate
        P6 --> P6_1[Gọi Local VLM qwen2.5vl:7b<br>Chỉ sửa chính tả - Cấm làm toán]
        P6_1 --> P6_2{Quality Gate<br>Kiểm tra kết quả bằng Regex}

        P6_2 -->|FAIL: Dính chữ 'Here is...' hoặc Mất số| P6_Abort:::error
        P6_2 -->|PASS: Chỉ chứa text sạch| P6_Success:::success

        P6_Abort -->|Hủy kết quả VLM| P6_Fallback_Text[Dùng lại Text thô chưa sửa ban đầu]
        P6_Success -->|Lấy lõi Markdown| P6_Clean_Text[Đoạn Markdown hoàn hảo]
    end

    P6_Fallback_Text & P6_Clean_Text --> Out([📦 Đích đến: Markdown sạch + JSON Metadata])
    Out --> RAG[(Lưu vào Vector DB Qdrant phục vụ RAG)]
```
