# 1. Bối Cảnh Bài Toán & Lý Do Triển Khai (Context & Motivation)

## Bối cảnh bài toán

Trong mọi doanh nghiệp, quy trình kiểm toán tài chính, đối soát hóa đơn và phân tích báo cáo luôn tốn nhiều nhân lực.

Đặc thù của tài liệu tài chính là sự hỗn hợp về cấu trúc (Semi-structured & Unstructured Data):

### Hóa đơn (Invoices/Receipts)

Định dạng thay đổi liên tục, chứa nhiều bảng không đường kẻ (borderless tables), con dấu đè lên chữ. Đặc biệt, trong thực tế, nhân viên đi hiện trường thường chỉ chụp lại ảnh hóa đơn bằng điện thoại (ảnh bị sọc, nghiêng, đổ bóng) chứ không có file PDF "sạch".

### Báo cáo tài chính (Financial Statements)

Các bảng cân đối kế toán có cấu trúc phân cấp lồng nhau phức tạp kèm hệ thống ghi chú (footnotes) chữ nhỏ ở cuối trang.

### Báo cáo phân tích thị trường

Chứa các biểu đồ trực quan (charts, graphs) thể hiện xu hướng tăng trưởng.

## Lý do triển khai dự án này

### Sự bế tắc của OCR truyền thống

Các giải pháp cũ chỉ chuyển ảnh thành văn bản phẳng từ trái sang phải, làm xáo trộn các cột số liệu và bỏ qua 100% biểu đồ. Nếu ảnh chụp bị nghiêng, OCR truyền thống hoàn toàn bị đọc lệch dòng.

### Rủi ro từ sự "ảo giác" của LLM

Nếu đẩy trực tiếp văn bản thô vào LLM, mô hình rất dễ tự "bịa" ra các con số hoặc tính toán sai các phép tính cơ bản do bản chất của LLM là dự đoán từ tiếp theo chứ không phải một máy tính logic.

### Nhu cầu kiểm tra chéo tự động

Doanh nghiệp cần một hệ thống có khả năng phát hiện sai sót logic (ví dụ: $Đơn\ giá \times Số\ lượng \neq Thành\ tiền$, hoặc Tổng tiền thuế bị tính lệch).

---

# 2. Kiến Trúc Tổng Quan Toàn Diện (System Architecture)

Để xử lý được cả file PDF số hóa lẫn ảnh chụp thực tế từ điện thoại, Tầng 1 của hệ thống được thiết kế theo dạng Rẽ nhánh thông minh (Dynamic Routing Pipeline).

```text
                      [ Tài liệu đầu vào ]
                                │
                                ▼
                  { Kiểm tra loại hình đầu vào }
                                │
         ┌──────────────────────┴──────────────────────┐
         ▼ (Luồng chuẩn: PDF/Ảnh scan thẳng)          ▼ (Luồng Fallback: Ảnh chụp điện thoại)

   ┌───────────┐                                 ┌───────────┐
   │  Docling  │                                 │ OpenCV /  │ ---> Cân sáng, khử nhiễu,
   │  MinerU   │                                 │  Kornia   │      xoay thẳng ảnh (Deskew)
   └─────┬─────┘                                 └─────┬─────┘
         │                                             │
         │                                             ▼
         │                                       ┌───────────┐
         │                                       │ Qwen2-VL  │ ---> Trích xuất End-to-End
         │                                       │ (Vision)  │      giữ cấu trúc bằng Prompt
         │                                       └─────┬─────┘
         │                                             │
         └──────────────────────┬──────────────────────┘
                                ▼
                     [ File Markdown sạch ]
```

---

# 3. Phân Tích Chi Tiết Các Tầng Của Bài Toán (Layer-by-Layer Breakdown)

# Tầng 1: Data Ingestion & Layout Parsing Layer (Tầng Thu Thập & Phân Tách Bố Cục)

Tầng này tiếp nhận file đầu vào (PDF, PNG, JPG), nhận diện loại hình đầu vào và quyết định luồng xử lý tương ứng.

## Bước 1: Phân loại đầu vào (Input Classification)

Hệ thống kiểm tra Metadata và ma trận pixel để xác định xem đây là PDF số hóa (Digital Native), ảnh scan phẳng, hay là một bức ảnh chụp từ camera (Scene Text/Camera Photo).

## Luồng 1 (Happy Path - Dành cho file PDF/Ảnh scan chuẩn)

- Sử dụng Docling hoặc MinerU để phân tích bounding box, chia tài liệu thành 3 thực thể: Text thường, Table (Bảng), và Figure (Biểu đồ/Hình ảnh).
- Chuyển bảng thành Markdown Table.
- Trích xuất Figure gửi qua Qwen2-VL-2B để mô tả biểu đồ thành văn bản.

## Luồng 2 (Fallback Path - Dành cho ảnh chụp điện thoại, mất góc, mờ nhòe)

### Tiền xử lý ảnh (Image Preprocessing)

Sử dụng OpenCV hoặc Kornia để tự động phát hiện góc (Perspective Transformation), xoay thẳng ảnh (Deskew), tăng độ tương phản và khử bóng mờ.

### VLM-First Extraction

Vì ảnh chụp thực tế có độ nhiễu cao khiến các bộ parse layout như Docling dễ thất bại, hệ thống sẽ bỏ qua bộ parse layout và đẩy thẳng ảnh đã tiền xử lý vào mạng thị giác của Qwen2-VL (hoặc API Gemini 1.5 Flash).

Sử dụng cấu trúc Prompt chuyên dụng (System Prompt) ép VLM quét ảnh theo cơ chế trượt (sliding window) nếu ảnh dài, nhận diện chữ viết tay/con dấu và tái cấu trúc toàn bộ nội dung trong ảnh chụp thành định dạng Markdown chuẩn xác.

## Đầu ra của Tầng 1

Một file Markdown tổng hợp hoàn chỉnh, sạch sẽ, cấu trúc bảng biểu đã được chuẩn hóa bất kể nguồn đầu vào là gì.

---

# Tầng 2: Multi-Modal Indexing & Retrieval Layer (Tầng Đánh Chỉ Mục & Tra Cứu)

Tầng này đảm bảo việc tra cứu thông tin con số chính xác tuyệt đối, không tìm kiếm gần đúng bừa bãi.

## Hierarchical Chunking

Chia nhỏ file Markdown theo cấu trúc tiêu đề (Header-based) kết hợp câu lệnh phân tách bảng.

Các bảng biểu phải được giữ nguyên trong một chunk, không được cắt đôi bảng.

## Hybrid Indexing

### Dense Vector Index

Dùng BGE-M3 hoặc nomic-embed-text để nhúng các đoạn văn bản mô tả ngữ nghĩa vào Vector DB (Qdrant).

### Sparse Keyword Index (BM25)

Đánh chỉ mục từ khóa chính xác cho các thực thể quan trọng như:

- Mã số thuế
- Số hóa đơn
- Ngày tháng
- Ký hiệu tiền tệ

để phục vụ tra cứu chính xác số liệu.

## Hybrid Retrieval & Reranking

Kết hợp kết quả từ Vector Search và BM25, đưa qua bộ bge-reranker-large để chọn ra top 3 chunks có độ liên quan cao nhất làm ngữ cảnh (Context) cho Agent.

---

# Tầng 3: Agentic Reasoning & Computation Layer (Tầng Suy Luận & Tính Toán)

Được xây dựng trên LangGraph dưới dạng một Đồ thị trạng thái có hướng (Stateful Directed Graph).

## Router Agent

Phân tích câu hỏi.

- Nếu câu hỏi dạng tra cứu ngữ nghĩa, chuyển sang QA RAG Agent.
- Nếu liên quan đến tính toán, đối soát số liệu, chuyển sang Extraction Agent.

## Extraction Agent

Nhận ngữ cảnh từ Tầng 2, trích xuất các con số cần thiết và chuyển thành một cấu trúc dữ liệu JSON sạch.

### Lưu ý cho luồng Fallback

Nếu dữ liệu đến từ luồng ảnh chụp có độ mờ cao, Agent sẽ trích xuất kèm theo một trường là `confidence_score`.

## Python Code Interpreter Agent (Hộp cát tính toán)

Nhận file JSON số liệu, tự động sinh mã Python để thực hiện phép tính toán học.

Mã được thực thi trong môi trường sandbox an toàn nhằm đảm bảo tính chính xác 100%, loại bỏ hoàn toàn hiện tượng ảo giác của LLM.

## Auditor Agent (Kiểm toán viên)

Đọc kết quả từ Code Interpreter, đối chiếu lại với tài liệu gốc.

Nếu phát hiện lỗi lệch số (ví dụ: tiền thuế tính lại lệch so với tiền thuế in trên ảnh), Agent sẽ đưa ra cảnh báo lỗi cụ thể.

Nếu `confidence_score` từ ảnh chụp quá thấp dẫn đến nghi ngờ số liệu, Agent sẽ chủ động phản hồi:

> "Hóa đơn có dấu hiệu mờ/sai số tại vùng [Tên Trường], vui lòng xác nhận lại hoặc cung cấp ảnh chụp rõ hơn".

---

# Tầng 4: Application & API Layer (Tầng Ứng Dụng & Giao Diện)

## Backend

FastAPI quản lý các endpoint nhận file/ảnh, lưu session chat, và truyền stream câu trả lời (Server-Sent Events - SSE) về giao diện theo thời gian thực.

## Frontend UI (Split-screen)

### Bên trái

Hiển thị file gốc.

Nếu là luồng Fallback, giao diện sẽ hiển thị song song ảnh chụp gốc của người dùng và ảnh đã được xử lý làm nét/xoay thẳng bởi hệ thống để người dùng tiện đối chiếu.

### Bên phải

Là khung cửa sổ Chat với Agent.

# 1. Bảng Đặc Tả Bộ Dữ Liệu Kiểm Thử (Golden Test Suite Specification)

Để chứng minh được các con số trong CV, bạn cần gom cấu trúc dữ liệu từ các nguồn mở thành một bộ dữ liệu kiểm thử phân tầng với số lượng mẫu cụ thể như sau:

| Tầng Hệ Thống Cần Test            | Tên Bộ Dataset Gốc                          | Số Lượng Mẫu (Size) | Thuật Toán Lọc Mẫu Tự Động (Metadata Filter)                                                   | Mục Tiêu Chứng Minh (Concept)                                                        |
| --------------------------------- | ------------------------------------------- | ------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Tầng 1: Bóc tách & Đối soát toán  | Voxel51/high-quality-invoice-images-for-ocr | 50 ảnh              | - 25 mẫu ngẫu nhiên cố định.<br>- 25 mẫu có số dòng sản phẩm > 10.                             | Đo năng lực đọc số gốc và khả năng viết vòng lặp xử lý bảng dài của Agent.           |
| Tầng 1: Đa dạng bố cục            | mychen76/invoices-and-receipts_ocr_v2       | 30 ảnh              | - 15 mẫu ngẫu nhiên cố định.<br>- 15 mẫu có tỷ lệ ảnh Cao/Rộng > 2.5 (Biên lai siêu thị).      | Đo độ bền bỉ của mô hình khi vị trí các trường số học bị thay đổi hoặc bị lệch dòng. |
| Tầng 2: RAG & Tính toán tài chính | TheFinAI/FinQA_v2                           | 30 tài liệu         | - 15 mẫu ngẫu nghĩa cố định.<br>- 15 mẫu câu hỏi chứa từ khóa: growth, margin, percentage.     | Đo năng lực tìm kiếm bảng biểu trong file lớn và thuật toán giải toán chuỗi.         |
| Tầng 3: Luồng Fallback (Ảnh lỗi)  | Theivaprakasham/wildreceipt                 | 40 ảnh              | - 20 ảnh có góc nghiêng chữ > 15 độ.<br>- 20 ảnh có độ sáng trung bình thấp (ảnh tối/đổ bóng). | Đo hiệu năng thực tế của các bộ lọc xử lý hình ảnh OpenCV trước khi đưa vào VLM.     |
| Tầng 3: Tài liệu nhiễu mực        | davidle7/funsd-json                         | 30 ảnh              | - 15 ảnh có độ tương phản pixel thấp nhất.<br>- 15 mẫu ngẫu nhiên cấu trúc.                    | Đo khả năng khôi phục nét chữ mờ nhạt của thuật toán lọc ngưỡng.                     |

# 2. Ma Trận Đánh Giá (Evaluation Matrix) & Phương Pháp Chấm Điểm

Với mỗi mẫu dữ liệu được nạp vào, hệ thống kiểm thử sẽ thực hiện đối chiếu văn bản giữa kết quả đầu ra của AI (Prediction) và đáp án chuẩn (Ground Truth) theo các tiêu chí sau:

## Tầng Bóc Tách Hóa Đơn (Invoice Extraction)

**Dữ liệu đầu vào:** Ảnh hóa đơn thô.

**Đáp án chuẩn (Ground Truth):** File JSON chứa giá trị chính xác của các trường:

* Total
* VAT
* Subtotal

### Cách chấm điểm ngữ nghĩa

Hệ thống sử dụng thuật toán tính khoảng cách ký tự (Levenshtein Distance) để đo độ tương đồng giữa chuỗi chữ AI đọc được và chuỗi chữ đáp án.

Kết quả trả về tỷ lệ tương đồng chuỗi văn bản (**ANLS Metric Score**).

### Cách chấm điểm số học (Match Rate)

Ép toán học so khớp tuyệt đối.

Giá trị số của trường Total hệ thống trích xuất ra phải trùng khớp hoàn toàn từng chữ số với đáp án.

Sai một chữ số hàng đơn vị cũng tính là 0 điểm cho ca test đó.

## Tầng Đọc Bảng Biểu & Báo Cáo Tài Chính (RAG & Agent)

**Dữ liệu đầu vào:** File PDF báo cáo tài chính dài + Câu hỏi yêu cầu tính toán.

**Đáp án chuẩn (Ground Truth):**

* ID của đoạn văn/bảng biểu chứa câu trả lời
* Con số đáp án cuối cùng của phép tính

### Cách chấm điểm Tầng RAG (Hit Rate)

Hệ thống kiểm tra xem danh sách 3 đoạn văn (Chunks) mà tầng RAG bốc lên từ Vector DB có chứa cái ID của đoạn văn đáp án hay không.

Nếu có nằm trong Top 3, tính là 1 điểm (Hit), nếu không có tính là 0 điểm (Miss).

### Cách chấm điểm Tầng Agent (Exact Match)

Hệ thống lấy con số đầu ra cuối cùng sau khi Agent chạy code Python đối chiếu với con số đáp án chuẩn.

Phép toán tài chính bắt buộc phải khớp 100% giá trị số học.

## Luồng Fallback (Xử lý ảnh chụp lỗi điện thoại)

**Dữ liệu đầu vào:** Ảnh chụp camera bị nghiêng, tối, mờ.

**Đáp án chuẩn (Ground Truth):** Chuỗi văn bản thô chuẩn của tờ hóa đơn.

### Cách chấm điểm (OCR Accuracy Gain)

Đây là bài toán đo mức độ cải thiện của thuật toán hình ảnh.

Ca test này sẽ được chạy qua 2 luồng:

#### Luồng A (Đối chứng)

Đẩy thẳng ảnh lỗi vào VLM đọc → Tính tỷ lệ lỗi ký tự (CER gốc).

#### Luồng B (Cải tiến)

Đẩy ảnh qua bộ lọc OpenCV (Xoay thẳng, khử bóng, làm nét) rồi mới đưa vào VLM đọc → Tính tỷ lệ lỗi ký tự (AI sau xử lý).

### Số liệu chứng minh

Hiệu số chênh lệch giữa tỷ lệ lỗi của Luồng A và Luồng B chính là mức tăng trưởng độ chính xác (Accuracy Gain) tạo ra bởi tầng OpenCV của bạn.

# 3. Cách Khai Thác Số Liệu Để Đưa Vào Báo Cáo Và CV

Sau khi chạy qua toàn bộ các bộ mẫu test phân tầng ở trên, bạn sẽ thu được các chỉ số tổng hợp (Aggregation Metrics) thuần túy bằng toán học để điền thẳng vào CV:

## Tỷ lệ tra cứu đúng của RAG (Retrieval Hit Rate @K=3)

[
\text{Hit Rate} = \left( \frac{\text{Tổng số ca bốc trúng bảng biểu đúng}}{\text{Tổng số câu hỏi mang đi test}} \right) \times 100
]

(Con số này chứng minh giải pháp cắt chunk Markdown và Hybrid Search của bạn hoạt động hiệu quả).

## Độ chính xác của Agent tính toán (Exact Match)

[
\text{Exact Match} = \left( \frac{\text{Số lần Agent chạy code ra kết quả đúng chóc}}{\text{Tổng số ca test yêu cầu tính toán}} \right) \times 100
]

(Con số này bảo chứng cho việc Python Code Interpreter đã triệt tiêu hoàn toàn lỗi ảo giác toán học của LLM).

## Mức độ giảm thiểu lỗi của tầng xử lý ảnh (CER Reduction Rate)

[
\text{Mức giảm lỗi} = \left( \frac{\text{Tỷ lệ lỗi ảnh thô} - \text{Tỷ lệ lỗi ảnh sau OpenCV}}{\text{Tỷ lệ lỗi ảnh thô}} \right) \times 100
]

(Con số này là bằng chứng chứng minh tầng tiền xử lý OpenCV thực sự cứu được mô hình VLM khi gặp ảnh chất lượng thấp ngoài đời thực).
