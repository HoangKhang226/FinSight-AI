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
