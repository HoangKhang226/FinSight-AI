# Dưới đây là đề xuất kiến trúc Scale-up thuần túy về mặt AI System, tập trung vào 3 mắt xích: Redis Caching, LLM Gateway (Fallback/Orchestration), và Bất đồng bộ hóa Pipeline.

# 1. Tầng Đệm Thông Minh: Nhân Bản Hiệu Năng Với Redis Cache

Đối với hệ thống kiểm toán tài chính, dữ liệu thường có tính lặp lại cực cao (cùng một biểu mẫu hóa đơn của một nhà cung cấp, hoặc các câu hỏi tra cứu báo cáo tài chính giống nhau giữa các phòng ban). Chúng ta sẽ khai thác Redis theo 2 cơ chế nâng cao:

## Cơ chế 1: Image Hashing Cache (Né hoàn toàn tầng Ingestion)

### Cách vận hành:
Khi người dùng upload một ảnh hóa đơn, trước khi đẩy vào OpenCV hay Gemma 4 E4B, hệ thống sẽ băm bức ảnh đó thành một mã định danh duy nhất bằng thuật toán MD5 hoặc SHA-256 (Image Hash).

### Logic Cache:
- Sử dụng mã Hash này làm Key để tra cứu trên Redis.
- Nếu Key đã tồn tại (hóa đơn này đã từng được xử lý bởi một ai đó trong công ty) → Redis lập tức trả về file Markdown sạch và cấu trúc JSON số liệu đã bóc tách thành công trước đó.

### Kết quả:
Pipeline chạy mất 0 mili-giây, bỏ qua hoàn toàn việc chạy GPU/CPU, tiết kiệm 100% tài nguyên hệ thống.

## Cơ chế 2: Semantic Caching với RedisVL (Bộ nhớ đệm ngữ nghĩa)

### Vấn đề:
Người dùng A hỏi:

> "Thuế suất VAT của công ty X năm 2025 là bao nhiêu?"

Người dùng B hỏi:

> "Check hộ tiền thuế VAT của doanh nghiệp X trong năm 2025"

Bản chất hai câu này là một, nhưng OCR/Text truyền thống sẽ coi là khác nhau và bắt Agent chạy lại từ đầu.

### Giải pháp:
Sử dụng Redis VL (Vector Library) để làm bộ đệm ngữ nghĩa.

Câu hỏi của user được Vector hóa qua mô hình Embedding nhẹ.

Nếu độ tương đồng (Cosine Similarity) với một câu hỏi cũ trong Cache đạt trên 95%, Redis sẽ bốc luôn câu trả lời của Agent cũ ra trả về.

# 2. Tầng Điều Phối: Triển Khai LLM Gateway (LiteLLM / Portkey)

Khi đưa sản phẩm ra thực tế, con card local RTX 2050 với Gemma 4 E4B đóng vai trò là một Edge Worker (tiết kiệm chi phí, bảo mật nội bộ). Tuy nhiên, nếu 10 người cùng upload tài liệu một lúc, VRAM 4GB sẽ bị nghẽn (Queue).

LLM Gateway sẽ là vị "nhạc trưởng" giải quyết bài toán này.

## Kịch bản Fallback tự động (Failover Routing)

Hệ thống cấu hình Gateway theo quy tắc:

**Ưu tiên Local (Cost = 0) → Dự phòng Cloud (Pay-per-token)**

Gateway liên tục giám sát trạng thái của con Gemma 4 E4B chạy local qua Ollama.

Nếu Local phản hồi lỗi (Error 500), hoặc thời gian chờ vượt quá cấu hình (Timeout > 15 giây do hàng đợi quá tải), Gateway sẽ ngay lập tức bẻ lái luồng Request sang API Cloud (ví dụ: Gemini 1.5 Flash hoặc GPT-4o-mini) một cách âm thầm.

Người dùng ở giao diện frontend hoàn toàn không nhận ra hệ thống vừa gặp sự cố, đảm bảo ứng dụng hoạt động ổn định 24/7.

## Quản lý hạn mức và Tải trọng (Load Balancing & Rate Limiting)

Nếu bạn mở rộng thêm một máy tính khác cũng có card GPU (Ví dụ: máy của đồng nghiệp), LLM Gateway sẽ tự động chia tải (Round-robin) giữa 2 máy chạy Gemma 4 local, nhân đôi băng thông xử lý mà không cần đụng vào code ứng dụng chính.

# 3. Tầng Bất Đồng Bộ: Giải Cứu FastAPI Bằng Task Queue (Redis + Arq/Celery)

Một sai lầm phổ biến khiến hệ thống AI bị sập khi có nhiều người dùng là xử lý các tác vụ nặng (như chạy OpenCV, ép Gemma đọc ảnh) trực tiếp bên trong luồng xử lý HTTP của FastAPI.

Nếu file tài liệu dài 50 trang, HTTP Request sẽ bị treo giữ (Keep-alive) quá lâu dẫn đến sập server.

## Kiến trúc tách rời luồng (Decoupled Architecture)

### Bước 1
Người dùng upload tài liệu.

FastAPI tiếp nhận file, tạo ra một `Task_ID` duy nhất, đẩy file vào kho chứa và ném `Task_ID` này vào Redis Queue (Hàng đợi tin nhắn).

### Bước 2
FastAPI ngay lập tức trả về client mã phản hồi `HTTP 202 Accepted` kèm theo `Task_ID` (Mất chưa đầy 50 mili-giây).

Giao diện người dùng lập tức hiển thị trạng thái:

> "Đang xử lý tài liệu, vui lòng đợi..."

### Bước 3
Ở phía sau, các Sandbox Workers (chạy độc lập với FastAPI) sẽ âm thầm nhặt các tác vụ từ Redis Queue ra để xử lý tuần tự:

```text
OpenCV → Gemma 4 → LangGraph
```

### Bước 4
Sau khi Worker xử lý xong và ghi số liệu vào PostgreSQL, nó sẽ bắn một tín hiệu qua Redis để FastAPI stream kết quả về cho giao diện người dùng thông qua giao thức Server-Sent Events (SSE).

# Cấu trúc thư mục cập nhật cho phần Scale-up

Để hiện thực hóa kiến trúc này, cấu trúc cây thư mục của bạn sẽ được bổ sung thêm các phân hệ điều phối cực kỳ chuyên nghiệp:

```text
smart-financial-auditor/
│
├── deployment/
│   ├── docker-compose.yml          # Thêm container Redis, LiteLLM Gateway và Celery Worker
│   └── gateway_config.yaml         # File cấu hình quy tắc Fallback từ Gemma4 Local sang Gemini Cloud
│
├── src/
│   ├── core/
│   │   ├── redis_cache.py          # Khởi tạo RedisVL phục vụ Semantic Cache và Image Hashing
│   │   └── llm_factory.py          # Thay vì gọi Ollama trực tiếp, code giờ sẽ gọi qua LLM Gateway Endpoint
│   │
│   ├── workers/                    # THÊM MỚI: Nơi chứa mã nguồn của các Task Worker bất đồng bộ
│   │   ├── tasks.py                # Định nghĩa hàm chạy ngầm (E.g., run_async_pipeline)
│   │   └── celery_app.py           # Cấu hình kết nối hàng đợi Redis
```

Sự kết hợp giữa Redis (làm cả nhiệm vụ Cache ngữ nghĩa lẫn Hàng đợi tác vụ) và LLM Gateway (làm nhiệm vụ điều phối Fallback luồng) biến dự án của bạn từ một ứng dụng chạy thử nghiệm (Prototype) thành một giải pháp kiến trúc hệ thống AI thực thụ, sẵn sàng chịu tải cao trong môi trường doanh nghiệp.