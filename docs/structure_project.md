smart-financial-auditor/
│
├── data/ # Dữ liệu cục bộ (Chặn đẩy lên Git qua .gitignore)
│ ├── raw/ # Tài liệu gốc: Ảnh hóa đơn, PDF báo cáo tài chính
│ └── processed/ # Kết quả trung gian: Ảnh sau OpenCV, file Markdown sạch
│
├── deployment/ # Đóng gói và triển khai hệ thống (DevOps/MLOps)
│ ├── Dockerfile.api # Đóng gói FastAPI Backend ứng dụng
│ ├── Dockerfile.sandbox # Môi trường gác cổng (Sandboxed Worker) cách ly cho Python Code
│ └── docker-compose.yml # Điều phối toàn bộ cụm: API, Qdrant, PostgreSQL, Sandbox
│
├── evaluation/ # PIPELINE KIỂM THỬ ĐỘC LẬP (BENCHMARK SUITE)
│ ├── golden_test_suite/ # Bộ mẫu test phân tầng đã trích xuất từ Hugging Face
│ │ ├── invoice_math_test.json # 50 mẫu Voxel51 (Test OCR + Toán)
│ │ ├── layout_test.json # 30 mẫu mychen76 (Test đa dạng bố cục)
│ │ ├── rag_finance_test.json # 30 mẫu FinQA (Test RAG + Logic tài chính)
│ │ └── fallback_test.json # 40 mẫu WildReceipt + 30 mẫu FUNSD (Test ảnh nhiễu)
│ ├── metrics.py # Hàm tính toán toán học độc lập: CER, WER, Hit Rate, Exact Match
│ ├── sampler.py # Script Python tự động cào và lọc dữ liệu từ Hugging Face
│ └── run_benchmark.py # Script chạy kiểm thử tự động và xuất báo cáo (CSV/Excel)
│
├── src/ # MÃ NGUỒN CHÍNH CỦA ỨNG DỤNG
│ ├── **init**.py
│ ├── config.py # Quản lý cấu hình hệ thống, API Keys, phân bổ VRAM
│ │
│ ├── core/ # Khởi tạo các kết nối dùng chung (Singletons)
│ │ ├── vector_db.py # Cấu hình Qdrant Client (Vector DB)
│ │ ├── relational_db.py # Cấu hình SQLAlchemy/SQLModel SessionPool (PostgreSQL)
│ │ └── llm_factory.py # Quản lý khởi tạo Qwen2-VL (Local) và Qwen 2.5 7B GGUF
│ │
│ ├── ingestion/ # TẦNG 1: LAYOUT PARSING & IMAGE PREPROCESSING
│ │ ├── classifier.py # Phân loại đầu vào (PDF chuẩn số hóa hay Ảnh chụp lỗi camera)
│ │ ├── path_happy.py # Xử lý PDF chuẩn (Tích hợp Docling / MinerU)
│ │ ├── path_fallback.py # Tiền xử lý ảnh lỗi bằng OpenCV (Deskew, Khử bóng, Lọc ngưỡng)
│ │ └── vlm_ocr.py # Luồng VLM OCR bóc tách thực thể phức tạp hoặc biểu đồ
│ │
│ ├── retrieval/ # TẦNG 2: INDEXING & HYBRID RETRIEVAL (LlamaIndex)
│ │ ├── chunker.py # Cắt chunk tài liệu theo cấu trúc Markdown (Bảo toàn bảng biểu)
│ │ ├── indexer.py # Đồng bộ dữ liệu vào Qdrant (Dense) và cấu hình chỉ mục BM25 (Sparse)
│ │ └── retriever.py # Công cụ tìm kiếm lai Hybrid Search kết hợp Reranker
│ │
│ ├── database/ # TẦNG QUẢN LÝ QUAN HỆ & PHIÊN LÀM VIỆC (SQL)
│ │ ├── models.py # Định nghĩa ORM Schema (Users, ChatSessions, Messages, LongTermMemory)
│ │ ├── manager.py # DatabaseManager OOP: tạo phiên, lưu tin nhắn, cập nhật bộ nhớ
│ │ └── migrations/ # Thư mục quản lý phiên bản database (Alembic)
│ │
│ └── agents/ # TẦNG 3: AGENTIC REASONING & COMPUTATION (LangGraph)
│ ├── state.py # Định nghĩa cấu trúc dữ liệu truyền giữa các Agent (Graph State)
│ ├── workflow.py # Cấu hình liên kết các Node và Edge của đồ thị LangGraph
│ │
│ ├── memory/ # Hệ thống quản lý bộ nhớ của Agent
│ │ ├── short_term.py # Kéo/Lưu lịch sử hội thoại từ SQL vào bộ nhớ đệm Graph
│ │ └── long_term.py # Agent ngầm đúc kết tri thức, thói quen kiểm toán cuối phiên vào SQL
│ │
│ └── nodes/ # Mã nguồn logic độc lập của từng Agent
│ ├── router.py # Phân loại ý định: Tra cứu tài liệu hay Đối soát số liệu
│ ├── extractor.py # Trích xuất dữ liệu thô từ văn bản/Markdown sang định dạng JSON
│ ├── interpreter.py # Sinh mã Python và gọi Sandbox Worker thực thi tính toán
│ ├── auditor.py # Kiểm toán viên đối soát logic, phát hiện sai sót, tính confidence_score
│ └── qa_rag.py # Trả lời câu hỏi ngữ nghĩa dựa trên ngữ cảnh được bốc từ RAG
│
└── api/ # TẦNG 4: APPLICATION & INTERACTION (FastAPI)
├── routes/
│ ├── auth.py # Endpoint đăng ký, đăng nhập người dùng
│ ├── document.py # Endpoint tải lên tài liệu, kích hoạt luồng Ingestion
│ └── chat.py # Endpoint quản lý phiên, truyền stream câu trả lời (SSE)
└── server.py # Điểm khởi chạy FastAPI Server
