"""
FinSight AI — Document Upload Router
Nhận file (PDF, ảnh), kích hoạt luồng Ingestion (Tầng 1).
"""

from fastapi import APIRouter, UploadFile, File, BackgroundTasks
import shutil
from pathlib import Path
import time

from src.config import settings, get_logger
from src.ingestion.pipeline import IngestionPipeline

logger = get_logger(__name__)
router = APIRouter()

RAW_DIR = Path(settings.get("storage.data.raw_dir", "data/raw"))
RAW_DIR.mkdir(parents=True, exist_ok=True)


PROCESSED_DIR = Path(settings.get("storage.data.processed_dir", "data/processed"))
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def process_document_bg(file_path_str: str):
    """
    Task chạy ngầm:
    1. Chạy local hybrid ingestion pipeline
    2. Lưu Markdown + JSON vào data/processed
    3. Cắt chunk -> index vào Qdrant (Tầng 2)
    """
    logger.info(f"[Ingestion Worker] Bắt đầu xử lý: {file_path_str}")
    start_time = time.time()

    try:
        pipeline = IngestionPipeline()
        result = pipeline.run(Path(file_path_str))
        pipeline.save_outputs(result)
        pipeline.index_result(result)

        elapsed = time.time() - start_time
        logger.info(
            "[Ingestion Worker] Hoàn thành pipeline. "
            f"confidence={result.confidence.overall}, "
            f"human_review={result.requires_human_review}, "
            f"elapsed={elapsed:.2f}s"
        )
    except Exception as e:
        logger.error(f"[Ingestion Worker] Lỗi xử lý {file_path_str}: {e}")


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Tải lên hóa đơn hoặc báo cáo. Server sẽ lưu file thô và kích hoạt Pipeline nền.
    """
    safe_name = file.filename.replace(" ", "_")
    file_path = RAW_DIR / safe_name
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    background_tasks.add_task(process_document_bg, str(file_path))
    
    return {
        "message": "File uploaded successfully. Processing started in background.",
        "filename": safe_name,
        "path": str(file_path)
    }
