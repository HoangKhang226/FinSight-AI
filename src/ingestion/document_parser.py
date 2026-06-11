"""
FinSight AI — Happy Path (Docling)
Xử lý tài liệu PDF số hóa hoặc ảnh scan sắc nét bằng Docling/MinerU.
Chuyển đổi văn bản, bảng biểu thành Markdown.
"""

from pathlib import Path
from docling.document_converter import DocumentConverter

from src.config import get_logger

logger = get_logger(__name__)


class DocumentParser:
    """Class phân tách cấu trúc tài liệu PDF/Scan chuẩn ra Markdown."""
    
    def __init__(self):
        self.converter = DocumentConverter()
        
    def parse(self, file_path: str | Path) -> str:
        """
        Xử lý PDF / file chuẩn bằng Docling.
        Trả về định dạng Markdown.
        """
        path = Path(file_path)
        logger.info(f"Bắt đầu xử lý {path.name} bằng Docling (Happy Path)")
        
        try:
            # Convert tài liệu
            result = self.converter.convert(str(path))
            
            # Lấy nội dung markdown
            markdown_text = result.document.export_to_markdown()
            
            logger.info(f"Xử lý thành công: {path.name} ({len(markdown_text)} bytes)")
            return markdown_text
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý bằng Docling: {e}")
            # Rơi về fallback nếu có lỗi (nên thêm cơ chế retry / fallback)
            raise RuntimeError(f"Docling conversion failed: {e}")
