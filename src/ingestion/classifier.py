"""
FinSight AI — Input Classifier
Phân loại tài liệu đầu vào:
1. PDF (Happy path)
2. Ảnh scan chuẩn (Happy path)
3. Ảnh chụp camera bị méo/sọc (Fallback path)
"""

from enum import Enum
from pathlib import Path
import mimetypes

from src.config import get_logger

logger = get_logger(__name__)


class InputType(Enum):
    DIGITAL_PDF = "DIGITAL_PDF"
    SCANNED_IMAGE = "SCANNED_IMAGE"
    CAMERA_PHOTO = "CAMERA_PHOTO"


class DocumentClassifier:
    """Class chịu trách nhiệm phân loại chất lượng tài liệu và định dạng đầu vào."""
    
    def __init__(self):
        pass

    def classify(self, file_path: str | Path) -> InputType:
        """Nhận file path và trả về InputType."""
        path = Path(file_path)
        mime_type, _ = mimetypes.guess_type(path)

        if mime_type == "application/pdf":
            logger.info(f"Phân loại PDF gốc: {path.name} -> DIGITAL_PDF")
            return InputType.DIGITAL_PDF

        if mime_type and mime_type.startswith("image/"):
            # Bỏ qua heuristic Laplacian, ép toàn bộ ảnh qua VLM để bóc tách bảng/Tiếng Việt tốt hơn
            logger.info(f"File ảnh: {path.name} -> Ép chạy qua VLM (CAMERA_PHOTO)")
            return InputType.CAMERA_PHOTO

        # Mặc định an toàn
        logger.warning(f"Không nhận diện được mime_type cho {path.name}. Rơi về CAMERA_PHOTO.")
        return InputType.CAMERA_PHOTO
