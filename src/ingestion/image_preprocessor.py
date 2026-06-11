"""
FinSight AI — Fallback Path (Image Preprocessing)
Tiền xử lý ảnh lỗi bằng OpenCV trước khi ném vào VLM.
Tăng cường: Khử bóng, Xoay thẳng (Deskew), Cân bằng sáng.
"""

import cv2
import numpy as np
from pathlib import Path

from src.config import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    """Class cung cấp các công cụ xử lý ảnh bằng OpenCV để tăng cường chất lượng."""



    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """Xoay thẳng ảnh dựa trên góc nghiêng của các đoạn text."""
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        if coords.size == 0:
            return image

        angle = cv2.minAreaRect(coords)[-1]

        if angle > 45:
            angle = angle - 90

        if abs(angle) < 0.5:
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        logger.info(f"Đã xoay ảnh một góc {angle:.2f} độ.")
        return rotated

    @staticmethod
    def enhance_brightness_if_dark(image: np.ndarray) -> np.ndarray:
        """Tự động nhận diện ảnh tối và áp dụng CLAHE để làm sáng, giữ nguyên chi tiết chữ."""
        # Chuyển sang không gian màu LAB để tách riêng kênh sáng (Lightness)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        avg_brightness = np.mean(l)
        # Ngưỡng 110: Nếu L trung bình dưới mức này nghĩa là ảnh khá tối/thiếu sáng
        if avg_brightness < 110:
            logger.info(f"Phát hiện ảnh tối (độ sáng: {avg_brightness:.1f}/255). Tiến hành cân bằng sáng CLAHE...")
            # Sử dụng CLAHE để làm sáng nhưng không gây lóa các vùng vốn đã sáng
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            
        return image

    def process(self, input_path: str | Path, output_path: str | Path) -> str:
        """
        Hàm tổng hợp tiền xử lý. Nhận đường dẫn ảnh lỗi, xuất ra ảnh sạch.
        """
        in_path = str(input_path)
        out_path = str(output_path)
        
        logger.info(f"Bắt đầu tiền xử lý ảnh: {in_path}")
        img = cv2.imread(in_path)
        if img is None:
            raise FileNotFoundError(f"Không thể đọc ảnh: {in_path}")
        
        # 1. Làm sáng ảnh nếu quá tối (giúp VLM và OCR đọc rõ hơn)
        brightened = self.enhance_brightness_if_dark(img)

        # 2. Xoay thẳng ảnh nhẹ nhàng (Deskew) để PaddleOCR lấy BBox dễ hơn,
        # tuyệt đối không làm mờ nét chữ nguyên bản.
        deskewed = self.deskew(brightened)

        cv2.imwrite(out_path, deskewed)
        logger.info(f"Đã lưu ảnh đã xử lý tại: {out_path}")
        return out_path
