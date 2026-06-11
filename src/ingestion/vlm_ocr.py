"""
FinSight AI — VLM OCR
Trích xuất thông tin end-to-end từ hình ảnh chụp (đã qua OpenCV) 
Sử dụng Qwen2-VL thông qua Ollama.
"""

from pathlib import Path
import base64

from langchain_core.messages import HumanMessage
from src.core.llm_factory import get_vlm
from src.config import get_logger

logger = get_logger(__name__)


class VLMOCRProcessor:
    """Class bóc tách nội dung ảnh bằng Vision Language Model."""
    
    def __init__(self, temperature: float = 0.0):
        self.vlm = get_vlm(temperature=temperature)
        self.default_prompt = (
            "You are a multilingual document OCR system. "
            "Extract all content faithfully, preserve the original language, "
            "construct valid Markdown tables, and never hallucinate data."
        )

    def _encode_image(self, image_path: Path) -> str:
        """Chuyển ảnh sang base64 để gửi vào prompt của LangChain."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def extract(self, image_path: str | Path, system_prompt: str | None = None, ocr_blocks: list | None = None) -> str:
        """
        Sử dụng VLM local để đọc nội dung ảnh.
        Nếu có ocr_blocks, pipeline sẽ truyền prompt đã format sẵn từ prompts/vlm_prompts.py.
        """
        path = Path(image_path)
        logger.info(f"Đang bóc tách ảnh bằng VLM: {path.name}")
        
        base64_image = self._encode_image(path)
        prompt = system_prompt or self.default_prompt
        
        # Tạo message theo chuẩn LangChain cho multi-modal
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )
        
        try:
            response = self.vlm.invoke([message])
            logger.info("VLM bóc tách thành công.")
            return response.content
        except Exception as e:
            logger.error(f"Lỗi khi gọi VLM: {e}")
            raise RuntimeError(f"VLM OCR failed: {e}")
