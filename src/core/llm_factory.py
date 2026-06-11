"""
FinSight AI — LLM Factory
Quản lý khởi tạo LLM qua Ollama (Qwen2-VL-2B cho vision, Qwen2.5-7B cho text).
"""

from langchain_ollama import ChatOllama
from src.config import settings, get_logger

logger = get_logger(__name__)


def get_text_llm(temperature: float = 0.0) -> ChatOllama:
    """Khởi tạo LLM cho tác vụ text (reasoning, extraction, routing)."""
    logger.info(f"Initializing text LLM: {settings.ollama_model}")
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=temperature,
    )


def get_vlm(temperature: float = 0.0) -> ChatOllama:
    """Khởi tạo VLM cho tác vụ vision (OCR hóa đơn, đọc biểu đồ)."""
    logger.info(f"Initializing VLM: {settings.vlm_model}")
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.vlm_model,
        temperature=temperature,
        num_ctx=8192,
    )
