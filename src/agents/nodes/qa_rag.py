"""
FinSight AI — QA RAG Node
Trả lời câu hỏi dạng Text/Ngữ nghĩa dựa trên context lấy từ Retriever.
"""

from src.agents.state import AgentState
from src.config import get_logger

logger = get_logger(__name__)


class QARAGNode:
    """Node trả lời các câu hỏi tra cứu thông tin thông thường (QA)."""
    
    def __init__(self):
        pass

    def run(self, state: AgentState) -> AgentState:
        logger.info("Node: QA RAG đang xử lý...")
        # TODO: Tích hợp Retriever
        state["final_answer"] = "Đây là câu trả lời Mock từ QA RAG."
        return state
