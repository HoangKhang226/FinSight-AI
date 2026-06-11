"""
FinSight AI — Extraction Node
Trích xuất con số từ text/bảng thành JSON chuẩn bị cho Code Interpreter.
"""

from src.agents.state import AgentState
from src.config import get_logger

logger = get_logger(__name__)


class ExtractorNode:
    """Node trích xuất dữ liệu tài chính thô thành định dạng JSON."""
    
    def __init__(self):
        pass

    def run(self, state: AgentState) -> AgentState:
        logger.info("Node: Extractor đang xử lý...")
        # TODO: Implement extraction prompt
        state["extracted_data"] = {}
        return state
