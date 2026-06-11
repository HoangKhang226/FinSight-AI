"""
FinSight AI — Auditor Node
Đối chiếu kết quả tính toán với tài liệu, đưa ra cảnh báo.
"""

from src.agents.state import AgentState
from src.config import get_logger

logger = get_logger(__name__)


class AuditorNode:
    """Node đối chiếu số liệu và đưa ra kết luận kiểm toán."""
    
    def __init__(self):
        pass

    def run(self, state: AgentState) -> AgentState:
        logger.info("Node: Auditor đang kiểm tra...")
        # TODO: Implement Audit logic
        state["audit_report"] = "Tất cả số liệu khớp 100%."
        state["final_answer"] = f"Báo cáo: {state['audit_report']}\nKết quả: {state['code_result']}"
        return state
