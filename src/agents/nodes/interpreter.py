"""
FinSight AI — Interpreter Node
Sinh mã Python dựa trên JSON data, chạy trong Sandbox và thu kết quả.
"""

from src.agents.state import AgentState
from src.config import get_logger

logger = get_logger(__name__)


class InterpreterNode:
    """Node sinh và thực thi Python code trong môi trường Sandbox."""
    
    def __init__(self):
        pass

    def run(self, state: AgentState) -> AgentState:
        logger.info("Node: Code Interpreter đang xử lý...")
        # TODO: Gọi Docker Sandbox API
        state["code_result"] = "42"
        return state
