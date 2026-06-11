"""
FinSight AI — Router Node
Phân tích intent của câu hỏi:
- Trả về 'lookup' nếu là tra cứu lý thuyết.
- Trả về 'computation' nếu cần tính toán, đối soát số liệu.
"""

from src.agents.state import AgentState
from src.core.llm_factory import get_text_llm
from src.config import get_logger

logger = get_logger(__name__)


class RouterNode:
    """Node phân loại intent câu hỏi người dùng."""
    
    def __init__(self):
        self.llm = get_text_llm()

    def run(self, state: AgentState) -> AgentState:
        logger.info("Node: Router đang xử lý...")
        # TODO: Implement LLM intent classification prompt
        state["input_type"] = "lookup"  # Mock default
        return state
