"""
FinSight AI — Graph Workflow
Liên kết các Node thành đồ thị LangGraph hoàn chỉnh.
"""

from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.config import get_logger

# Import các node class
from src.agents.nodes.router import RouterNode
from src.agents.nodes.qa_rag import QARAGNode
from src.agents.nodes.extractor import ExtractorNode
from src.agents.nodes.interpreter import InterpreterNode
from src.agents.nodes.auditor import AuditorNode

logger = get_logger(__name__)


class AgentWorkflowBuilder:
    """Class chịu trách nhiệm xây dựng và cấu hình đồ thị LangGraph."""
    
    def __init__(self):
        self.workflow = StateGraph(AgentState)
        self._register_nodes()
        self._register_edges()

    def _register_nodes(self):
        self.workflow.add_node("router", RouterNode().run)
        self.workflow.add_node("qa_rag", QARAGNode().run)
        self.workflow.add_node("extractor", ExtractorNode().run)
        self.workflow.add_node("interpreter", InterpreterNode().run)
        self.workflow.add_node("auditor", AuditorNode().run)

    def _router_condition(self, state: AgentState) -> str:
        if state.get("input_type") == "computation":
            return "extractor"
        return "qa_rag"

    def _register_edges(self):
        self.workflow.set_entry_point("router")
        
        self.workflow.add_conditional_edges(
            "router",
            self._router_condition,
            {
                "extractor": "extractor",
                "qa_rag": "qa_rag"
            }
        )
        
        self.workflow.add_edge("extractor", "interpreter")
        self.workflow.add_edge("interpreter", "auditor")
        
        self.workflow.add_edge("qa_rag", END)
        self.workflow.add_edge("auditor", END)

    def build(self):
        """Compile thành LangGraph workflow."""
        logger.info("Building Agent Workflow Graph...")
        return self.workflow.compile()
