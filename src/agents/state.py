"""
FinSight AI — Graph State
Định nghĩa cấu trúc dữ liệu truyền giữa các Agent trong LangGraph.
"""

from typing import TypedDict, Annotated, List, Optional
import operator

class AgentState(TypedDict):
    """
    Trạng thái của LangGraph. Dữ liệu này sẽ được truyền từ Node này sang Node khác.
    """
    query: str                                      # Câu hỏi của người dùng
    input_type: str                                 # "lookup" (tra cứu) hoặc "computation" (tính toán)
    context_chunks: List[str]                       # Danh sách các đoạn văn/bảng tìm được từ Retriever
    extracted_data: Optional[dict]                  # JSON số liệu thô trích xuất từ Extractor Agent
    code_result: Optional[str]                      # Kết quả từ Sandbox (Code Interpreter)
    confidence_score: Optional[float]               # Độ tin cậy (khi đọc ảnh)
    audit_report: Optional[str]                     # Báo cáo đối soát
    final_answer: str                               # Câu trả lời cuối cùng gửi cho user
    messages: Annotated[List[dict], operator.add]   # Lịch sử chat (để LLM nhớ context)
