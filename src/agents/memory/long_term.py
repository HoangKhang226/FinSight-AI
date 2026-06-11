"""
FinSight AI — Long-term Memory
Agent ngầm đúc kết tri thức (rule, preference) cuối phiên.
"""

from sqlalchemy.orm import Session
from src.database.manager import DatabaseManager
from src.config import get_logger
from src.core.llm_factory import get_text_llm

logger = get_logger(__name__)


class LongTermMemoryWorker:
    """Class xử lý ngầm đúc kết tri thức hội thoại vào Long Term Memory."""
    
    def __init__(self, db: Session):
        self.db_manager = DatabaseManager(db)
        # self.llm = get_text_llm()

    def summarize_and_save(self, session_id: str, user_id: str) -> None:
        """
        Đọc toàn bộ chat của session_id, dùng LLM tóm tắt lại 
        các quy tắc kiểm toán (nếu có) rồi lưu vào user_longterm_memory.
        """
        messages = self.db_manager.get_recent_messages(session_id, limit=50)
        if not messages:
            return
            
        chat_text = "\n".join([f"{m.role}: {m.content}" for m in messages])
        
        logger.info(f"Đang đúc kết tri thức cho session {session_id[:8]}...")
        
        # TODO: Prompt LLM để trích xuất quy tắc / thói quen.
        # response = self.llm.invoke(...)
        
        mock_memory = "Cần chú ý đối soát kỹ thuế VAT của nhà cung cấp A."
        
        self.db_manager.upsert_memory(user_id=user_id, memory_type="vendor_alert", content=mock_memory)
        logger.info("Đã lưu memory thành công.")
