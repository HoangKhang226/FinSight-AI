"""
FinSight AI — Short-term Memory
Load lịch sử hội thoại vào Graph State trước khi chạy.
"""

from typing import List, Dict, Any
from sqlalchemy.orm import Session
from src.database.manager import DatabaseManager
from src.config import get_logger

logger = get_logger(__name__)


class ShortTermMemory:
    """Class quản lý bộ nhớ ngắn hạn của phiên hội thoại thông qua SQL Database."""
    
    def __init__(self, db: Session):
        self.db_manager = DatabaseManager(db)

    def load_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Lấy N tin nhắn gần nhất từ DB SQL."""
        messages = self.db_manager.get_recent_messages(session_id, limit=limit)
        return [{"role": m.role, "content": m.content} for m in messages]
