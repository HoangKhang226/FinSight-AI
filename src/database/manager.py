"""
FinSight AI — Database Manager
Lớp Repository/Manager Pattern để giao tiếp với relational database.
Chứa các method tạo user, quản lý session, lưu tin nhắn, quản lý bộ nhớ dài hạn.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from src.config import get_logger
from src.database.models import ChatMessage, ChatSession, User, UserLongTermMemory

logger = get_logger(__name__)


class DatabaseManager:
    """Class quản lý các nghiệp vụ truy xuất và ghi vào Database SQL."""

    def __init__(self, db: Session):
        self.db = db

    # ── Users ────────────────────────────────────────────────────────────────

    def create_user(self, username: str, email: str) -> User:
        """Tạo người dùng mới."""
        user = User(username=username, email=email)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        logger.info(f"Created user: {user.username} ({user.id})")
        return user

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Tìm user theo ID."""
        return self.db.query(User).filter(User.id == user_id).first()

    # ── Chat Sessions ────────────────────────────────────────────────────────

    def create_session(
        self,
        user_id: str,
        title: str,
        metadata: dict | None = None,
    ) -> ChatSession:
        """Tạo phiên chat mới."""
        session = ChatSession(user_id=user_id, title=title, metadata_=metadata)
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        logger.info(f"Created session: {session.title} ({session.id})")
        return session

    def get_user_sessions(self, user_id: str) -> List[ChatSession]:
        """Lấy danh sách phiên chat của user, mới nhất trước."""
        return (
            self.db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .all()
        )

    # ── Chat Messages ────────────────────────────────────────────────────────

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tokens_used: int | None = None,
    ) -> ChatMessage:
        """Lưu 1 tin nhắn vào lịch sử chat."""
        msg = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            tokens_used=tokens_used,
        )
        self.db.add(msg)

        session = self.db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            session.updated_at = datetime.utcnow()

        self.db.commit()
        self.db.refresh(msg)
        return msg

    def get_recent_messages(self, session_id: str, limit: int = 10) -> List[ChatMessage]:
        """Bốc N tin nhắn gần nhất của phiên chat theo thứ tự thời gian tăng dần."""
        return (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
            .all()
        )[::-1]

    # ── Long-term Memory ────────────────────────────────────────────────────

    def get_user_memories(
        self,
        user_id: str,
        memory_type: str | None = None,
    ) -> List[UserLongTermMemory]:
        """Lấy tất cả bộ nhớ dài hạn của user hoặc lọc theo type."""
        query = self.db.query(UserLongTermMemory).filter(
            UserLongTermMemory.user_id == user_id
        )
        if memory_type:
            query = query.filter(UserLongTermMemory.memory_type == memory_type)
        return query.order_by(UserLongTermMemory.updated_at.desc()).all()

    def upsert_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
    ) -> UserLongTermMemory:
        """Cập nhật hoặc tạo mới bộ nhớ dài hạn."""
        existing = (
            self.db.query(UserLongTermMemory)
            .filter(
                UserLongTermMemory.user_id == user_id,
                UserLongTermMemory.memory_type == memory_type,
                UserLongTermMemory.content == content,
            )
            .first()
        )

        if existing:
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(existing)
            logger.info(f"Updated existing memory: {existing.id}")
            return existing

        memory = UserLongTermMemory(
            user_id=user_id,
            memory_type=memory_type,
            content=content,
        )
        self.db.add(memory)
        self.db.commit()
        self.db.refresh(memory)
        logger.info(f"Created new memory: {memory.memory_type} for user {user_id[:8]}")
        return memory
