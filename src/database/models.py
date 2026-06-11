"""
FinSight AI — ORM Models
Định nghĩa 4 bảng: users, chat_sessions, chat_messages, user_longterm_memory.
Theo thiết kế trong docs/db.md.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship

from src.core.relational_db import Base


def _generate_uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    """Bảng users — Quản lý người dùng (kế toán viên / kiểm toán viên)."""

    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=_generate_uuid)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("UserLongTermMemory", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User {self.username}>"


class ChatSession(Base):
    """Bảng chat_sessions — Mỗi phiên upload hóa đơn / báo cáo tạo 1 session."""

    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=_generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=False)
    metadata_ = Column("metadata", JSON, nullable=True)  # Danh sách file PDF, context phụ
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<ChatSession {self.title[:30]}>"


class ChatMessage(Base):
    """
    Bảng chat_messages — Lịch sử hội thoại (Short-term Memory).
    Mỗi lần hỏi, bốc lại N tin nhắn gần nhất làm ngữ cảnh.
    """

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(36),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role = Column(String(20), nullable=False)  # 'user' | 'assistant' | 'system'
    content = Column(Text, nullable=False)
    tokens_used = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self) -> str:
        return f"<ChatMessage role={self.role} session={self.session_id[:8]}>"


class UserLongTermMemory(Base):
    """
    Bảng user_longterm_memory — Tri thức đúc kết (Long-term Memory).
    Cuối phiên, Memory Agent tóm tắt thói quen / quy tắc kiểm toán
    và upsert vào đây. Phiên sau Agent tự động bốc lên dùng.
    """

    __tablename__ = "user_longterm_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    memory_type = Column(String(50), nullable=False)  # 'preference' | 'vendor_alert' | 'rule'
    content = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="memories")

    def __repr__(self) -> str:
        return f"<LongTermMemory type={self.memory_type} user={self.user_id[:8]}>"
