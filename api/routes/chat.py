"""
FinSight AI — Chat Router
Quản lý phiên làm việc, stream SSE (Server-Sent Events) kết quả từ LangGraph.
"""

import asyncio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from src.core.relational_db import get_db
from src.database.manager import DatabaseManager

router = APIRouter()


class SessionCreate(BaseModel):
    user_id: str
    title: str


class ChatRequest(BaseModel):
    query: str


@router.post("/sessions")
def create_chat_session(req: SessionCreate, db: Session = Depends(get_db)):
    """Tạo một phiên làm việc mới."""
    db_manager = DatabaseManager(db)
    user = db_manager.get_user_by_id(req.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    session = db_manager.create_session(req.user_id, req.title)
    return {"session_id": session.id, "title": session.title}


@router.get("/sessions/{session_id}/history")
def get_session_history(session_id: str, limit: int = 10, db: Session = Depends(get_db)):
    """Lấy lịch sử hội thoại gần nhất."""
    db_manager = DatabaseManager(db)
    messages = db_manager.get_recent_messages(session_id, limit=limit)
    return [
        {"role": m.role, "content": m.content, "timestamp": m.created_at}
        for m in messages
    ]


async def graph_stream_generator(session_id: str, query: str):
    """
    Generator để stream từng chunk từ LangGraph về client thông qua SSE.
    """
    # 1. Gửi tin nhắn của User vào short-term memory (SQL)
    # TODO: db = next(get_db()) -> save_message(...)
    
    yield {"data": f"Đã nhận yêu cầu: {query}\n"}
    await asyncio.sleep(0.5)
    
    yield {"data": "Đang chạy Router Agent...\n"}
    await asyncio.sleep(0.5)
    
    # 2. Chạy LangGraph stream events
    # TODO: Tích hợp src/agents/workflow.py vào đây
    
    yield {"data": "Code Interpreter đang đối soát...\n"}
    await asyncio.sleep(1)
    
    yield {"data": "✅ Kết quả cuối cùng: Mọi thứ đều khớp!"}


@router.post("/sessions/{session_id}/ask")
async def chat_with_agent(session_id: str, req: ChatRequest):
    """
    Nhận câu hỏi từ User, trả về Server-Sent Events stream từ Agent.
    """
    # TODO: Kiểm tra session có tồn tại không
    return EventSourceResponse(graph_stream_generator(session_id, req.query))
