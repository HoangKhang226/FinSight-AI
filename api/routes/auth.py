"""
FinSight AI — Auth Router
Mockup router cho Đăng ký / Đăng nhập.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.core.relational_db import get_db
from src.database.manager import DatabaseManager

router = APIRouter()


class UserCreate(BaseModel):
    username: str
    email: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str

    class Config:
        from_attributes = True


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    """Đăng ký user mới."""
    db_manager = DatabaseManager(db)
    # Kiểm tra đơn giản
    from src.database.models import User
    existing_user = db.query(User).filter(
        (User.username == user_in.username) | (User.email == user_in.email)
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already exists."
        )
    
    user = db_manager.create_user(username=user_in.username, email=user_in.email)
    return user


@router.post("/login")
def login(username: str, db: Session = Depends(get_db)):
    """Đăng nhập đơn giản bằng username (mockup)."""
    from src.database.models import User
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": "Login successful", "user_id": user.id}
