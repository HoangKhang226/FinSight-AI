"""
FinSight AI — Relational Database (SQLite via SQLAlchemy)
Quản lý engine, session factory cho SQLite.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase

from src.config import settings, get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class cho tất cả ORM models."""
    pass


# ── Engine & Session Factory ─────────────────────────────────────────────
_engine = None
_SessionLocal = None


def get_engine():
    """Lấy hoặc tạo SQLAlchemy engine (singleton)."""
    global _engine
    if _engine is None:
        db_url = settings.database_url
        logger.info(f"Creating database engine: {db_url}")
        _engine = create_engine(
            db_url,
            echo=settings.database_echo,
            connect_args={"check_same_thread": False},  # SQLite cần cái này
        )
    return _engine


def get_session_factory() -> sessionmaker:
    """Lấy session factory (singleton)."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
        )
    return _SessionLocal


def get_db() -> Session:
    """
    Dependency injection cho FastAPI.
    Usage:
        @app.get("/")
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    session_factory = get_session_factory()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Tạo tất cả bảng trong database (chạy 1 lần khi khởi động)."""
    engine = get_engine()
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully.")
