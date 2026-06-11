"""
FinSight AI — FastAPI Server
Điểm khởi chạy ứng dụng, cấu hình CORS, register routers, xử lý startup/shutdown.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings, setup_logging, get_logger
from src.core.relational_db import init_db
from src.core.vector_db import ensure_collection

from api.routes import auth, chat, document

# Khởi tạo logging từ config
setup_logging()
logger = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.get("app.version", "1.0.0"),
        debug=settings.debug,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Thay đổi trên production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Đăng ký các router
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(document.router, prefix="/api/v1/documents", tags=["Documents"])
    app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])

    @app.on_event("startup")
    async def startup_event():
        logger.info(f"Starting {settings.app_name} API Server...")
        
        # 1. Khởi tạo SQLite database (tạo bảng nếu chưa có)
        init_db()
        
        # 2. Đảm bảo Qdrant collection tồn tại
        ensure_collection()
        
        logger.info("Server is ready.")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down API Server.")

    @app.get("/health")
    def health_check():
        return {"status": "ok", "app": settings.app_name}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    # Khởi chạy server: python -m api.server
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
