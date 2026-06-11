"""
FinSight AI — Vector Database (Qdrant Local)
Quản lý Qdrant client chạy local mode (embedded), không cần Docker khi dev.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config import settings, get_logger

logger = get_logger(__name__)

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Lấy hoặc tạo Qdrant client (singleton, local mode)."""
    global _client
    if _client is None:
        db_path = settings.get("storage.vector_store.qdrant.vector_db_dir", "storage/vector_db")
        logger.info(f"Initializing Qdrant client (local mode): {db_path}")
        _client = QdrantClient(path=db_path)
    return _client


def ensure_collection(
    collection_name: str | None = None,
    vector_size: int | None = None,
) -> None:
    """Tạo collection nếu chưa tồn tại."""
    client = get_qdrant_client()
    name = collection_name or settings.get(
        "storage.vector_store.qdrant.collection_name", "finsight_collection"
    )
    size = vector_size or settings.get("storage.vector_store.qdrant.vector_size", 768)
    distance_str = settings.get("storage.vector_store.qdrant.distance", "cosine")
    distance = Distance.COSINE if distance_str.lower() == "cosine" else Distance.EUCLID

    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        logger.info(f"Creating Qdrant collection: {name} (dim={size}, dist={distance_str})")
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=size, distance=distance),
        )
    else:
        logger.info(f"Qdrant collection '{name}' already exists.")
