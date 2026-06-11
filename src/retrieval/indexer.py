"""
FinSight AI — Indexer
Chịu trách nhiệm đồng bộ dữ liệu vào Qdrant (Dense Vector) và cấu hình BM25 (Sparse Vector).
"""

import uuid
from typing import List, Dict, Any
from qdrant_client.models import PointStruct
from src.core.vector_db import get_qdrant_client, ensure_collection
from src.config import settings, get_logger

from langchain_ollama import OllamaEmbeddings

logger = get_logger(__name__)


class VectorIndexer:
    """Class nhúng dữ liệu và đưa vào Qdrant Vector DB."""
    
    def __init__(self, collection_name: str | None = None):
        self.collection_name = collection_name or settings.get("storage.vector_store.qdrant.collection_name", "finsight_collection")
        self.client = get_qdrant_client()
        self.embed_model = OllamaEmbeddings(
            model=settings.embed_model,
            base_url=settings.ollama_base_url
        )
        
    def index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Nhúng các chunks thành Vector và đưa vào Qdrant.
        """
        if not chunks:
            logger.warning("Không có chunk nào để index.")
            return

        ensure_collection(self.collection_name)

        logger.info(f"Tiến hành nhúng (embedding) {len(chunks)} chunks...")
        
        texts = [c["content"] for c in chunks]
        embeddings = self.embed_model.embed_documents(texts)
        
        points = []
        for chunk, vector in zip(chunks, embeddings):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"content": chunk["content"], **chunk["metadata"]}
                )
            )
            
        logger.info(f"Đang đẩy {len(points)} points vào Qdrant collection '{self.collection_name}'.")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info("Index hoàn tất.")
