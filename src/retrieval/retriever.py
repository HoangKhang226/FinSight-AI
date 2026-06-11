"""
FinSight AI — Hybrid Retriever
Thực hiện truy vấn Qdrant (Dense Vector) và kết hợp với kết quả từ BM25 (Sparse Vector).
(Phiên bản sử dụng Dense Vector Search).
"""

from typing import List, Dict, Any
from src.core.vector_db import get_qdrant_client
from src.config import settings, get_logger

from langchain_ollama import OllamaEmbeddings

logger = get_logger(__name__)


class HybridRetriever:
    """Class tìm kiếm tài liệu từ Vector DB theo phương pháp Hybrid (Dense + Sparse)."""
    
    def __init__(self, collection_name: str | None = None):
        self.collection_name = collection_name or settings.get("storage.vector_store.qdrant.collection_name", "finsight_collection")
        self.client = get_qdrant_client()
        self.embed_model = OllamaEmbeddings(
            model=settings.embed_model,
            base_url=settings.ollama_base_url
        )
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Tìm kiếm các chunk liên quan nhất đến câu hỏi.
        Trả về list payload (chứa content và metadata).
        """
        logger.info(f"Đang tìm kiếm ngữ cảnh cho câu hỏi: '{query}'")
        
        query_vector = self.embed_model.embed_query(query)
        
        # Dense Search
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        results = []
        for scored_point in search_result:
            results.append({
                "score": scored_point.score,
                "content": scored_point.payload.get("content", ""),
                "metadata": {k: v for k, v in scored_point.payload.items() if k != "content"}
            })
            
        logger.info(f"Tìm thấy {len(results)} chunks phù hợp.")
        return results
