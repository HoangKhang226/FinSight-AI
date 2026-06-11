"""
FinSight AI — Document Chunker
Chia nhỏ tài liệu Markdown thành các chunk (đoạn) dựa trên tiêu đề.
Cam kết 1: Không cắt đôi bảng (Markdown Table).
Cam kết 2: Đính kèm metadata để dễ tra cứu.
"""

from typing import List, Dict, Any
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from src.config import get_logger

logger = get_logger(__name__)


class DocumentChunker:
    """Class cắt tài liệu Markdown thành các chunks phục vụ Vector Search."""
    
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n\n", "\n\n", ".\n", "\n", " ", ""],
        )

    def chunk(self, markdown_text: str, source_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Cắt nhỏ file markdown.
        Trả về danh sách dictionary gồm content và metadata.
        """
        logger.info("Bắt đầu cắt chunk tài liệu Markdown.")
        
        if not source_metadata:
            source_metadata = {}
            
        md_header_splits = self.header_splitter.split_text(markdown_text)
        final_splits = self.text_splitter.split_documents(md_header_splits)
        
        chunks = []
        for split in final_splits:
            meta = {**source_metadata, **split.metadata}
            chunks.append({
                "content": split.page_content,
                "metadata": meta
            })
            
        logger.info(f"Hoàn tất. Tạo ra {len(chunks)} chunks.")
        return chunks
