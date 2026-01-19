"""
Knowledge Base - Base Classes and Data Structures

This module defines the foundational types for the knowledge base system.
"""

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"
    MD = "markdown"
    HTML = "html"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    PPTX = "pptx"
    XLSX = "xlsx"
    RTF = "rtf"
    EPUB = "epub"
    URL = "url"
    AUDIO = "audio"        # Transcribed audio
    VIDEO = "video"        # Transcribed video
    IMAGE = "image"        # OCR'd image
    CODE = "code"          # Source code
    UNKNOWN = "unknown"


@dataclass
class DocumentMetadata:
    """Metadata for a document."""

    # Source information
    source: str = ""                    # File path, URL, or identifier
    source_type: DocumentType = DocumentType.UNKNOWN

    # Content metadata
    title: str = ""
    author: str = ""
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    language: str = "en"

    # Processing metadata
    processed_at: float = field(default_factory=time.time)
    processing_version: str = "1.0"

    # Size information
    file_size_bytes: int = 0
    page_count: int = 0
    word_count: int = 0
    char_count: int = 0

    # Organization
    organization_id: Optional[str] = None
    collection_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    # Custom metadata
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "source_type": self.source_type.value,
            "title": self.title,
            "author": self.author,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "language": self.language,
            "processed_at": self.processed_at,
            "file_size_bytes": self.file_size_bytes,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "organization_id": self.organization_id,
            "collection_id": self.collection_id,
            "tags": self.tags,
            "categories": self.categories,
            "custom": self.custom,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """Create from dictionary."""
        return cls(
            source=data.get("source", ""),
            source_type=DocumentType(data.get("source_type", "unknown")),
            title=data.get("title", ""),
            author=data.get("author", ""),
            created_at=data.get("created_at"),
            modified_at=data.get("modified_at"),
            language=data.get("language", "en"),
            processed_at=data.get("processed_at", time.time()),
            file_size_bytes=data.get("file_size_bytes", 0),
            page_count=data.get("page_count", 0),
            word_count=data.get("word_count", 0),
            char_count=data.get("char_count", 0),
            organization_id=data.get("organization_id"),
            collection_id=data.get("collection_id"),
            tags=data.get("tags", []),
            categories=data.get("categories", []),
            custom=data.get("custom", {}),
        )


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""

    # Parent document
    document_id: str = ""
    document_source: str = ""

    # Position in document
    chunk_index: int = 0
    total_chunks: int = 0
    start_char: int = 0
    end_char: int = 0
    page_number: Optional[int] = None

    # Content information
    char_count: int = 0
    word_count: int = 0
    token_count: int = 0

    # Chunking information
    chunking_strategy: str = ""
    overlap_chars: int = 0

    # Section information
    section_title: Optional[str] = None
    section_level: int = 0
    parent_chunk_id: Optional[str] = None

    # Semantic information
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)

    # Custom metadata (inherited from document + chunk-specific)
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "document_source": self.document_source,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "page_number": self.page_number,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "token_count": self.token_count,
            "chunking_strategy": self.chunking_strategy,
            "overlap_chars": self.overlap_chars,
            "section_title": self.section_title,
            "section_level": self.section_level,
            "parent_chunk_id": self.parent_chunk_id,
            "summary": self.summary,
            "keywords": self.keywords,
            "entities": self.entities,
            "custom": self.custom,
        }


@dataclass
class SearchFilter:
    """Filters for search queries."""

    # Document-level filters
    document_ids: Optional[List[str]] = None
    source_types: Optional[List[DocumentType]] = None
    tags: Optional[List[str]] = None
    categories: Optional[List[str]] = None

    # Organization filters
    organization_id: Optional[str] = None
    collection_id: Optional[str] = None

    # Date filters
    created_after: Optional[float] = None
    created_before: Optional[float] = None

    # Custom metadata filters
    custom_filters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for vector store queries."""
        filters = {}

        if self.document_ids:
            filters["document_id"] = {"$in": self.document_ids}

        if self.source_types:
            filters["source_type"] = {"$in": [t.value for t in self.source_types]}

        if self.tags:
            filters["tags"] = {"$in": self.tags}

        if self.categories:
            filters["categories"] = {"$in": self.categories}

        if self.organization_id:
            filters["organization_id"] = self.organization_id

        if self.collection_id:
            filters["collection_id"] = self.collection_id

        if self.created_after:
            filters["created_at"] = {"$gte": self.created_after}

        if self.created_before:
            if "created_at" in filters:
                filters["created_at"]["$lte"] = self.created_before
            else:
                filters["created_at"] = {"$lte": self.created_before}

        filters.update(self.custom_filters)

        return filters


@dataclass
class SearchQuery:
    """Configuration for a search query."""

    query: str
    top_k: int = 5

    # Search type
    use_semantic: bool = True      # Vector similarity search
    use_keyword: bool = False      # BM25/keyword search
    use_hybrid: bool = False       # Combine semantic + keyword

    # Hybrid search weights
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3

    # Score thresholds
    min_score: float = 0.0
    max_results: int = 100

    # Filters
    filters: Optional[SearchFilter] = None

    # Reranking
    rerank: bool = False
    rerank_model: Optional[str] = None
    rerank_top_k: Optional[int] = None

    # Query expansion
    expand_query: bool = False
    num_expansions: int = 3

    # Result options
    include_metadata: bool = True
    include_embeddings: bool = False
    include_scores: bool = True


@dataclass
class RetrievalResult:
    """Result from knowledge retrieval."""

    # Chunk information
    chunk_id: str
    content: str
    metadata: ChunkMetadata

    # Scoring
    score: float = 0.0
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    rerank_score: Optional[float] = None

    # Vector (optional)
    embedding: Optional[List[float]] = None

    # Source context
    document_title: str = ""
    source: str = ""

    # Surrounding context (for better understanding)
    prev_chunk_content: Optional[str] = None
    next_chunk_content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "score": self.score,
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
            "rerank_score": self.rerank_score,
            "document_title": self.document_title,
            "source": self.source,
        }

    def to_context_string(self, include_source: bool = True) -> str:
        """Convert to string for LLM context."""
        parts = []

        if include_source and (self.document_title or self.source):
            source_info = self.document_title or self.source
            parts.append(f"[Source: {source_info}]")

        parts.append(self.content)

        return "\n".join(parts)


def generate_document_id(
    source: str,
    content_hash: Optional[str] = None,
) -> str:
    """Generate a unique document ID."""
    if content_hash:
        return f"doc_{hashlib.sha256((source + content_hash).encode()).hexdigest()[:16]}"
    return f"doc_{hashlib.sha256(source.encode()).hexdigest()[:16]}_{uuid.uuid4().hex[:8]}"


def generate_chunk_id(
    document_id: str,
    chunk_index: int,
    content_hash: Optional[str] = None,
) -> str:
    """Generate a unique chunk ID."""
    base = f"{document_id}_{chunk_index}"
    if content_hash:
        return f"chunk_{hashlib.sha256((base + content_hash).encode()).hexdigest()[:16]}"
    return f"chunk_{hashlib.sha256(base.encode()).hexdigest()[:16]}"


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate token count for text."""
    return int(len(text) / chars_per_token)


def detect_document_type(filename: str) -> DocumentType:
    """Detect document type from filename."""
    ext = filename.lower().split(".")[-1] if "." in filename else ""

    type_map = {
        "pdf": DocumentType.PDF,
        "docx": DocumentType.DOCX,
        "doc": DocumentType.DOC,
        "txt": DocumentType.TXT,
        "md": DocumentType.MD,
        "markdown": DocumentType.MD,
        "html": DocumentType.HTML,
        "htm": DocumentType.HTML,
        "csv": DocumentType.CSV,
        "json": DocumentType.JSON,
        "xml": DocumentType.XML,
        "pptx": DocumentType.PPTX,
        "xlsx": DocumentType.XLSX,
        "rtf": DocumentType.RTF,
        "epub": DocumentType.EPUB,
        "py": DocumentType.CODE,
        "js": DocumentType.CODE,
        "ts": DocumentType.CODE,
        "java": DocumentType.CODE,
        "cpp": DocumentType.CODE,
        "c": DocumentType.CODE,
        "go": DocumentType.CODE,
        "rs": DocumentType.CODE,
    }

    return type_map.get(ext, DocumentType.UNKNOWN)


__all__ = [
    "DocumentType",
    "DocumentMetadata",
    "ChunkMetadata",
    "SearchFilter",
    "SearchQuery",
    "RetrievalResult",
    "generate_document_id",
    "generate_chunk_id",
    "estimate_tokens",
    "detect_document_type",
]
