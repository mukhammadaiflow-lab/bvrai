"""Knowledge base schemas."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    """Schema for creating a document."""

    title: str = Field(..., max_length=255)
    content: str
    source: Optional[str] = None
    metadata: Optional[dict] = Field(default_factory=dict)


class DocumentUpdate(BaseModel):
    """Schema for updating a document."""

    title: Optional[str] = Field(None, max_length=255)
    content: Optional[str] = None
    metadata: Optional[dict] = None


class DocumentResponse(BaseModel):
    """Schema for document response."""

    id: UUID
    knowledge_base_id: UUID
    title: str
    content: str
    source: Optional[str] = None
    chunk_count: int = 0
    metadata: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class KnowledgeBaseCreate(BaseModel):
    """Schema for creating a knowledge base."""

    name: str = Field(..., max_length=255)
    description: Optional[str] = None
    settings: Optional[dict] = Field(default_factory=dict)


class KnowledgeBaseUpdate(BaseModel):
    """Schema for updating a knowledge base."""

    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    settings: Optional[dict] = None


class KnowledgeBaseResponse(BaseModel):
    """Schema for knowledge base response."""

    id: UUID
    agent_id: UUID
    name: str
    description: Optional[str] = None
    document_count: int = 0
    total_chunks: int = 0
    settings: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class KnowledgeBaseListResponse(BaseModel):
    """Paginated list of knowledge bases."""

    items: list[KnowledgeBaseResponse]
    total: int
    page: int
    page_size: int


class SearchQuery(BaseModel):
    """Schema for knowledge base search."""

    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    filter_metadata: Optional[dict] = None
    threshold: float = Field(default=0.7, ge=0, le=1)


class SearchResult(BaseModel):
    """Schema for search result."""

    document_id: UUID
    document_title: str
    chunk_id: str
    text: str
    score: float
    metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Schema for search response."""

    query: str
    results: list[SearchResult]
    total_results: int


class IngestRequest(BaseModel):
    """Request to ingest content."""

    content: str
    title: Optional[str] = None
    source: Optional[str] = None
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=200)


class IngestResponse(BaseModel):
    """Response after ingesting content."""

    document_id: UUID
    chunk_count: int
    message: str


class WebCrawlRequest(BaseModel):
    """Request to crawl a web page."""

    url: str
    max_depth: int = Field(default=1, ge=1, le=5)
    max_pages: int = Field(default=10, ge=1, le=100)
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)


class WebCrawlResponse(BaseModel):
    """Response after crawling."""

    pages_crawled: int
    documents_created: int
    total_chunks: int
    errors: list[str] = Field(default_factory=list)
