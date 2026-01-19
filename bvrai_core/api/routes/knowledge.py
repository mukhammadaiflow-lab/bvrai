"""
Knowledge Base API Routes

This module provides REST API endpoints for managing knowledge bases.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, UploadFile, File
from pydantic import BaseModel, Field

from ..base import (
    APIResponse,
    ListResponse,
    NotFoundError,
    ValidationError,
    success_response,
    paginated_response,
)
from ..auth import (
    AuthContext,
    Permission,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge-bases", tags=["Knowledge Bases"])


# =============================================================================
# Request/Response Models
# =============================================================================


class DocumentType(str):
    """Document types."""

    TEXT = "text"
    PDF = "pdf"
    URL = "url"
    FAQ = "faq"
    CSV = "csv"


class KnowledgeBaseCreateRequest(BaseModel):
    """Request to create a knowledge base."""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)

    # Embedding configuration
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model to use",
    )
    chunk_size: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Document chunk size",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between chunks",
    )

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseUpdateRequest(BaseModel):
    """Request to update a knowledge base."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)
    metadata: Optional[Dict[str, Any]] = None


class KnowledgeBaseResponse(BaseModel):
    """Knowledge base response."""

    id: str
    organization_id: str
    name: str
    description: Optional[str] = None

    # Configuration
    embedding_model: str
    chunk_size: int
    chunk_overlap: int

    # Stats
    document_count: int = 0
    chunk_count: int = 0
    total_tokens: int = 0

    # Agents using this KB
    agent_ids: List[str] = []

    # Metadata
    metadata: Dict[str, Any] = {}

    # Timestamps
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class DocumentCreateRequest(BaseModel):
    """Request to add a document to knowledge base."""

    type: str = Field(..., description="Document type")
    name: str = Field(..., description="Document name")

    # For text/FAQ type
    content: Optional[str] = Field(
        default=None,
        description="Document content (for text type)",
    )

    # For URL type
    url: Optional[str] = Field(
        default=None,
        description="URL to scrape (for URL type)",
    )

    # For FAQ type
    faqs: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="List of {question, answer} pairs",
    )

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    """Document response."""

    id: str
    knowledge_base_id: str
    type: str
    name: str
    status: str  # processing, ready, failed

    # Stats
    chunk_count: int = 0
    token_count: int = 0
    character_count: int = 0

    # Error info
    error_message: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = {}

    # Timestamps
    created_at: datetime
    updated_at: datetime


class SearchRequest(BaseModel):
    """Request to search knowledge base."""

    query: str = Field(..., description="Search query")
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results",
    )
    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Similarity threshold",
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filter",
    )


class SearchResult(BaseModel):
    """Search result."""

    chunk_id: str
    document_id: str
    document_name: str
    content: str
    score: float
    metadata: Dict[str, Any] = {}


# =============================================================================
# Routes
# =============================================================================


@router.post(
    "",
    response_model=APIResponse[KnowledgeBaseResponse],
    status_code=201,
    summary="Create Knowledge Base",
    description="Create a new knowledge base for storing documents.",
)
async def create_knowledge_base(
    request: KnowledgeBaseCreateRequest,
    auth: AuthContext = Depends(),
):
    """Create a knowledge base."""
    auth.require_permission(Permission.KNOWLEDGE_WRITE)

    kb = KnowledgeBaseResponse(
        id="kb_" + "x" * 24,
        organization_id=auth.organization_id,
        name=request.name,
        description=request.description,
        embedding_model=request.embedding_model,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        metadata=request.metadata,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    return success_response(kb.dict())


@router.get(
    "",
    response_model=ListResponse[KnowledgeBaseResponse],
    summary="List Knowledge Bases",
    description="List all knowledge bases.",
)
async def list_knowledge_bases(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None),
    auth: AuthContext = Depends(),
):
    """List knowledge bases."""
    auth.require_permission(Permission.KNOWLEDGE_READ)

    return paginated_response(
        items=[],
        page=page,
        page_size=page_size,
        total_items=0,
    )


@router.get(
    "/{kb_id}",
    response_model=APIResponse[KnowledgeBaseResponse],
    summary="Get Knowledge Base",
)
async def get_knowledge_base(
    kb_id: str = Path(..., description="Knowledge base ID"),
    auth: AuthContext = Depends(),
):
    """Get knowledge base by ID."""
    auth.require_permission(Permission.KNOWLEDGE_READ)

    raise NotFoundError("KnowledgeBase", kb_id)


@router.patch(
    "/{kb_id}",
    response_model=APIResponse[KnowledgeBaseResponse],
    summary="Update Knowledge Base",
)
async def update_knowledge_base(
    kb_id: str = Path(...),
    request: KnowledgeBaseUpdateRequest = Body(...),
    auth: AuthContext = Depends(),
):
    """Update a knowledge base."""
    auth.require_permission(Permission.KNOWLEDGE_WRITE)

    raise NotFoundError("KnowledgeBase", kb_id)


@router.delete(
    "/{kb_id}",
    status_code=204,
    summary="Delete Knowledge Base",
)
async def delete_knowledge_base(
    kb_id: str = Path(...),
    auth: AuthContext = Depends(),
):
    """Delete a knowledge base."""
    auth.require_permission(Permission.KNOWLEDGE_DELETE)

    raise NotFoundError("KnowledgeBase", kb_id)


# Document routes
@router.post(
    "/{kb_id}/documents",
    response_model=APIResponse[DocumentResponse],
    status_code=201,
    summary="Add Document",
    description="Add a document to a knowledge base.",
)
async def add_document(
    kb_id: str = Path(...),
    request: DocumentCreateRequest = Body(...),
    auth: AuthContext = Depends(),
):
    """Add a document to knowledge base."""
    auth.require_permission(Permission.KNOWLEDGE_WRITE)

    raise NotFoundError("KnowledgeBase", kb_id)


@router.post(
    "/{kb_id}/documents/upload",
    response_model=APIResponse[DocumentResponse],
    status_code=201,
    summary="Upload Document",
    description="Upload a file to a knowledge base.",
)
async def upload_document(
    kb_id: str = Path(...),
    file: UploadFile = File(...),
    name: Optional[str] = Query(None),
    auth: AuthContext = Depends(),
):
    """Upload a document file."""
    auth.require_permission(Permission.KNOWLEDGE_WRITE)

    # Validate file type
    allowed_types = [".pdf", ".txt", ".md", ".csv", ".json"]
    file_ext = "." + file.filename.split(".")[-1].lower() if file.filename else ""
    if file_ext not in allowed_types:
        raise ValidationError(f"File type not allowed. Allowed: {allowed_types}")

    raise NotFoundError("KnowledgeBase", kb_id)


@router.get(
    "/{kb_id}/documents",
    response_model=ListResponse[DocumentResponse],
    summary="List Documents",
)
async def list_documents(
    kb_id: str = Path(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    auth: AuthContext = Depends(),
):
    """List documents in a knowledge base."""
    auth.require_permission(Permission.KNOWLEDGE_READ)

    raise NotFoundError("KnowledgeBase", kb_id)


@router.get(
    "/{kb_id}/documents/{doc_id}",
    response_model=APIResponse[DocumentResponse],
    summary="Get Document",
)
async def get_document(
    kb_id: str = Path(...),
    doc_id: str = Path(...),
    auth: AuthContext = Depends(),
):
    """Get a document."""
    auth.require_permission(Permission.KNOWLEDGE_READ)

    raise NotFoundError("Document", doc_id)


@router.delete(
    "/{kb_id}/documents/{doc_id}",
    status_code=204,
    summary="Delete Document",
)
async def delete_document(
    kb_id: str = Path(...),
    doc_id: str = Path(...),
    auth: AuthContext = Depends(),
):
    """Delete a document."""
    auth.require_permission(Permission.KNOWLEDGE_DELETE)

    raise NotFoundError("Document", doc_id)


# Search
@router.post(
    "/{kb_id}/search",
    response_model=APIResponse[List[SearchResult]],
    summary="Search Knowledge Base",
    description="Search for relevant content in a knowledge base.",
)
async def search_knowledge_base(
    kb_id: str = Path(...),
    request: SearchRequest = Body(...),
    auth: AuthContext = Depends(),
):
    """Search a knowledge base."""
    auth.require_permission(Permission.KNOWLEDGE_READ)

    # In production, this would:
    # 1. Embed the query
    # 2. Search vector store
    # 3. Return ranked results

    raise NotFoundError("KnowledgeBase", kb_id)


@router.post(
    "/{kb_id}/reindex",
    response_model=APIResponse[Dict[str, Any]],
    summary="Reindex Knowledge Base",
    description="Reindex all documents in a knowledge base.",
)
async def reindex_knowledge_base(
    kb_id: str = Path(...),
    auth: AuthContext = Depends(),
):
    """Reindex a knowledge base."""
    auth.require_permission(Permission.KNOWLEDGE_WRITE)

    raise NotFoundError("KnowledgeBase", kb_id)
