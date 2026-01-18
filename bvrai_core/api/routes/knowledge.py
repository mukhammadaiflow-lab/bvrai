"""
Knowledge Base API Routes

This module provides REST API endpoints for managing knowledge bases.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Path, Body, UploadFile, File
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..base import (
    APIResponse,
    ListResponse,
    NotFoundError,
    ValidationError,
    success_response,
    paginated_response,
)
from ..auth import AuthContext, Permission
from ..dependencies import get_db_session, get_auth_context
from ...database.repositories import KnowledgeBaseRepository


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge-bases", tags=["Knowledge Bases"])


# =============================================================================
# Request/Response Models
# =============================================================================


class KnowledgeBaseCreateRequest(BaseModel):
    """Request to create a knowledge base."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=500)

    # Embedding configuration
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model to use",
    )
    embedding_provider: str = Field(
        default="openai",
        description="Embedding provider",
    )
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Document chunk size",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Overlap between chunks",
    )

    # Vector store
    vector_store: str = Field(default="qdrant", description="Vector store: qdrant, pinecone, weaviate")

    # Metadata
    extra_data: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseUpdateRequest(BaseModel):
    """Request to update a knowledge base."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=500)
    status: Optional[str] = Field(default=None)


class KnowledgeBaseResponse(BaseModel):
    """Knowledge base response."""

    id: str
    organization_id: str
    name: str
    description: Optional[str] = None

    # Configuration
    embedding_model: str
    embedding_provider: str
    chunk_size: int
    chunk_overlap: int
    vector_store: str
    vector_collection: Optional[str] = None

    # Status
    status: str = "active"

    # Stats
    document_count: int = 0
    chunk_count: int = 0
    total_tokens: int = 0

    # Timestamps
    last_synced_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class DocumentCreateRequest(BaseModel):
    """Request to add a document to knowledge base."""

    name: str = Field(..., description="Document name", max_length=255)
    doc_type: str = Field(default="text", description="Document type: text, pdf, url, faq, csv")
    description: Optional[str] = Field(default=None, max_length=500)

    # For text/FAQ type
    content: Optional[str] = Field(
        default=None,
        description="Document content (for text type)",
    )

    # For URL type
    source_url: Optional[str] = Field(
        default=None,
        description="URL to scrape (for URL type)",
    )

    # Metadata
    extra_data: Dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    """Document response."""

    id: str
    knowledge_base_id: str
    name: str
    description: Optional[str] = None
    doc_type: str
    status: str  # pending, processing, completed, failed

    # Content info
    source_url: Optional[str] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None

    # Stats
    chunk_count: int = 0
    token_count: int = 0

    # Error info
    error_message: Optional[str] = None

    # Timestamps
    processed_at: Optional[datetime] = None
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


class SearchResult(BaseModel):
    """Search result."""

    chunk_id: str
    document_id: str
    document_name: str
    content: str
    score: float
    chunk_metadata: Dict[str, Any] = {}


# =============================================================================
# Helper Functions
# =============================================================================


def kb_to_response(kb) -> dict:
    """Convert database knowledge base model to response dict."""
    return {
        "id": kb.id,
        "organization_id": kb.organization_id,
        "name": kb.name,
        "description": kb.description,
        "embedding_model": kb.embedding_model,
        "embedding_provider": kb.embedding_provider,
        "chunk_size": kb.chunk_size,
        "chunk_overlap": kb.chunk_overlap,
        "vector_store": kb.vector_store,
        "vector_collection": kb.vector_collection,
        "status": kb.status,
        "document_count": kb.document_count,
        "chunk_count": kb.chunk_count,
        "total_tokens": kb.total_tokens,
        "last_synced_at": kb.last_synced_at,
        "created_at": kb.created_at,
        "updated_at": kb.updated_at,
    }


def doc_to_response(doc) -> dict:
    """Convert database document model to response dict."""
    return {
        "id": doc.id,
        "knowledge_base_id": doc.knowledge_base_id,
        "name": doc.name,
        "description": doc.description,
        "doc_type": doc.doc_type,
        "status": doc.status,
        "source_url": doc.source_url,
        "file_path": doc.file_path,
        "file_size": doc.file_size,
        "mime_type": doc.mime_type,
        "chunk_count": doc.chunk_count,
        "token_count": doc.token_count,
        "error_message": doc.error_message,
        "processed_at": doc.processed_at,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
    }


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
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Create a knowledge base."""
    auth.require_permission(Permission.KNOWLEDGE_WRITE)

    repo = KnowledgeBaseRepository(db)

    kb = await repo.create(
        organization_id=auth.organization_id,
        name=request.name,
        description=request.description,
        embedding_model=request.embedding_model,
        embedding_provider=request.embedding_provider,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        vector_store=request.vector_store,
        extra_data=request.extra_data,
        status="active",
    )

    await db.commit()

    logger.info(f"Created knowledge base {kb.id} for org {auth.organization_id}")

    return success_response(kb_to_response(kb))


@router.get(
    "",
    response_model=ListResponse[KnowledgeBaseResponse],
    summary="List Knowledge Bases",
    description="List all knowledge bases.",
)
async def list_knowledge_bases(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """List knowledge bases."""
    auth.require_permission(Permission.KNOWLEDGE_READ)

    repo = KnowledgeBaseRepository(db)

    skip = (page - 1) * page_size
    kbs = await repo.list_by_organization(
        organization_id=auth.organization_id,
        status=status,
        skip=skip,
        limit=page_size,
    )

    total = await repo.count_by_organization(
        organization_id=auth.organization_id,
        status=status,
    )

    items = [kb_to_response(kb) for kb in kbs]

    return paginated_response(
        items=items,
        page=page,
        page_size=page_size,
        total_items=total,
    )


@router.get(
    "/{kb_id}",
    response_model=APIResponse[KnowledgeBaseResponse],
    summary="Get Knowledge Base",
)
async def get_knowledge_base(
    kb_id: str = Path(..., description="Knowledge base ID"),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Get knowledge base by ID."""
    auth.require_permission(Permission.KNOWLEDGE_READ)

    repo = KnowledgeBaseRepository(db)
    kb = await repo.get_by_id(kb_id)

    if not kb or kb.organization_id != auth.organization_id or kb.is_deleted:
        raise NotFoundError("KnowledgeBase", kb_id)

    return success_response(kb_to_response(kb))


@router.patch(
    "/{kb_id}",
    response_model=APIResponse[KnowledgeBaseResponse],
    summary="Update Knowledge Base",
)
async def update_knowledge_base(
    kb_id: str = Path(...),
    request: KnowledgeBaseUpdateRequest = Body(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Update a knowledge base."""
    auth.require_permission(Permission.KNOWLEDGE_WRITE)

    repo = KnowledgeBaseRepository(db)
    kb = await repo.get_by_id(kb_id)

    if not kb or kb.organization_id != auth.organization_id or kb.is_deleted:
        raise NotFoundError("KnowledgeBase", kb_id)

    update_data = {}
    if request.name is not None:
        update_data["name"] = request.name
    if request.description is not None:
        update_data["description"] = request.description
    if request.status is not None:
        update_data["status"] = request.status

    if update_data:
        kb = await repo.update(kb_id, **update_data)

    await db.commit()

    logger.info(f"Updated knowledge base {kb_id}")

    return success_response(kb_to_response(kb))


@router.delete(
    "/{kb_id}",
    status_code=204,
    summary="Delete Knowledge Base",
)
async def delete_knowledge_base(
    kb_id: str = Path(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Delete a knowledge base."""
    auth.require_permission(Permission.KNOWLEDGE_DELETE)

    repo = KnowledgeBaseRepository(db)
    kb = await repo.get_by_id(kb_id)

    if not kb or kb.organization_id != auth.organization_id or kb.is_deleted:
        raise NotFoundError("KnowledgeBase", kb_id)

    await repo.soft_delete(kb_id)
    await db.commit()

    logger.info(f"Deleted knowledge base {kb_id}")

    return None


# =============================================================================
# Document Routes
# =============================================================================


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
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Add a document to knowledge base."""
    auth.require_permission(Permission.KNOWLEDGE_WRITE)

    repo = KnowledgeBaseRepository(db)
    kb = await repo.get_by_id(kb_id)

    if not kb or kb.organization_id != auth.organization_id or kb.is_deleted:
        raise NotFoundError("KnowledgeBase", kb_id)

    # Validate based on type
    if request.doc_type == "text" and not request.content:
        raise ValidationError("Content is required for text documents", {"content": "required"})
    if request.doc_type == "url" and not request.source_url:
        raise ValidationError("Source URL is required for URL documents", {"source_url": "required"})

    doc = await repo.add_document(
        knowledge_base_id=kb_id,
        organization_id=auth.organization_id,
        name=request.name,
        doc_type=request.doc_type,
        content=request.content,
        source_url=request.source_url,
        extra_data=request.extra_data,
    )

    await db.commit()

    logger.info(f"Added document {doc.id} to knowledge base {kb_id}")

    # In production, this would trigger background processing to:
    # 1. Extract/scrape content
    # 2. Chunk the content
    # 3. Generate embeddings
    # 4. Store in vector database

    return success_response(doc_to_response(doc))


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
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Upload a document file."""
    auth.require_permission(Permission.KNOWLEDGE_WRITE)

    repo = KnowledgeBaseRepository(db)
    kb = await repo.get_by_id(kb_id)

    if not kb or kb.organization_id != auth.organization_id or kb.is_deleted:
        raise NotFoundError("KnowledgeBase", kb_id)

    # Validate file type
    allowed_types = {".pdf": "pdf", ".txt": "text", ".md": "text", ".csv": "csv", ".json": "text"}
    file_ext = "." + file.filename.split(".")[-1].lower() if file.filename else ""
    if file_ext not in allowed_types:
        raise ValidationError(
            f"File type not allowed. Allowed: {list(allowed_types.keys())}",
            {"file": "invalid_type"},
        )

    doc_name = name or file.filename or "Uploaded Document"
    doc_type = allowed_types.get(file_ext, "text")

    # Read file content
    content = await file.read()
    file_size = len(content)

    # For text files, decode and store content
    text_content = None
    if doc_type == "text":
        try:
            text_content = content.decode("utf-8")
        except UnicodeDecodeError:
            text_content = content.decode("latin-1")

    doc = await repo.add_document(
        knowledge_base_id=kb_id,
        organization_id=auth.organization_id,
        name=doc_name,
        doc_type=doc_type,
        content=text_content,
        file_size=file_size,
        mime_type=file.content_type,
    )

    await db.commit()

    logger.info(f"Uploaded document {doc.id} to knowledge base {kb_id}")

    return success_response(doc_to_response(doc))


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
    doc_type: Optional[str] = Query(None),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """List documents in a knowledge base."""
    auth.require_permission(Permission.KNOWLEDGE_READ)

    repo = KnowledgeBaseRepository(db)
    kb = await repo.get_by_id(kb_id)

    if not kb or kb.organization_id != auth.organization_id or kb.is_deleted:
        raise NotFoundError("KnowledgeBase", kb_id)

    skip = (page - 1) * page_size
    docs = await repo.list_documents(
        knowledge_base_id=kb_id,
        status=status,
        doc_type=doc_type,
        skip=skip,
        limit=page_size,
    )

    # Estimate total (could add count method)
    total = len(docs) if len(docs) < page_size else page_size * 2

    items = [doc_to_response(doc) for doc in docs]

    return paginated_response(
        items=items,
        page=page,
        page_size=page_size,
        total_items=total,
    )


@router.get(
    "/{kb_id}/documents/{doc_id}",
    response_model=APIResponse[DocumentResponse],
    summary="Get Document",
)
async def get_document(
    kb_id: str = Path(...),
    doc_id: str = Path(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Get a document."""
    auth.require_permission(Permission.KNOWLEDGE_READ)

    repo = KnowledgeBaseRepository(db)
    kb = await repo.get_by_id(kb_id)

    if not kb or kb.organization_id != auth.organization_id or kb.is_deleted:
        raise NotFoundError("KnowledgeBase", kb_id)

    doc = await repo.get_document(doc_id)

    if not doc or doc.knowledge_base_id != kb_id or doc.is_deleted:
        raise NotFoundError("Document", doc_id)

    return success_response(doc_to_response(doc))


@router.delete(
    "/{kb_id}/documents/{doc_id}",
    status_code=204,
    summary="Delete Document",
)
async def delete_document(
    kb_id: str = Path(...),
    doc_id: str = Path(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Delete a document."""
    auth.require_permission(Permission.KNOWLEDGE_DELETE)

    repo = KnowledgeBaseRepository(db)
    kb = await repo.get_by_id(kb_id)

    if not kb or kb.organization_id != auth.organization_id or kb.is_deleted:
        raise NotFoundError("KnowledgeBase", kb_id)

    doc = await repo.get_document(doc_id)

    if not doc or doc.knowledge_base_id != kb_id or doc.is_deleted:
        raise NotFoundError("Document", doc_id)

    await repo.delete_document(doc_id)
    await db.commit()

    logger.info(f"Deleted document {doc_id} from knowledge base {kb_id}")

    # In production, also remove from vector store

    return None


# =============================================================================
# Search Routes
# =============================================================================


@router.post(
    "/{kb_id}/search",
    response_model=APIResponse[List[SearchResult]],
    summary="Search Knowledge Base",
    description="Search for relevant content in a knowledge base.",
)
async def search_knowledge_base(
    kb_id: str = Path(...),
    request: SearchRequest = Body(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Search a knowledge base."""
    auth.require_permission(Permission.KNOWLEDGE_READ)

    repo = KnowledgeBaseRepository(db)
    kb = await repo.get_by_id(kb_id)

    if not kb or kb.organization_id != auth.organization_id or kb.is_deleted:
        raise NotFoundError("KnowledgeBase", kb_id)

    # In production, this would:
    # 1. Embed the query using the configured embedding model
    # 2. Search the vector store
    # 3. Return ranked results

    # For MVP, return empty results
    # When vector store is integrated, this will perform actual semantic search

    return success_response([])


@router.post(
    "/{kb_id}/reindex",
    response_model=APIResponse[Dict[str, Any]],
    summary="Reindex Knowledge Base",
    description="Reindex all documents in a knowledge base.",
)
async def reindex_knowledge_base(
    kb_id: str = Path(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Reindex a knowledge base."""
    auth.require_permission(Permission.KNOWLEDGE_WRITE)

    repo = KnowledgeBaseRepository(db)
    kb = await repo.get_by_id(kb_id)

    if not kb or kb.organization_id != auth.organization_id or kb.is_deleted:
        raise NotFoundError("KnowledgeBase", kb_id)

    # Update status to processing
    await repo.update(kb_id, status="processing")
    await db.commit()

    # In production, this would trigger background job to:
    # 1. Clear existing vectors
    # 2. Re-chunk all documents
    # 3. Re-generate embeddings
    # 4. Re-insert into vector store

    return success_response({
        "knowledge_base_id": kb_id,
        "status": "processing",
        "message": "Reindexing started. This may take a while depending on the number of documents.",
    })


@router.get(
    "/{kb_id}/stats",
    response_model=APIResponse[Dict[str, Any]],
    summary="Get Knowledge Base Stats",
    description="Get statistics for a knowledge base.",
)
async def get_knowledge_base_stats(
    kb_id: str = Path(...),
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Get knowledge base statistics."""
    auth.require_permission(Permission.KNOWLEDGE_READ)

    repo = KnowledgeBaseRepository(db)
    kb = await repo.get_by_id(kb_id)

    if not kb or kb.organization_id != auth.organization_id or kb.is_deleted:
        raise NotFoundError("KnowledgeBase", kb_id)

    return success_response({
        "knowledge_base_id": kb_id,
        "document_count": kb.document_count,
        "chunk_count": kb.chunk_count,
        "total_tokens": kb.total_tokens,
        "embedding_model": kb.embedding_model,
        "vector_store": kb.vector_store,
        "status": kb.status,
        "last_synced_at": kb.last_synced_at.isoformat() if kb.last_synced_at else None,
    })
