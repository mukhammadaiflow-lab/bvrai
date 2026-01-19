"""Knowledge base API routes."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.knowledge.schemas import (
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeBaseResponse,
    KnowledgeBaseListResponse,
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse,
    SearchQuery,
    SearchResponse,
    IngestRequest,
    IngestResponse,
)
from app.knowledge.service import KnowledgeService
from app.auth.dependencies import get_current_user_id

router = APIRouter(prefix="/agents/{agent_id}/knowledge", tags=["knowledge"])


# =====================
# Knowledge Base Routes
# =====================


@router.post("", response_model=KnowledgeBaseResponse, status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    agent_id: UUID,
    data: KnowledgeBaseCreate,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Create a new knowledge base for an agent."""
    service = KnowledgeService(db)
    kb = await service.create_knowledge_base(agent_id, data)
    await db.commit()

    stats = await service.get_kb_stats(kb.id)
    response = KnowledgeBaseResponse.model_validate(kb)
    response.document_count = stats["document_count"]
    response.total_chunks = stats["total_chunks"]

    return response


@router.get("", response_model=KnowledgeBaseListResponse)
async def list_knowledge_bases(
    agent_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """List all knowledge bases for an agent."""
    service = KnowledgeService(db)
    items, total = await service.list_knowledge_bases(agent_id, page, page_size)

    responses = []
    for kb in items:
        stats = await service.get_kb_stats(kb.id)
        response = KnowledgeBaseResponse.model_validate(kb)
        response.document_count = stats["document_count"]
        response.total_chunks = stats["total_chunks"]
        responses.append(response)

    return KnowledgeBaseListResponse(
        items=responses,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    agent_id: UUID,
    kb_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Get a knowledge base by ID."""
    service = KnowledgeService(db)
    kb = await service.get_knowledge_base(kb_id, agent_id)

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found",
        )

    stats = await service.get_kb_stats(kb.id)
    response = KnowledgeBaseResponse.model_validate(kb)
    response.document_count = stats["document_count"]
    response.total_chunks = stats["total_chunks"]

    return response


@router.patch("/{kb_id}", response_model=KnowledgeBaseResponse)
async def update_knowledge_base(
    agent_id: UUID,
    kb_id: UUID,
    data: KnowledgeBaseUpdate,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Update a knowledge base."""
    service = KnowledgeService(db)
    kb = await service.update_knowledge_base(kb_id, data, agent_id)

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found",
        )

    await db.commit()

    stats = await service.get_kb_stats(kb.id)
    response = KnowledgeBaseResponse.model_validate(kb)
    response.document_count = stats["document_count"]
    response.total_chunks = stats["total_chunks"]

    return response


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_base(
    agent_id: UUID,
    kb_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Delete a knowledge base and all its documents."""
    service = KnowledgeService(db)
    deleted = await service.delete_knowledge_base(kb_id, agent_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found",
        )

    await db.commit()


# =====================
# Document Routes
# =====================


@router.post("/{kb_id}/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def add_document(
    agent_id: UUID,
    kb_id: UUID,
    data: DocumentCreate,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Add a document to a knowledge base."""
    service = KnowledgeService(db)

    # Verify KB exists and belongs to agent
    kb = await service.get_knowledge_base(kb_id, agent_id)
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found",
        )

    doc = await service.add_document(kb_id, data)
    await db.commit()

    return DocumentResponse.model_validate(doc)


@router.get("/{kb_id}/documents", response_model=list[DocumentResponse])
async def list_documents(
    agent_id: UUID,
    kb_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """List documents in a knowledge base."""
    service = KnowledgeService(db)

    # Verify KB exists
    kb = await service.get_knowledge_base(kb_id, agent_id)
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found",
        )

    docs, _ = await service.list_documents(kb_id, page, page_size)
    return [DocumentResponse.model_validate(d) for d in docs]


@router.get("/{kb_id}/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(
    agent_id: UUID,
    kb_id: UUID,
    doc_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Get a document by ID."""
    service = KnowledgeService(db)
    doc = await service.get_document(doc_id, kb_id)

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return DocumentResponse.model_validate(doc)


@router.patch("/{kb_id}/documents/{doc_id}", response_model=DocumentResponse)
async def update_document(
    agent_id: UUID,
    kb_id: UUID,
    doc_id: UUID,
    data: DocumentUpdate,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Update a document."""
    service = KnowledgeService(db)
    doc = await service.update_document(doc_id, data, kb_id)

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    await db.commit()
    return DocumentResponse.model_validate(doc)


@router.delete("/{kb_id}/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    agent_id: UUID,
    kb_id: UUID,
    doc_id: UUID,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Delete a document."""
    service = KnowledgeService(db)
    deleted = await service.delete_document(doc_id, kb_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    await db.commit()


# =====================
# Search Routes
# =====================


@router.post("/{kb_id}/search", response_model=SearchResponse)
async def search_knowledge_base(
    agent_id: UUID,
    kb_id: UUID,
    query: SearchQuery,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Search a knowledge base."""
    service = KnowledgeService(db)

    # Verify KB exists
    kb = await service.get_knowledge_base(kb_id, agent_id)
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found",
        )

    return await service.search(kb_id, query)


# =====================
# Ingest Routes
# =====================


@router.post("/{kb_id}/ingest", response_model=IngestResponse)
async def ingest_content(
    agent_id: UUID,
    kb_id: UUID,
    request: IngestRequest,
    db: AsyncSession = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
):
    """Ingest content into a knowledge base."""
    service = KnowledgeService(db)

    # Verify KB exists
    kb = await service.get_knowledge_base(kb_id, agent_id)
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found",
        )

    # Create document
    doc_data = DocumentCreate(
        title=request.title or "Ingested Content",
        content=request.content,
        source=request.source,
        metadata={
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
        },
    )

    doc = await service.add_document(kb_id, doc_data)
    await db.commit()

    return IngestResponse(
        document_id=doc.id,
        chunk_count=doc.chunk_count,
        message=f"Successfully ingested {doc.chunk_count} chunks",
    )
