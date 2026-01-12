"""Knowledge base service."""

from typing import Optional
from uuid import UUID

import httpx
import structlog
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database.models import KnowledgeBase, KnowledgeDocument
from app.knowledge.schemas import (
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    DocumentCreate,
    DocumentUpdate,
    SearchQuery,
    SearchResult,
    SearchResponse,
)

logger = structlog.get_logger()
settings = get_settings()


class KnowledgeService:
    """Service for knowledge base operations."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.logger = logger.bind(service="knowledge")
        self.ai_orchestrator_url = settings.ai_orchestrator_url

    # =====================
    # Knowledge Base CRUD
    # =====================

    async def create_knowledge_base(
        self,
        agent_id: UUID,
        data: KnowledgeBaseCreate,
    ) -> KnowledgeBase:
        """Create a new knowledge base."""
        kb = KnowledgeBase(
            agent_id=agent_id,
            name=data.name,
            description=data.description,
            settings=data.settings or {},
        )

        self.db.add(kb)
        await self.db.flush()
        await self.db.refresh(kb)

        self.logger.info(
            "Knowledge base created",
            kb_id=str(kb.id),
            agent_id=str(agent_id),
        )

        return kb

    async def get_knowledge_base(
        self,
        kb_id: UUID,
        agent_id: Optional[UUID] = None,
    ) -> Optional[KnowledgeBase]:
        """Get a knowledge base by ID."""
        query = select(KnowledgeBase).where(KnowledgeBase.id == kb_id)

        if agent_id:
            query = query.where(KnowledgeBase.agent_id == agent_id)

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def list_knowledge_bases(
        self,
        agent_id: UUID,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[KnowledgeBase], int]:
        """List knowledge bases for an agent."""
        query = select(KnowledgeBase).where(KnowledgeBase.agent_id == agent_id)

        # Count
        count_query = select(func.count()).select_from(query.subquery())
        total = await self.db.execute(count_query)
        total_count = total.scalar() or 0

        # Paginate
        query = query.order_by(KnowledgeBase.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await self.db.execute(query)
        items = result.scalars().all()

        return list(items), total_count

    async def update_knowledge_base(
        self,
        kb_id: UUID,
        data: KnowledgeBaseUpdate,
        agent_id: Optional[UUID] = None,
    ) -> Optional[KnowledgeBase]:
        """Update a knowledge base."""
        kb = await self.get_knowledge_base(kb_id, agent_id)
        if not kb:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(kb, field, value)

        await self.db.flush()
        await self.db.refresh(kb)

        return kb

    async def delete_knowledge_base(
        self,
        kb_id: UUID,
        agent_id: Optional[UUID] = None,
    ) -> bool:
        """Delete a knowledge base and all its documents."""
        kb = await self.get_knowledge_base(kb_id, agent_id)
        if not kb:
            return False

        # Delete all documents first
        await self._delete_documents_from_vector_store(kb_id)

        await self.db.delete(kb)
        await self.db.flush()

        self.logger.info("Knowledge base deleted", kb_id=str(kb_id))
        return True

    # =====================
    # Document CRUD
    # =====================

    async def add_document(
        self,
        kb_id: UUID,
        data: DocumentCreate,
    ) -> Optional[KnowledgeDocument]:
        """Add a document to a knowledge base."""
        kb = await self.get_knowledge_base(kb_id)
        if not kb:
            return None

        doc = KnowledgeDocument(
            knowledge_base_id=kb_id,
            title=data.title,
            content=data.content,
            source=data.source,
            metadata=data.metadata or {},
        )

        self.db.add(doc)
        await self.db.flush()
        await self.db.refresh(doc)

        # Ingest into vector store
        chunk_count = await self._ingest_document(kb_id, doc)
        doc.chunk_count = chunk_count

        await self.db.flush()
        await self.db.refresh(doc)

        self.logger.info(
            "Document added",
            doc_id=str(doc.id),
            kb_id=str(kb_id),
            chunks=chunk_count,
        )

        return doc

    async def get_document(
        self,
        doc_id: UUID,
        kb_id: Optional[UUID] = None,
    ) -> Optional[KnowledgeDocument]:
        """Get a document by ID."""
        query = select(KnowledgeDocument).where(KnowledgeDocument.id == doc_id)

        if kb_id:
            query = query.where(KnowledgeDocument.knowledge_base_id == kb_id)

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def list_documents(
        self,
        kb_id: UUID,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[KnowledgeDocument], int]:
        """List documents in a knowledge base."""
        query = select(KnowledgeDocument).where(
            KnowledgeDocument.knowledge_base_id == kb_id
        )

        # Count
        count_query = select(func.count()).select_from(query.subquery())
        total = await self.db.execute(count_query)
        total_count = total.scalar() or 0

        # Paginate
        query = query.order_by(KnowledgeDocument.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await self.db.execute(query)
        items = result.scalars().all()

        return list(items), total_count

    async def update_document(
        self,
        doc_id: UUID,
        data: DocumentUpdate,
        kb_id: Optional[UUID] = None,
    ) -> Optional[KnowledgeDocument]:
        """Update a document."""
        doc = await self.get_document(doc_id, kb_id)
        if not doc:
            return None

        update_data = data.model_dump(exclude_unset=True)

        # If content changed, re-ingest
        if "content" in update_data:
            await self._delete_document_from_vector_store(doc_id)
            for field, value in update_data.items():
                setattr(doc, field, value)
            chunk_count = await self._ingest_document(doc.knowledge_base_id, doc)
            doc.chunk_count = chunk_count
        else:
            for field, value in update_data.items():
                setattr(doc, field, value)

        await self.db.flush()
        await self.db.refresh(doc)

        return doc

    async def delete_document(
        self,
        doc_id: UUID,
        kb_id: Optional[UUID] = None,
    ) -> bool:
        """Delete a document."""
        doc = await self.get_document(doc_id, kb_id)
        if not doc:
            return False

        # Delete from vector store
        await self._delete_document_from_vector_store(doc_id)

        await self.db.delete(doc)
        await self.db.flush()

        self.logger.info("Document deleted", doc_id=str(doc_id))
        return True

    # =====================
    # Search
    # =====================

    async def search(
        self,
        kb_id: UUID,
        query: SearchQuery,
    ) -> SearchResponse:
        """Search a knowledge base."""
        kb = await self.get_knowledge_base(kb_id)
        if not kb:
            return SearchResponse(
                query=query.query,
                results=[],
                total_results=0,
            )

        # Call AI Orchestrator for RAG search
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.ai_orchestrator_url}/rag/search",
                    json={
                        "query": query.query,
                        "agent_id": str(kb.agent_id),
                        "top_k": query.top_k,
                        "filter_metadata": query.filter_metadata,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

                # Convert to SearchResults
                results = []
                for r in data.get("results", []):
                    results.append(
                        SearchResult(
                            document_id=UUID(r.get("metadata", {}).get("document_id", "00000000-0000-0000-0000-000000000000")),
                            document_title=r.get("metadata", {}).get("title", "Unknown"),
                            chunk_id=r["id"],
                            text=r["text"],
                            score=r["score"],
                            metadata=r.get("metadata", {}),
                        )
                    )

                return SearchResponse(
                    query=query.query,
                    results=results,
                    total_results=len(results),
                )

            except Exception as e:
                self.logger.error("Search failed", error=str(e))
                return SearchResponse(
                    query=query.query,
                    results=[],
                    total_results=0,
                )

    # =====================
    # Vector Store Integration
    # =====================

    async def _ingest_document(
        self,
        kb_id: UUID,
        doc: KnowledgeDocument,
    ) -> int:
        """Ingest document into vector store."""
        kb = await self.get_knowledge_base(kb_id)
        if not kb:
            return 0

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.ai_orchestrator_url}/rag/ingest",
                    json={
                        "text": doc.content,
                        "agent_id": str(kb.agent_id),
                        "source": f"doc:{doc.id}",
                        "metadata": {
                            "document_id": str(doc.id),
                            "title": doc.title,
                            "source": doc.source,
                            **doc.metadata,
                        },
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("chunk_count", 0)

            except Exception as e:
                self.logger.error(
                    "Ingestion failed",
                    doc_id=str(doc.id),
                    error=str(e),
                )
                return 0

    async def _delete_document_from_vector_store(self, doc_id: UUID) -> None:
        """Delete document chunks from vector store."""
        doc = await self.get_document(doc_id)
        if not doc:
            return

        kb = await self.get_knowledge_base(doc.knowledge_base_id)
        if not kb:
            return

        async with httpx.AsyncClient() as client:
            try:
                await client.delete(
                    f"{self.ai_orchestrator_url}/rag/documents/doc:{doc_id}",
                    params={"agent_id": str(kb.agent_id)},
                    timeout=30.0,
                )
            except Exception as e:
                self.logger.error(
                    "Delete from vector store failed",
                    doc_id=str(doc_id),
                    error=str(e),
                )

    async def _delete_documents_from_vector_store(self, kb_id: UUID) -> None:
        """Delete all documents in a knowledge base from vector store."""
        kb = await self.get_knowledge_base(kb_id)
        if not kb:
            return

        # Get all documents
        docs, _ = await self.list_documents(kb_id, page_size=1000)

        for doc in docs:
            await self._delete_document_from_vector_store(doc.id)

    async def get_kb_stats(self, kb_id: UUID) -> dict:
        """Get knowledge base statistics."""
        # Document count
        doc_count_query = select(func.count()).where(
            KnowledgeDocument.knowledge_base_id == kb_id
        )
        doc_result = await self.db.execute(doc_count_query)
        doc_count = doc_result.scalar() or 0

        # Total chunks
        chunk_query = select(func.sum(KnowledgeDocument.chunk_count)).where(
            KnowledgeDocument.knowledge_base_id == kb_id
        )
        chunk_result = await self.db.execute(chunk_query)
        total_chunks = chunk_result.scalar() or 0

        return {
            "document_count": doc_count,
            "total_chunks": total_chunks,
        }
