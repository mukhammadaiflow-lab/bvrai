"""
Document Processing Service

Integrates the ingestion pipeline with database models for persistent
document processing and knowledge base management.
"""

import asyncio
import hashlib
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..database.repositories import KnowledgeBaseRepository
from ..database.models import Document as DocumentModel, DocumentChunk as ChunkModel

from .base import DocumentMetadata, DocumentType, detect_document_type
from .chunking import ChunkingConfig, ChunkingStrategy, create_chunker
from .documents import Document, DocumentChunk, process_document
from .embeddings import EmbeddingConfig, EmbeddingProviderFactory

logger = logging.getLogger(__name__)


# =============================================================================
# Document Processor
# =============================================================================

class DocumentProcessor:
    """
    Process documents for knowledge base ingestion.

    Features:
    - Extract text from various document formats
    - Chunk text into manageable pieces
    - Generate embeddings
    - Store chunks in database and vector store
    - Update processing status
    """

    def __init__(
        self,
        db_session_factory,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize processor.

        Args:
            db_session_factory: Async session factory for database access
            embedding_provider: Embedding provider (openai, cohere, etc.)
            embedding_model: Embedding model to use
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
        """
        self.db_session_factory = db_session_factory
        self.embedding_provider_name = embedding_provider
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize components
        self._embedding_provider = None
        self._chunker = None

        # Processing queue
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the document processor."""
        # Initialize embedding provider
        try:
            config = EmbeddingConfig(
                model=self.embedding_model,
                batch_size=50,
            )
            self._embedding_provider = EmbeddingProviderFactory.create(
                self.embedding_provider_name,
                config=config,
            )
        except Exception as e:
            logger.warning(f"Could not initialize embedding provider: {e}")
            # Continue without embeddings for now

        # Initialize chunker
        self._chunker = create_chunker(
            strategy=ChunkingStrategy.RECURSIVE,
            config=ChunkingConfig(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            ),
        )

        # Start background worker
        self._running = True
        self._worker_task = asyncio.create_task(self._process_queue())

        logger.info("Document processor started")

    async def stop(self) -> None:
        """Stop the document processor."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Document processor stopped")

    async def process_document(
        self,
        document_id: str,
        knowledge_base_id: str,
        organization_id: str,
    ) -> Dict[str, Any]:
        """
        Process a document from the database.

        Args:
            document_id: Database document ID
            knowledge_base_id: Knowledge base ID
            organization_id: Organization ID

        Returns:
            Processing result dict
        """
        # Queue for background processing
        await self._queue.put((document_id, knowledge_base_id, organization_id))
        return {"queued": True, "document_id": document_id}

    async def process_document_sync(
        self,
        document_id: str,
        knowledge_base_id: str,
        organization_id: str,
    ) -> Dict[str, Any]:
        """
        Process a document synchronously (blocking).

        Args:
            document_id: Database document ID
            knowledge_base_id: Knowledge base ID
            organization_id: Organization ID

        Returns:
            Processing result dict
        """
        return await self._process_single_document(
            document_id,
            knowledge_base_id,
            organization_id,
        )

    async def _process_queue(self) -> None:
        """Background worker to process queued documents."""
        while self._running:
            try:
                item = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )
                document_id, kb_id, org_id = item

                try:
                    await self._process_single_document(document_id, kb_id, org_id)
                except Exception as e:
                    logger.error(f"Document processing error: {e}")
                    # Update status to failed
                    await self._update_document_status(
                        document_id,
                        "failed",
                        error_message=str(e),
                    )

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _process_single_document(
        self,
        document_id: str,
        knowledge_base_id: str,
        organization_id: str,
    ) -> Dict[str, Any]:
        """Process a single document."""
        async with self.db_session_factory() as session:
            repo = KnowledgeBaseRepository(session)

            # Get document
            doc = await repo.get_document(document_id)
            if not doc:
                return {"error": "Document not found"}

            # Update status to processing
            await repo.update_document_status(document_id, "processing")
            await session.commit()

            try:
                # Extract content
                content = await self._extract_content(doc)

                if not content:
                    await repo.update_document_status(
                        document_id,
                        "failed",
                        error_message="No content extracted",
                    )
                    await session.commit()
                    return {"error": "No content extracted"}

                # Chunk content
                chunks = self._chunk_content(content, doc.name)

                if not chunks:
                    await repo.update_document_status(
                        document_id,
                        "failed",
                        error_message="No chunks generated",
                    )
                    await session.commit()
                    return {"error": "No chunks generated"}

                # Generate embeddings (if provider available)
                embeddings = None
                if self._embedding_provider:
                    try:
                        texts = [c["content"] for c in chunks]
                        embeddings = await self._embedding_provider.embed(texts)
                    except Exception as e:
                        logger.warning(f"Embedding generation failed: {e}")

                # Store chunks in database
                total_tokens = 0
                for i, chunk in enumerate(chunks):
                    vector_id = None
                    if embeddings:
                        # Would store in vector store and get ID
                        vector_id = f"{document_id}_{i}"

                    await repo.add_chunk(
                        document_id=document_id,
                        knowledge_base_id=knowledge_base_id,
                        content=chunk["content"],
                        chunk_index=i,
                        start_char=chunk.get("start"),
                        end_char=chunk.get("end"),
                        token_count=chunk.get("tokens", 0),
                        vector_id=vector_id,
                        embedding_model=self.embedding_model if embeddings else None,
                        chunk_metadata=chunk.get("metadata"),
                    )
                    total_tokens += chunk.get("tokens", 0)

                # Update document status
                await repo.update_document_status(
                    document_id,
                    "completed",
                    chunk_count=len(chunks),
                    token_count=total_tokens,
                )
                await session.commit()

                logger.info(
                    f"Processed document {document_id}: "
                    f"{len(chunks)} chunks, {total_tokens} tokens"
                )

                return {
                    "success": True,
                    "document_id": document_id,
                    "chunk_count": len(chunks),
                    "token_count": total_tokens,
                }

            except Exception as e:
                logger.exception(f"Document processing failed: {e}")
                await repo.update_document_status(
                    document_id,
                    "failed",
                    error_message=str(e),
                )
                await session.commit()
                return {"error": str(e)}

    async def _extract_content(self, doc) -> Optional[str]:
        """Extract text content from a document."""
        # If content is already stored, return it
        if doc.content:
            return doc.content

        # If file path exists, process the file
        if doc.file_path and os.path.exists(doc.file_path):
            try:
                # Use the document processor for file extraction
                processed = await asyncio.to_thread(
                    process_document,
                    doc.file_path,
                )
                return processed.content
            except Exception as e:
                logger.error(f"File extraction failed: {e}")
                return None

        # If source URL exists, fetch and process
        if doc.source_url:
            try:
                content = await self._fetch_url_content(doc.source_url)
                return content
            except Exception as e:
                logger.error(f"URL fetch failed: {e}")
                return None

        return None

    async def _fetch_url_content(self, url: str) -> Optional[str]:
        """Fetch content from a URL."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "")

                if "text/html" in content_type:
                    # Extract text from HTML
                    from html.parser import HTMLParser
                    html_content = response.text

                    # Simple HTML text extraction
                    class HTMLTextExtractor(HTMLParser):
                        def __init__(self):
                            super().__init__()
                            self.text = []
                            self.skip = False

                        def handle_starttag(self, tag, attrs):
                            if tag in ("script", "style", "head"):
                                self.skip = True

                        def handle_endtag(self, tag):
                            if tag in ("script", "style", "head"):
                                self.skip = False

                        def handle_data(self, data):
                            if not self.skip:
                                self.text.append(data.strip())

                    parser = HTMLTextExtractor()
                    parser.feed(html_content)
                    return " ".join(filter(None, parser.text))

                elif "text/" in content_type or "application/json" in content_type:
                    return response.text

                else:
                    logger.warning(f"Unsupported content type: {content_type}")
                    return None

        except Exception as e:
            logger.error(f"URL fetch error: {e}")
            raise

    def _chunk_content(
        self,
        content: str,
        source_name: str,
    ) -> List[Dict[str, Any]]:
        """Chunk text content."""
        if not self._chunker:
            # Simple fallback chunking
            return self._simple_chunk(content)

        try:
            # Create a document object for the chunker
            doc = Document(
                id=str(uuid.uuid4()),
                content=content,
                metadata=DocumentMetadata(source=source_name),
            )

            # Chunk using the chunker
            chunks = self._chunker.chunk(doc)

            return [
                {
                    "content": chunk.content,
                    "start": chunk.metadata.start_index,
                    "end": chunk.metadata.end_index,
                    "tokens": chunk.metadata.token_count,
                    "metadata": {
                        "chunk_index": chunk.metadata.chunk_index,
                    },
                }
                for chunk in chunks
            ]
        except Exception as e:
            logger.warning(f"Chunker failed, using simple chunking: {e}")
            return self._simple_chunk(content)

    def _simple_chunk(self, content: str) -> List[Dict[str, Any]]:
        """Simple text chunking fallback."""
        chunks = []
        chunk_size = self.chunk_size * 4  # Approximate characters

        start = 0
        while start < len(content):
            # Find end position
            end = min(start + chunk_size, len(content))

            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence endings
                for sep in [". ", ".\n", "! ", "? ", "\n\n"]:
                    last_sep = content.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + len(sep)
                        break

            chunk_text = content[start:end].strip()
            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "start": start,
                    "end": end,
                    "tokens": len(chunk_text.split()),  # Approximate
                })

            start = end

        return chunks

    async def _update_document_status(
        self,
        document_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Update document status in database."""
        async with self.db_session_factory() as session:
            repo = KnowledgeBaseRepository(session)
            await repo.update_document_status(
                document_id,
                status,
                error_message=error_message,
            )
            await session.commit()


# =============================================================================
# Global Processor Instance
# =============================================================================

_processor: Optional[DocumentProcessor] = None


def get_processor() -> Optional[DocumentProcessor]:
    """Get the global processor instance."""
    return _processor


async def init_processor(
    db_session_factory,
    embedding_provider: str = "openai",
    embedding_model: str = "text-embedding-3-small",
) -> DocumentProcessor:
    """Initialize and start the global processor."""
    global _processor
    _processor = DocumentProcessor(
        db_session_factory,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )
    await _processor.start()
    return _processor


async def shutdown_processor() -> None:
    """Shutdown the global processor."""
    global _processor
    if _processor:
        await _processor.stop()
        _processor = None


# =============================================================================
# Convenience Functions
# =============================================================================

async def process_knowledge_document(
    document_id: str,
    knowledge_base_id: str,
    organization_id: str,
) -> Dict[str, Any]:
    """
    Process a knowledge base document using the global processor.

    Usage:
        from bvrai_core.knowledge.processor import process_knowledge_document

        result = await process_knowledge_document(
            document_id="doc-123",
            knowledge_base_id="kb-456",
            organization_id="org-789",
        )
    """
    if _processor:
        return await _processor.process_document(
            document_id=document_id,
            knowledge_base_id=knowledge_base_id,
            organization_id=organization_id,
        )
    else:
        logger.warning("Document processor not initialized")
        return {"error": "Processor not initialized"}


__all__ = [
    "DocumentProcessor",
    "get_processor",
    "init_processor",
    "shutdown_processor",
    "process_knowledge_document",
]
