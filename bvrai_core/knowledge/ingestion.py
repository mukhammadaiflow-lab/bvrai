"""
Document Ingestion Pipeline

This module provides a complete pipeline for ingesting documents
into the knowledge base with processing, chunking, embedding, and storage.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from .base import (
    DocumentMetadata,
    DocumentType,
    detect_document_type,
)
from .documents import (
    Document,
    DocumentChunk,
    process_document,
)
from .chunking import (
    ChunkingConfig,
    ChunkingStrategy,
    create_chunker,
    TextChunker,
)
from .embeddings import (
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingProviderFactory,
)
from .vectorstore import (
    VectorStore,
    VectorStoreConfig,
    VectorStoreFactory,
)


logger = logging.getLogger(__name__)


class IngestionStatus(str, Enum):
    """Status of ingestion."""
    PENDING = "pending"
    PROCESSING = "processing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IngestionResult:
    """Result of document ingestion."""

    # Document info
    document_id: str
    source: str
    document_type: DocumentType

    # Processing results
    status: IngestionStatus = IngestionStatus.PENDING
    error: Optional[str] = None

    # Statistics
    total_chunks: int = 0
    total_tokens: int = 0
    processing_time_ms: float = 0.0
    chunking_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    storage_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Chunk IDs for reference
    chunk_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "source": self.source,
            "document_type": self.document_type.value,
            "status": self.status.value,
            "error": self.error,
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "processing_time_ms": self.processing_time_ms,
            "total_time_ms": self.total_time_ms,
        }


@dataclass
class IngestionConfig:
    """Configuration for the ingestion pipeline."""

    # Processing options
    extract_metadata: bool = True
    clean_text: bool = True
    detect_language: bool = False

    # Chunking configuration
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    # Embedding configuration
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 100

    # Vector store configuration
    vector_store_type: str = "memory"
    collection_name: str = "documents"

    # Organization
    organization_id: Optional[str] = None
    collection_id: Optional[str] = None
    default_tags: List[str] = field(default_factory=list)

    # Processing options
    parallel_processing: bool = True
    max_concurrent_documents: int = 5
    max_concurrent_embeddings: int = 3

    # Callbacks
    on_progress: Optional[Callable[[str, IngestionStatus, float], None]] = None

    # Deduplication
    deduplicate_documents: bool = True
    deduplicate_chunks: bool = False


class IngestionPipeline:
    """
    Complete document ingestion pipeline.

    Handles document processing, chunking, embedding generation,
    and storage in vector store.
    """

    def __init__(
        self,
        config: Optional[IngestionConfig] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None,
        chunker: Optional[TextChunker] = None,
    ):
        self.config = config or IngestionConfig()

        # Initialize embedding provider
        if embedding_provider:
            self._embedding_provider = embedding_provider
        else:
            embedding_config = EmbeddingConfig(
                model=self.config.embedding_model,
                batch_size=self.config.embedding_batch_size,
            )
            self._embedding_provider = EmbeddingProviderFactory.create(
                self.config.embedding_provider,
                config=embedding_config,
            )

        # Initialize vector store
        if vector_store:
            self._vector_store = vector_store
        else:
            store_config = VectorStoreConfig(
                collection_name=self.config.collection_name,
                dimensions=self._embedding_provider.get_dimensions(),
            )
            self._vector_store = VectorStoreFactory.create(
                self.config.vector_store_type,
                config=store_config,
            )

        # Initialize chunker
        if chunker:
            self._chunker = chunker
        else:
            self._chunker = create_chunker(
                strategy=self.config.chunking.strategy,
                config=self.config.chunking,
            )

        # Track processed documents for deduplication
        self._processed_hashes: set = set()

        # Semaphores for concurrency control
        self._doc_semaphore = asyncio.Semaphore(self.config.max_concurrent_documents)
        self._embed_semaphore = asyncio.Semaphore(self.config.max_concurrent_embeddings)

    @property
    def vector_store(self) -> VectorStore:
        """Get the vector store."""
        return self._vector_store

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Get the embedding provider."""
        return self._embedding_provider

    async def ingest(
        self,
        source: Union[str, Path, List[Union[str, Path]]],
        metadata: Optional[DocumentMetadata] = None,
    ) -> Union[IngestionResult, List[IngestionResult]]:
        """
        Ingest one or more documents.

        Args:
            source: File path, URL, or list of sources
            metadata: Optional metadata to apply to all documents

        Returns:
            IngestionResult or list of results
        """
        # Handle single source
        if isinstance(source, (str, Path)):
            return await self._ingest_single(source, metadata)

        # Handle multiple sources
        if self.config.parallel_processing:
            tasks = [
                self._ingest_single(s, metadata)
                for s in source
            ]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for s in source:
                result = await self._ingest_single(s, metadata)
                results.append(result)
            return results

    async def _ingest_single(
        self,
        source: Union[str, Path],
        metadata: Optional[DocumentMetadata] = None,
    ) -> IngestionResult:
        """Ingest a single document."""
        start_time = time.time()
        source_str = str(source)

        # Initialize result
        result = IngestionResult(
            document_id="",
            source=source_str,
            document_type=detect_document_type(source_str),
        )

        async with self._doc_semaphore:
            try:
                # Step 1: Process document
                self._emit_progress(source_str, IngestionStatus.PROCESSING, 0.1)
                process_start = time.time()

                document = await self._process_document(source, metadata)
                result.document_id = document.id
                result.processing_time_ms = (time.time() - process_start) * 1000

                # Check for duplicates
                if self.config.deduplicate_documents:
                    if document.content_hash in self._processed_hashes:
                        logger.info(f"Skipping duplicate document: {source_str}")
                        result.status = IngestionStatus.COMPLETED
                        result.total_time_ms = (time.time() - start_time) * 1000
                        return result
                    self._processed_hashes.add(document.content_hash)

                # Step 2: Chunk document
                self._emit_progress(source_str, IngestionStatus.CHUNKING, 0.3)
                chunk_start = time.time()

                chunks = await self._chunk_document(document)
                result.total_chunks = len(chunks)
                result.chunking_time_ms = (time.time() - chunk_start) * 1000

                if not chunks:
                    logger.warning(f"No chunks generated for: {source_str}")
                    result.status = IngestionStatus.COMPLETED
                    result.total_time_ms = (time.time() - start_time) * 1000
                    return result

                # Step 3: Generate embeddings
                self._emit_progress(source_str, IngestionStatus.EMBEDDING, 0.5)
                embed_start = time.time()

                embeddings = await self._generate_embeddings(chunks)
                result.embedding_time_ms = (time.time() - embed_start) * 1000
                result.total_tokens = sum(c.metadata.token_count for c in chunks)

                # Step 4: Store in vector store
                self._emit_progress(source_str, IngestionStatus.STORING, 0.8)
                store_start = time.time()

                chunk_ids = await self._store_chunks(chunks, embeddings)
                result.chunk_ids = chunk_ids
                result.storage_time_ms = (time.time() - store_start) * 1000

                # Complete
                result.status = IngestionStatus.COMPLETED
                result.total_time_ms = (time.time() - start_time) * 1000

                self._emit_progress(source_str, IngestionStatus.COMPLETED, 1.0)

                logger.info(
                    f"Ingested {source_str}: {result.total_chunks} chunks, "
                    f"{result.total_time_ms:.0f}ms"
                )

            except Exception as e:
                logger.error(f"Ingestion failed for {source_str}: {e}")
                result.status = IngestionStatus.FAILED
                result.error = str(e)
                result.total_time_ms = (time.time() - start_time) * 1000

        return result

    async def _process_document(
        self,
        source: Union[str, Path],
        metadata: Optional[DocumentMetadata] = None,
    ) -> Document:
        """Process a document."""
        # Create metadata with organization info
        if metadata is None:
            metadata = DocumentMetadata(
                source=str(source),
                source_type=detect_document_type(str(source)),
            )

        if self.config.organization_id:
            metadata.organization_id = self.config.organization_id
        if self.config.collection_id:
            metadata.collection_id = self.config.collection_id
        if self.config.default_tags:
            metadata.tags.extend(self.config.default_tags)

        # Process document in thread pool
        document = await asyncio.to_thread(
            process_document,
            source,
            metadata=metadata,
        )

        return document

    async def _chunk_document(
        self,
        document: Document,
    ) -> List[DocumentChunk]:
        """Chunk a document."""
        # Run chunking in thread pool (can be CPU intensive)
        chunks = await asyncio.to_thread(
            self._chunker.chunk,
            document,
        )

        return chunks

    async def _generate_embeddings(
        self,
        chunks: List[DocumentChunk],
    ) -> List[List[float]]:
        """Generate embeddings for chunks."""
        async with self._embed_semaphore:
            contents = [chunk.content for chunk in chunks]
            embeddings = await self._embedding_provider.embed(contents)
            return embeddings

    async def _store_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> List[str]:
        """Store chunks in vector store."""
        chunk_ids = await self._vector_store.add(chunks, embeddings)
        return chunk_ids

    def _emit_progress(
        self,
        source: str,
        status: IngestionStatus,
        progress: float,
    ) -> None:
        """Emit progress callback."""
        if self.config.on_progress:
            try:
                self.config.on_progress(source, status, progress)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    async def ingest_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        file_types: Optional[List[DocumentType]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[IngestionResult]:
        """
        Ingest all documents from a directory.

        Args:
            directory: Directory path
            recursive: Whether to process subdirectories
            file_types: Optional list of file types to process
            exclude_patterns: Optional patterns to exclude

        Returns:
            List of ingestion results
        """
        directory = Path(directory)

        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        # Find files
        files = []
        pattern = "**/*" if recursive else "*"

        for path in directory.glob(pattern):
            if not path.is_file():
                continue

            # Check file type
            file_type = detect_document_type(path.name)
            if file_types and file_type not in file_types:
                continue

            # Check exclusions
            if exclude_patterns:
                excluded = False
                for pattern in exclude_patterns:
                    if path.match(pattern):
                        excluded = True
                        break
                if excluded:
                    continue

            files.append(path)

        logger.info(f"Found {len(files)} files to ingest in {directory}")

        # Ingest all files
        return await self.ingest(files)

    async def ingest_text(
        self,
        text: str,
        source_name: str = "text_input",
        metadata: Optional[DocumentMetadata] = None,
    ) -> IngestionResult:
        """
        Ingest raw text content.

        Args:
            text: Text content to ingest
            source_name: Name/identifier for the source
            metadata: Optional metadata

        Returns:
            Ingestion result
        """
        from io import StringIO

        # Create a file-like object from text
        text_io = StringIO(text)
        text_io.name = source_name

        return await self._ingest_single(source_name, metadata)

    async def delete_document(
        self,
        document_id: str,
    ) -> int:
        """
        Delete a document and its chunks from the store.

        Args:
            document_id: ID of document to delete

        Returns:
            Number of chunks deleted
        """
        from .base import SearchFilter

        filters = SearchFilter(document_ids=[document_id])
        deleted = await self._vector_store.delete(filters=filters)

        logger.info(f"Deleted document {document_id}: {deleted} chunks")
        return deleted

    async def update_document(
        self,
        source: Union[str, Path],
        document_id: Optional[str] = None,
    ) -> IngestionResult:
        """
        Update (re-ingest) a document.

        Args:
            source: Document source
            document_id: Optional existing document ID to replace

        Returns:
            Ingestion result
        """
        # Delete existing if specified
        if document_id:
            await self.delete_document(document_id)

        # Re-ingest
        return await self.ingest(source)

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "processed_documents": len(self._processed_hashes),
            "embedding_provider": self.config.embedding_provider,
            "embedding_model": self.config.embedding_model,
            "chunking_strategy": self.config.chunking.strategy.value,
            "chunk_size": self.config.chunking.chunk_size,
            "vector_store_type": self.config.vector_store_type,
        }


# Convenience function for quick ingestion
async def ingest_documents(
    sources: Union[str, Path, List[Union[str, Path]]],
    embedding_provider: str = "openai",
    vector_store_type: str = "memory",
    collection_name: str = "documents",
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    **kwargs,
) -> Union[IngestionResult, List[IngestionResult]]:
    """
    Quick document ingestion with default settings.

    Args:
        sources: Document source(s) to ingest
        embedding_provider: Embedding provider name
        vector_store_type: Vector store type
        collection_name: Collection name
        chunking_strategy: Chunking strategy
        **kwargs: Additional config options

    Returns:
        Ingestion result(s)
    """
    config = IngestionConfig(
        embedding_provider=embedding_provider,
        vector_store_type=vector_store_type,
        collection_name=collection_name,
        chunking=ChunkingConfig(strategy=chunking_strategy),
        **kwargs,
    )

    pipeline = IngestionPipeline(config=config)
    return await pipeline.ingest(sources)


__all__ = [
    "IngestionStatus",
    "IngestionResult",
    "IngestionConfig",
    "IngestionPipeline",
    "ingest_documents",
]
