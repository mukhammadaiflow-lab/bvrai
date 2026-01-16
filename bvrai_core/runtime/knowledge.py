"""
Knowledge Retrieval Module

This module provides knowledge base retrieval capabilities for the agent runtime,
including vector search, semantic matching, and context augmentation.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
)

from .base import (
    KnowledgeChunk,
    KnowledgeQuery,
    KnowledgeResult,
    KnowledgeRetrievalError,
)


logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding dimensions."""
        pass

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    @property
    def name(self) -> str:
        return "openai"

    @property
    def dimensions(self) -> int:
        return 1536  # text-embedding-ada-002

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-ada-002",
    ):
        """
        Initialize provider.

        Args:
            api_key: OpenAI API key
            model: Embedding model to use
        """
        self.api_key = api_key
        self.model = model

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings."""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self.api_key)

            response = await client.embeddings.create(
                model=self.model,
                input=texts,
            )

            return [item.embedding for item in response.data]

        except ImportError:
            raise KnowledgeRetrievalError("openai package not installed")
        except Exception as e:
            raise KnowledgeRetrievalError(f"Embedding failed: {e}")


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    @property
    def name(self) -> str:
        return "mock"

    @property
    def dimensions(self) -> int:
        return 384

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings."""
        import hashlib

        embeddings = []
        for text in texts:
            # Generate deterministic pseudo-embeddings based on text hash
            hash_val = hashlib.md5(text.encode()).hexdigest()
            embedding = [
                int(hash_val[i:i+2], 16) / 255.0 - 0.5
                for i in range(0, min(len(hash_val), self.dimensions * 2), 2)
            ]
            # Pad if needed
            while len(embedding) < self.dimensions:
                embedding.append(0.0)
            embeddings.append(embedding[:self.dimensions])

        return embeddings


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Store name."""
        pass

    @abstractmethod
    async def search(
        self,
        embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors.

        Returns list of (id, score, metadata) tuples.
        """
        pass

    @abstractmethod
    async def add(
        self,
        id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Add a vector to the store."""
        pass

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete a vector from the store."""
        pass


class InMemoryVectorStore(VectorStore):
    """In-memory vector store for development and testing."""

    @property
    def name(self) -> str:
        return "memory"

    def __init__(self):
        """Initialize store."""
        self._vectors: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}

    async def search(
        self,
        embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search using cosine similarity."""
        if not self._vectors:
            return []

        scores = []
        for id, (vec, metadata) in self._vectors.items():
            # Apply filters
            if filters:
                match = all(
                    metadata.get(k) == v
                    for k, v in filters.items()
                )
                if not match:
                    continue

            # Calculate cosine similarity
            score = self._cosine_similarity(embedding, vec)
            scores.append((id, score, metadata))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def add(
        self,
        id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Add a vector."""
        self._vectors[id] = (embedding, metadata)

    async def delete(self, id: str) -> bool:
        """Delete a vector."""
        if id in self._vectors:
            del self._vectors[id]
            return True
        return False


class PineconeVectorStore(VectorStore):
    """Pinecone vector store."""

    @property
    def name(self) -> str:
        return "pinecone"

    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: str = "us-east-1",
        index_name: str = "knowledge",
        namespace: str = "",
    ):
        """
        Initialize store.

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Index name
            namespace: Namespace
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.namespace = namespace
        self._index: Optional[Any] = None

    async def _get_index(self) -> Any:
        """Get Pinecone index."""
        if self._index is None:
            try:
                from pinecone import Pinecone

                pc = Pinecone(api_key=self.api_key)
                self._index = pc.Index(self.index_name)
            except ImportError:
                raise KnowledgeRetrievalError("pinecone package not installed")

        return self._index

    async def search(
        self,
        embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search Pinecone index."""
        index = await self._get_index()

        query_params = {
            "vector": embedding,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": self.namespace,
        }

        if filters:
            query_params["filter"] = filters

        # Run in executor since Pinecone client may be sync
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: index.query(**query_params),
        )

        return [
            (match.id, match.score, match.metadata or {})
            for match in results.matches
        ]

    async def add(
        self,
        id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Add to Pinecone index."""
        index = await self._get_index()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: index.upsert(
                vectors=[(id, embedding, metadata)],
                namespace=self.namespace,
            ),
        )

    async def delete(self, id: str) -> bool:
        """Delete from Pinecone index."""
        index = await self._get_index()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: index.delete(ids=[id], namespace=self.namespace),
        )

        return True


@dataclass
class KnowledgeBaseConfig:
    """Configuration for knowledge base."""

    knowledge_base_id: str
    name: str = ""

    # Retrieval settings
    default_top_k: int = 5
    min_relevance_score: float = 0.5
    max_chunks_per_query: int = 10

    # Context settings
    max_context_tokens: int = 2000
    include_source_info: bool = True

    # Caching
    cache_ttl_seconds: int = 300


class KnowledgeBase:
    """
    Knowledge base with semantic search capabilities.

    Provides:
    - Semantic search over documents
    - Relevance ranking
    - Context preparation for LLM
    """

    def __init__(
        self,
        config: KnowledgeBaseConfig,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        """
        Initialize knowledge base.

        Args:
            config: Knowledge base configuration
            embedding_provider: Embedding provider
            vector_store: Vector store
        """
        self.config = config
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

        # Query cache
        self._cache: Dict[str, Tuple[datetime, KnowledgeResult]] = {}

    async def search(self, query: KnowledgeQuery) -> KnowledgeResult:
        """
        Search the knowledge base.

        Args:
            query: Search query

        Returns:
            Search results
        """
        start_time = time.time()

        # Check cache
        cache_key = f"{query.query}:{query.top_k}:{str(query.filters)}"
        if cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            if (datetime.utcnow() - cached_time).total_seconds() < self.config.cache_ttl_seconds:
                return cached_result

        # Generate query embedding
        embeddings = await self.embedding_provider.embed([query.query])
        query_embedding = embeddings[0]

        # Build filters
        filters = query.filters.copy() if query.filters else {}
        filters["knowledge_base_id"] = self.config.knowledge_base_id

        # Search vector store
        top_k = min(query.top_k, self.config.max_chunks_per_query)
        results = await self.vector_store.search(
            embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

        # Build chunks
        chunks = []
        for id, score, metadata in results:
            if score < query.min_score:
                continue

            chunk = KnowledgeChunk(
                id=id,
                content=metadata.get("content", ""),
                source=metadata.get("source", ""),
                score=score,
                document_id=metadata.get("document_id"),
                document_title=metadata.get("document_title"),
                page_number=metadata.get("page_number"),
                section=metadata.get("section"),
                embedding_model=self.embedding_provider.name,
                chunk_index=metadata.get("chunk_index", 0),
            )
            chunks.append(chunk)

        query_time_ms = (time.time() - start_time) * 1000

        result = KnowledgeResult(
            chunks=chunks,
            query_time_ms=query_time_ms,
            total_results=len(chunks),
        )

        # Cache result
        self._cache[cache_key] = (datetime.utcnow(), result)

        return result

    async def add_document(
        self,
        document_id: str,
        content: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> int:
        """
        Add a document to the knowledge base.

        Args:
            document_id: Document ID
            content: Document content
            source: Document source
            metadata: Additional metadata
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks

        Returns:
            Number of chunks added
        """
        # Chunk the document
        chunks = self._chunk_text(content, chunk_size, chunk_overlap)

        # Generate embeddings
        embeddings = await self.embedding_provider.embed(chunks)

        # Add to vector store
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{document_id}_chunk_{i}"
            chunk_metadata = {
                "knowledge_base_id": self.config.knowledge_base_id,
                "document_id": document_id,
                "source": source,
                "content": chunk,
                "chunk_index": i,
                **(metadata or {}),
            }

            await self.vector_store.add(chunk_id, embedding, chunk_metadata)

        return len(chunks)

    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end
                for sep in [". ", "! ", "? ", "\n\n", "\n"]:
                    pos = text.rfind(sep, start, end)
                    if pos > start:
                        end = pos + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    async def delete_document(self, document_id: str) -> int:
        """
        Delete a document from the knowledge base.

        Args:
            document_id: Document ID

        Returns:
            Number of chunks deleted
        """
        # In production, would query for all chunks with this document_id
        # For now, just return 0
        return 0


class KnowledgeRetriever:
    """
    Knowledge retriever for agent runtime.

    Manages multiple knowledge bases and provides
    unified retrieval interface.
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        """
        Initialize retriever.

        Args:
            embedding_provider: Default embedding provider
            vector_store: Default vector store
        """
        self.embedding_provider = embedding_provider or MockEmbeddingProvider()
        self.vector_store = vector_store or InMemoryVectorStore()
        self._knowledge_bases: Dict[str, KnowledgeBase] = {}

    def add_knowledge_base(
        self,
        config: KnowledgeBaseConfig,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> KnowledgeBase:
        """
        Add a knowledge base.

        Args:
            config: Knowledge base configuration
            embedding_provider: Optional custom embedding provider
            vector_store: Optional custom vector store

        Returns:
            Created knowledge base
        """
        kb = KnowledgeBase(
            config=config,
            embedding_provider=embedding_provider or self.embedding_provider,
            vector_store=vector_store or self.vector_store,
        )
        self._knowledge_bases[config.knowledge_base_id] = kb
        return kb

    def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBase]:
        """Get a knowledge base by ID."""
        return self._knowledge_bases.get(kb_id)

    async def retrieve(
        self,
        query: str,
        knowledge_base_ids: Optional[List[str]] = None,
        top_k: int = 5,
        min_score: float = 0.5,
    ) -> List[KnowledgeChunk]:
        """
        Retrieve relevant knowledge chunks.

        Args:
            query: Search query
            knowledge_base_ids: Knowledge bases to search
            top_k: Results per knowledge base
            min_score: Minimum relevance score

        Returns:
            List of relevant chunks
        """
        kb_ids = knowledge_base_ids or list(self._knowledge_bases.keys())

        all_chunks = []

        for kb_id in kb_ids:
            kb = self._knowledge_bases.get(kb_id)
            if not kb:
                continue

            try:
                result = await kb.search(KnowledgeQuery(
                    query=query,
                    top_k=top_k,
                    min_score=min_score,
                ))
                all_chunks.extend(result.chunks)

            except Exception as e:
                logger.exception(f"Knowledge retrieval from {kb_id} failed: {e}")

        # Sort by score and deduplicate
        all_chunks.sort(key=lambda c: c.score, reverse=True)

        seen_content = set()
        unique_chunks = []
        for chunk in all_chunks:
            content_key = chunk.content[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_chunks.append(chunk)

        return unique_chunks[:top_k * 2]  # Return up to 2x top_k from all sources

    def format_context(
        self,
        chunks: List[KnowledgeChunk],
        max_length: int = 4000,
    ) -> str:
        """
        Format chunks as context for LLM.

        Args:
            chunks: Knowledge chunks
            max_length: Maximum context length

        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0

        for chunk in chunks:
            # Format chunk with source info
            source_info = f"[Source: {chunk.source}"
            if chunk.document_title:
                source_info += f" - {chunk.document_title}"
            if chunk.page_number:
                source_info += f", page {chunk.page_number}"
            source_info += "]"

            chunk_text = f"{source_info}\n{chunk.content}"

            if current_length + len(chunk_text) > max_length:
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text) + 2

        return "\n\n".join(context_parts)


__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "MockEmbeddingProvider",
    "VectorStore",
    "InMemoryVectorStore",
    "PineconeVectorStore",
    "KnowledgeBaseConfig",
    "KnowledgeBase",
    "KnowledgeRetriever",
]
