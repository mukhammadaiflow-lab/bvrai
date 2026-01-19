"""Vector retriever for similarity search."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
)

from app.config import get_settings
from app.rag.embeddings import EmbeddingAdapter

logger = structlog.get_logger()


@dataclass
class SearchResult:
    """A single search result."""
    id: str
    score: float
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def is_relevant(self) -> bool:
        """Check if result is above relevance threshold."""
        return self.score >= 0.7


@dataclass
class SearchResults:
    """Collection of search results."""
    results: list[SearchResult]
    query: str
    total_found: int = 0

    @property
    def top_result(self) -> Optional[SearchResult]:
        """Get the top result."""
        return self.results[0] if self.results else None

    @property
    def relevant_results(self) -> list[SearchResult]:
        """Get only relevant results."""
        return [r for r in self.results if r.is_relevant]

    def get_context(self, max_results: int = 3, separator: str = "\n\n") -> str:
        """Get combined context from top results."""
        texts = [r.text for r in self.results[:max_results]]
        return separator.join(texts)


class VectorRetriever(ABC):
    """Abstract base class for vector retrievers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get retriever name."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> SearchResults:
        """
        Search for similar documents.

        Args:
            query: Search query text
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            SearchResults with ranked results
        """
        pass

    @abstractmethod
    async def add_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Add documents to the index.

        Args:
            texts: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document

        Returns:
            List of document IDs
        """
        pass

    @abstractmethod
    async def delete_documents(self, ids: list[str]) -> int:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """
        pass

    async def close(self) -> None:
        """Clean up resources."""
        pass


class QdrantRetriever(VectorRetriever):
    """
    Qdrant vector retriever.

    Production-ready vector search with filtering.
    """

    def __init__(
        self,
        embedding_adapter: EmbeddingAdapter,
        collection_name: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        settings = get_settings()

        self.embeddings = embedding_adapter
        self.collection_name = collection_name or settings.qdrant_collection
        self.url = url or settings.qdrant_url

        self.client = AsyncQdrantClient(url=self.url)
        self.logger = logger.bind(
            retriever="qdrant",
            collection=self.collection_name,
        )

        self._initialized = False

    @property
    def name(self) -> str:
        return "qdrant"

    async def _ensure_collection(self) -> None:
        """Ensure the collection exists."""
        if self._initialized:
            return

        try:
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embeddings.dimensions,
                        distance=Distance.COSINE,
                    ),
                )
                self.logger.info("Created collection", collection=self.collection_name)

            self._initialized = True

        except Exception as e:
            self.logger.error("Failed to ensure collection", error=str(e))
            raise

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> SearchResults:
        """Search for similar documents."""
        await self._ensure_collection()

        try:
            # Generate query embedding
            query_vector = await self.embeddings.embed(query)

            # Build filter
            query_filter = None
            if filter_metadata:
                conditions = []
                for key, value in filter_metadata.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value),
                        )
                    )
                query_filter = Filter(must=conditions)

            # Search
            results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
                search_params=SearchParams(
                    hnsw_ef=128,
                    exact=False,
                ),
            )

            # Convert to SearchResults
            search_results = []
            for result in results:
                payload = result.payload or {}
                search_results.append(
                    SearchResult(
                        id=str(result.id),
                        score=result.score,
                        text=payload.get("text", ""),
                        metadata={k: v for k, v in payload.items() if k != "text"},
                    )
                )

            self.logger.debug(
                "Search completed",
                query=query[:50],
                results=len(search_results),
            )

            return SearchResults(
                results=search_results,
                query=query,
                total_found=len(search_results),
            )

        except Exception as e:
            self.logger.error("Search failed", error=str(e))
            raise

    async def add_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Add documents to the index."""
        await self._ensure_collection()

        if not texts:
            return []

        try:
            # Generate embeddings
            embeddings = await self.embeddings.embed_batch(texts)

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in texts]

            # Prepare points
            points = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                payload = {"text": text}
                if metadatas and i < len(metadatas):
                    payload.update(metadatas[i])

                points.append(
                    PointStruct(
                        id=ids[i],
                        vector=embedding,
                        payload=payload,
                    )
                )

            # Upsert to Qdrant
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            self.logger.info("Added documents", count=len(texts))
            return ids

        except Exception as e:
            self.logger.error("Failed to add documents", error=str(e))
            raise

    async def delete_documents(self, ids: list[str]) -> int:
        """Delete documents by ID."""
        await self._ensure_collection()

        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids,
            )

            self.logger.info("Deleted documents", count=len(ids))
            return len(ids)

        except Exception as e:
            self.logger.error("Failed to delete documents", error=str(e))
            raise

    async def close(self) -> None:
        """Close the Qdrant client."""
        await self.client.close()


class MockRetriever(VectorRetriever):
    """
    Mock retriever for testing.

    Stores documents in memory with hash-based similarity.
    """

    def __init__(self, embedding_adapter: EmbeddingAdapter) -> None:
        self.embeddings = embedding_adapter
        self.documents: dict[str, dict] = {}
        self.logger = logger.bind(retriever="mock")

    @property
    def name(self) -> str:
        return "mock"

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> SearchResults:
        """Search in-memory documents."""
        if not self.documents:
            return SearchResults(results=[], query=query, total_found=0)

        # Get query embedding
        query_embedding = await self.embeddings.embed(query)

        # Calculate similarity scores
        scored_docs = []
        for doc_id, doc in self.documents.items():
            # Apply metadata filter
            if filter_metadata:
                match = all(
                    doc.get("metadata", {}).get(k) == v
                    for k, v in filter_metadata.items()
                )
                if not match:
                    continue

            # Calculate cosine similarity
            doc_embedding = doc["embedding"]
            score = self._cosine_similarity(query_embedding, doc_embedding)

            scored_docs.append((doc_id, score, doc))

        # Sort by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Convert to results
        results = []
        for doc_id, score, doc in scored_docs[:top_k]:
            results.append(
                SearchResult(
                    id=doc_id,
                    score=score,
                    text=doc["text"],
                    metadata=doc.get("metadata", {}),
                )
            )

        return SearchResults(
            results=results,
            query=query,
            total_found=len(results),
        )

    async def add_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Add documents to in-memory store."""
        if not texts:
            return []

        # Generate embeddings
        embeddings = await self.embeddings.embed_batch(texts)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Store documents
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            doc = {
                "text": text,
                "embedding": embedding,
                "metadata": metadatas[i] if metadatas and i < len(metadatas) else {},
            }
            self.documents[ids[i]] = doc

        self.logger.info("Added documents", count=len(texts))
        return ids

    async def delete_documents(self, ids: list[str]) -> int:
        """Delete documents from in-memory store."""
        deleted = 0
        for doc_id in ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
                deleted += 1

        return deleted

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)
