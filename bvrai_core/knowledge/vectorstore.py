"""
Vector Store Module

This module provides integrations with various vector databases
for storing and retrieving embeddings.
"""

import asyncio
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from .base import (
    ChunkMetadata,
    RetrievalResult,
    SearchFilter,
)
from .documents import DocumentChunk
from .embeddings import cosine_similarity


logger = logging.getLogger(__name__)


class DistanceMetric(str, Enum):
    """Distance metrics for similarity search."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


@dataclass
class VectorStoreConfig:
    """Configuration for vector stores."""

    # Collection/index settings
    collection_name: str = "documents"
    dimensions: int = 1536
    distance_metric: DistanceMetric = DistanceMetric.COSINE

    # Connection settings
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None

    # Index settings
    index_type: str = "hnsw"  # hnsw, flat, ivf
    ef_construction: int = 128
    m: int = 16  # HNSW M parameter

    # Batch settings
    batch_size: int = 100

    # Persistence
    persist_directory: Optional[str] = None


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or VectorStoreConfig()

    @abstractmethod
    async def add(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> List[str]:
        """Add chunks with embeddings to the store."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[SearchFilter] = None,
    ) -> List[RetrievalResult]:
        """Search for similar chunks."""
        pass

    @abstractmethod
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[SearchFilter] = None,
    ) -> int:
        """Delete chunks by ID or filter."""
        pass

    @abstractmethod
    async def update(
        self,
        chunk_id: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a chunk's embedding or metadata."""
        pass

    async def get(self, ids: List[str]) -> List[Optional[DocumentChunk]]:
        """Get chunks by ID."""
        raise NotImplementedError

    async def count(self, filters: Optional[SearchFilter] = None) -> int:
        """Count chunks matching filters."""
        raise NotImplementedError

    async def clear(self) -> None:
        """Clear all data from the store."""
        raise NotImplementedError


class InMemoryStore(VectorStore):
    """In-memory vector store for testing and small datasets."""

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        super().__init__(config)
        self._chunks: Dict[str, DocumentChunk] = {}
        self._embeddings: Dict[str, List[float]] = {}

    async def add(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> List[str]:
        """Add chunks to memory."""
        ids = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.id or str(uuid.uuid4())
            self._chunks[chunk_id] = chunk
            self._embeddings[chunk_id] = embedding
            chunk.embedding = embedding
            ids.append(chunk_id)
        return ids

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[SearchFilter] = None,
    ) -> List[RetrievalResult]:
        """Search in memory using cosine similarity."""
        scores = []

        for chunk_id, embedding in self._embeddings.items():
            chunk = self._chunks[chunk_id]

            # Apply filters
            if filters:
                filter_dict = filters.to_dict()
                if not self._matches_filter(chunk, filter_dict):
                    continue

            score = cosine_similarity(query_embedding, embedding)
            scores.append((chunk_id, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        results = []
        for chunk_id, score in scores[:top_k]:
            chunk = self._chunks[chunk_id]
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                content=chunk.content,
                metadata=chunk.metadata,
                score=score,
                semantic_score=score,
                source=chunk.metadata.document_source,
            ))

        return results

    def _matches_filter(
        self,
        chunk: DocumentChunk,
        filters: Dict[str, Any],
    ) -> bool:
        """Check if chunk matches filters."""
        metadata = chunk.metadata

        for key, condition in filters.items():
            value = getattr(metadata, key, metadata.custom.get(key))

            if isinstance(condition, dict):
                if "$in" in condition:
                    if value not in condition["$in"]:
                        return False
                if "$gte" in condition:
                    if value is None or value < condition["$gte"]:
                        return False
                if "$lte" in condition:
                    if value is None or value > condition["$lte"]:
                        return False
            else:
                if value != condition:
                    return False

        return True

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[SearchFilter] = None,
    ) -> int:
        """Delete chunks from memory."""
        deleted = 0

        if ids:
            for chunk_id in ids:
                if chunk_id in self._chunks:
                    del self._chunks[chunk_id]
                    del self._embeddings[chunk_id]
                    deleted += 1

        if filters:
            filter_dict = filters.to_dict()
            to_delete = []

            for chunk_id, chunk in self._chunks.items():
                if self._matches_filter(chunk, filter_dict):
                    to_delete.append(chunk_id)

            for chunk_id in to_delete:
                del self._chunks[chunk_id]
                del self._embeddings[chunk_id]
                deleted += 1

        return deleted

    async def update(
        self,
        chunk_id: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a chunk."""
        if chunk_id not in self._chunks:
            return False

        if embedding:
            self._embeddings[chunk_id] = embedding
            self._chunks[chunk_id].embedding = embedding

        if metadata:
            for key, value in metadata.items():
                if hasattr(self._chunks[chunk_id].metadata, key):
                    setattr(self._chunks[chunk_id].metadata, key, value)
                else:
                    self._chunks[chunk_id].metadata.custom[key] = value

        return True

    async def get(self, ids: List[str]) -> List[Optional[DocumentChunk]]:
        """Get chunks by ID."""
        return [self._chunks.get(chunk_id) for chunk_id in ids]

    async def count(self, filters: Optional[SearchFilter] = None) -> int:
        """Count chunks."""
        if not filters:
            return len(self._chunks)

        filter_dict = filters.to_dict()
        count = 0
        for chunk in self._chunks.values():
            if self._matches_filter(chunk, filter_dict):
                count += 1
        return count

    async def clear(self) -> None:
        """Clear all data."""
        self._chunks.clear()
        self._embeddings.clear()


class PineconeStore(VectorStore):
    """Pinecone vector store."""

    def __init__(
        self,
        config: Optional[VectorStoreConfig] = None,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
    ):
        super().__init__(config)
        self._api_key = api_key or config.api_key if config else None
        self._api_key = self._api_key or os.environ.get("PINECONE_API_KEY")
        self._environment = environment or os.environ.get("PINECONE_ENVIRONMENT")
        self._index = None

    async def _get_index(self):
        """Get or create Pinecone index."""
        if self._index is None:
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=self._api_key)
                self._index = pc.Index(self.config.collection_name)
            except ImportError:
                raise ImportError("pinecone-client package required. Install with: pip install pinecone-client")
        return self._index

    async def add(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> List[str]:
        """Add chunks to Pinecone."""
        index = await self._get_index()

        vectors = []
        ids = []

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.id or str(uuid.uuid4())
            ids.append(chunk_id)

            metadata = {
                "content": chunk.content[:1000],  # Pinecone metadata limit
                "document_id": chunk.metadata.document_id,
                "document_source": chunk.metadata.document_source,
                "chunk_index": chunk.metadata.chunk_index,
            }
            metadata.update(chunk.metadata.custom)

            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": metadata,
            })

        # Batch upsert
        for i in range(0, len(vectors), self.config.batch_size):
            batch = vectors[i:i + self.config.batch_size]
            await asyncio.to_thread(index.upsert, vectors=batch)

        return ids

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[SearchFilter] = None,
    ) -> List[RetrievalResult]:
        """Search Pinecone."""
        index = await self._get_index()

        query_params = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True,
        }

        if filters:
            query_params["filter"] = filters.to_dict()

        response = await asyncio.to_thread(index.query, **query_params)

        results = []
        for match in response.matches:
            metadata = match.metadata or {}

            chunk_metadata = ChunkMetadata(
                document_id=metadata.get("document_id", ""),
                document_source=metadata.get("document_source", ""),
                chunk_index=metadata.get("chunk_index", 0),
            )

            results.append(RetrievalResult(
                chunk_id=match.id,
                content=metadata.get("content", ""),
                metadata=chunk_metadata,
                score=match.score,
                semantic_score=match.score,
                source=metadata.get("document_source", ""),
            ))

        return results

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[SearchFilter] = None,
    ) -> int:
        """Delete from Pinecone."""
        index = await self._get_index()

        if ids:
            await asyncio.to_thread(index.delete, ids=ids)
            return len(ids)

        if filters:
            await asyncio.to_thread(index.delete, filter=filters.to_dict())
            return -1  # Pinecone doesn't return count

        return 0

    async def update(
        self,
        chunk_id: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update in Pinecone."""
        index = await self._get_index()

        update_params = {"id": chunk_id}
        if embedding:
            update_params["values"] = embedding
        if metadata:
            update_params["set_metadata"] = metadata

        await asyncio.to_thread(index.update, **update_params)
        return True


class QdrantStore(VectorStore):
    """Qdrant vector store."""

    def __init__(
        self,
        config: Optional[VectorStoreConfig] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(config)
        self._url = url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        self._api_key = api_key or config.api_key if config else None
        self._api_key = self._api_key or os.environ.get("QDRANT_API_KEY")
        self._client = None

    async def _get_client(self):
        """Get or create Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import VectorParams, Distance

                self._client = QdrantClient(
                    url=self._url,
                    api_key=self._api_key,
                )

                # Create collection if not exists
                collections = self._client.get_collections().collections
                if self.config.collection_name not in [c.name for c in collections]:
                    distance = {
                        DistanceMetric.COSINE: Distance.COSINE,
                        DistanceMetric.EUCLIDEAN: Distance.EUCLID,
                        DistanceMetric.DOT_PRODUCT: Distance.DOT,
                    }[self.config.distance_metric]

                    self._client.create_collection(
                        collection_name=self.config.collection_name,
                        vectors_config=VectorParams(
                            size=self.config.dimensions,
                            distance=distance,
                        ),
                    )

            except ImportError:
                raise ImportError("qdrant-client package required. Install with: pip install qdrant-client")
        return self._client

    async def add(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> List[str]:
        """Add chunks to Qdrant."""
        client = await self._get_client()

        from qdrant_client.models import PointStruct

        points = []
        ids = []

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.id or str(uuid.uuid4())
            ids.append(chunk_id)

            # Use hash for numeric ID
            numeric_id = hash(chunk_id) % (10 ** 18)

            payload = {
                "chunk_id": chunk_id,
                "content": chunk.content,
                "document_id": chunk.metadata.document_id,
                "document_source": chunk.metadata.document_source,
                "chunk_index": chunk.metadata.chunk_index,
            }
            payload.update(chunk.metadata.custom)

            points.append(PointStruct(
                id=numeric_id,
                vector=embedding,
                payload=payload,
            ))

        # Batch upsert
        for i in range(0, len(points), self.config.batch_size):
            batch = points[i:i + self.config.batch_size]
            await asyncio.to_thread(
                client.upsert,
                collection_name=self.config.collection_name,
                points=batch,
            )

        return ids

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[SearchFilter] = None,
    ) -> List[RetrievalResult]:
        """Search Qdrant."""
        client = await self._get_client()

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_filter = None
        if filters:
            conditions = []
            filter_dict = filters.to_dict()

            for key, value in filter_dict.items():
                if isinstance(value, dict):
                    # Handle complex conditions
                    if "$in" in value:
                        for v in value["$in"]:
                            conditions.append(FieldCondition(
                                key=key,
                                match=MatchValue(value=v),
                            ))
                else:
                    conditions.append(FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    ))

            if conditions:
                query_filter = Filter(should=conditions)

        response = await asyncio.to_thread(
            client.search,
            collection_name=self.config.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
        )

        results = []
        for point in response:
            payload = point.payload or {}

            chunk_metadata = ChunkMetadata(
                document_id=payload.get("document_id", ""),
                document_source=payload.get("document_source", ""),
                chunk_index=payload.get("chunk_index", 0),
            )

            results.append(RetrievalResult(
                chunk_id=payload.get("chunk_id", str(point.id)),
                content=payload.get("content", ""),
                metadata=chunk_metadata,
                score=point.score,
                semantic_score=point.score,
                source=payload.get("document_source", ""),
            ))

        return results

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[SearchFilter] = None,
    ) -> int:
        """Delete from Qdrant."""
        client = await self._get_client()

        from qdrant_client.models import PointIdsList, FilterSelector, Filter, FieldCondition, MatchValue

        if ids:
            numeric_ids = [hash(i) % (10 ** 18) for i in ids]
            await asyncio.to_thread(
                client.delete,
                collection_name=self.config.collection_name,
                points_selector=PointIdsList(points=numeric_ids),
            )
            return len(ids)

        if filters:
            filter_dict = filters.to_dict()
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_dict.items()
                if not isinstance(v, dict)
            ]

            if conditions:
                await asyncio.to_thread(
                    client.delete,
                    collection_name=self.config.collection_name,
                    points_selector=FilterSelector(filter=Filter(must=conditions)),
                )

        return -1

    async def update(
        self,
        chunk_id: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update in Qdrant."""
        client = await self._get_client()

        numeric_id = hash(chunk_id) % (10 ** 18)

        if metadata:
            await asyncio.to_thread(
                client.set_payload,
                collection_name=self.config.collection_name,
                payload=metadata,
                points=[numeric_id],
            )

        if embedding:
            from qdrant_client.models import PointStruct
            await asyncio.to_thread(
                client.upsert,
                collection_name=self.config.collection_name,
                points=[PointStruct(id=numeric_id, vector=embedding)],
            )

        return True

    async def count(self, filters: Optional[SearchFilter] = None) -> int:
        """Count points in collection."""
        client = await self._get_client()
        info = await asyncio.to_thread(
            client.get_collection,
            collection_name=self.config.collection_name,
        )
        return info.points_count


class ChromaStore(VectorStore):
    """ChromaDB vector store."""

    def __init__(
        self,
        config: Optional[VectorStoreConfig] = None,
        persist_directory: Optional[str] = None,
    ):
        super().__init__(config)
        self._persist_dir = persist_directory or config.persist_directory if config else None
        self._client = None
        self._collection = None

    async def _get_collection(self):
        """Get or create Chroma collection."""
        if self._collection is None:
            try:
                import chromadb
                from chromadb.config import Settings

                if self._persist_dir:
                    self._client = chromadb.Client(Settings(
                        chroma_db_impl="duckdb+parquet",
                        persist_directory=self._persist_dir,
                    ))
                else:
                    self._client = chromadb.Client()

                self._collection = self._client.get_or_create_collection(
                    name=self.config.collection_name,
                    metadata={"hnsw:space": self.config.distance_metric.value},
                )

            except ImportError:
                raise ImportError("chromadb package required. Install with: pip install chromadb")
        return self._collection

    async def add(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> List[str]:
        """Add chunks to Chroma."""
        collection = await self._get_collection()

        ids = []
        documents = []
        metadatas = []

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.id or str(uuid.uuid4())
            ids.append(chunk_id)
            documents.append(chunk.content)

            metadata = {
                "document_id": chunk.metadata.document_id,
                "document_source": chunk.metadata.document_source,
                "chunk_index": chunk.metadata.chunk_index,
            }
            # Chroma requires simple types
            for k, v in chunk.metadata.custom.items():
                if isinstance(v, (str, int, float, bool)):
                    metadata[k] = v
            metadatas.append(metadata)

        await asyncio.to_thread(
            collection.add,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return ids

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[SearchFilter] = None,
    ) -> List[RetrievalResult]:
        """Search Chroma."""
        collection = await self._get_collection()

        where = None
        if filters:
            where = {}
            filter_dict = filters.to_dict()
            for k, v in filter_dict.items():
                if not isinstance(v, dict):
                    where[k] = v

        response = await asyncio.to_thread(
            collection.query,
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where if where else None,
        )

        results = []
        if response["ids"] and response["ids"][0]:
            for i, chunk_id in enumerate(response["ids"][0]):
                metadata_dict = response["metadatas"][0][i] if response["metadatas"] else {}
                distance = response["distances"][0][i] if response["distances"] else 0

                chunk_metadata = ChunkMetadata(
                    document_id=metadata_dict.get("document_id", ""),
                    document_source=metadata_dict.get("document_source", ""),
                    chunk_index=metadata_dict.get("chunk_index", 0),
                )

                # Convert distance to similarity score
                score = 1 - distance if distance else 1

                results.append(RetrievalResult(
                    chunk_id=chunk_id,
                    content=response["documents"][0][i] if response["documents"] else "",
                    metadata=chunk_metadata,
                    score=score,
                    semantic_score=score,
                    source=metadata_dict.get("document_source", ""),
                ))

        return results

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[SearchFilter] = None,
    ) -> int:
        """Delete from Chroma."""
        collection = await self._get_collection()

        if ids:
            await asyncio.to_thread(collection.delete, ids=ids)
            return len(ids)

        if filters:
            where = filters.to_dict()
            await asyncio.to_thread(collection.delete, where=where)

        return -1

    async def update(
        self,
        chunk_id: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update in Chroma."""
        collection = await self._get_collection()

        update_kwargs = {"ids": [chunk_id]}
        if embedding:
            update_kwargs["embeddings"] = [embedding]
        if metadata:
            update_kwargs["metadatas"] = [metadata]

        await asyncio.to_thread(collection.update, **update_kwargs)
        return True

    async def count(self, filters: Optional[SearchFilter] = None) -> int:
        """Count items in collection."""
        collection = await self._get_collection()
        return collection.count()

    async def clear(self) -> None:
        """Clear collection."""
        if self._client:
            self._client.delete_collection(self.config.collection_name)
            self._collection = None


class VectorStoreFactory:
    """Factory for creating vector stores."""

    _stores: Dict[str, Type[VectorStore]] = {
        "memory": InMemoryStore,
        "pinecone": PineconeStore,
        "qdrant": QdrantStore,
        "chroma": ChromaStore,
    }

    @classmethod
    def register(cls, name: str, store_class: Type[VectorStore]) -> None:
        """Register a new store class."""
        cls._stores[name.lower()] = store_class

    @classmethod
    def create(
        cls,
        store_type: str,
        config: Optional[VectorStoreConfig] = None,
        **kwargs,
    ) -> VectorStore:
        """Create a store instance."""
        store_lower = store_type.lower()

        if store_lower not in cls._stores:
            raise ValueError(
                f"Unknown store: {store_type}. "
                f"Available: {list(cls._stores.keys())}"
            )

        store_class = cls._stores[store_lower]
        return store_class(config=config, **kwargs)

    @classmethod
    def list_stores(cls) -> List[str]:
        """List available stores."""
        return list(cls._stores.keys())


__all__ = [
    "DistanceMetric",
    "VectorStoreConfig",
    "VectorStore",
    "InMemoryStore",
    "PineconeStore",
    "QdrantStore",
    "ChromaStore",
    "VectorStoreFactory",
]
