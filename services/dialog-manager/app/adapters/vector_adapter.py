"""
Vector DB Adapter - Abstract interface and implementations for vector stores.

This module provides:
- Abstract VectorDBAdapter interface
- LocalVectorAdapter using SQLite for local development/testing
- Comments for plugging in Pinecone

TODO: Implement real vector DB adapters:
- PineconeAdapter: Use pinecone-client
"""
import hashlib
import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from app.config import get_settings

logger = structlog.get_logger()


@dataclass
class VectorDocument:
    """Document stored in vector database."""

    id: str
    content: str
    metadata: dict[str, Any]
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    """Result from vector similarity search."""

    id: str
    content: str
    metadata: dict[str, Any]
    score: float


class VectorDBAdapter(ABC):
    """
    Abstract base class for vector database adapters.

    Implementations should handle:
    - Document storage with embeddings
    - Similarity search
    - Metadata filtering
    """

    @abstractmethod
    async def upsert(self, documents: list[VectorDocument]) -> int:
        """
        Insert or update documents with their embeddings.

        Args:
            documents: List of documents to upsert

        Returns:
            Number of documents upserted
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of search results sorted by similarity
        """
        pass

    @abstractmethod
    async def delete(self, ids: list[str]) -> int:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """
        pass

    @abstractmethod
    async def get_by_tenant(self, tenant_id: str) -> list[VectorDocument]:
        """
        Get all documents for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of documents
        """
        pass


class LocalVectorAdapter(VectorDBAdapter):
    """
    Local SQLite-based vector store for development and testing.

    Uses a simple hash-based "embedding" for testing purposes.
    In production, use a real embedding model.

    Features:
    - SQLite storage for persistence
    - Simple cosine similarity search
    - Metadata filtering
    """

    def __init__(self, db_path: str = "./data/vectors.db"):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create database and tables if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tenant
            ON documents((json_extract(metadata, '$.tenant_id')))
        """)

        conn.commit()
        conn.close()

    def _generate_embedding(self, text: str, dim: int = 128) -> list[float]:
        """
        Generate a simple hash-based embedding for testing.

        In production, replace with a real embedding model:
        - OpenAI embeddings: openai.Embedding.create()
        - Sentence transformers: model.encode()
        - Cohere embeddings: co.embed()
        """
        # Simple hash-based embedding for testing
        # This provides consistent embeddings for the same text
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Convert to floats and normalize
        embedding = []
        for i in range(0, min(len(hash_bytes), dim), 1):
            val = hash_bytes[i % len(hash_bytes)] / 255.0
            embedding.append(val)

        # Pad if needed
        while len(embedding) < dim:
            embedding.append(0.0)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [v / norm for v in embedding]

        return embedding

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a_np = np.array(a)
        b_np = np.array(b)

        dot_product = np.dot(a_np, b_np)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    async def upsert(self, documents: list[VectorDocument]) -> int:
        """Insert or update documents."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        count = 0
        for doc in documents:
            # Generate embedding if not provided
            embedding = doc.embedding or self._generate_embedding(doc.content)

            cursor.execute(
                """
                INSERT OR REPLACE INTO documents (id, content, metadata, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (doc.id, doc.content, json.dumps(doc.metadata), json.dumps(embedding)),
            )
            count += 1

        conn.commit()
        conn.close()

        logger.debug("upserted_documents", count=count)
        return count

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents using cosine similarity."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query with optional metadata filter
        query = "SELECT id, content, metadata, embedding FROM documents"
        params: list[Any] = []

        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                params.append(value)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        # Calculate similarities
        results: list[SearchResult] = []
        for row in rows:
            doc_id, content, metadata_str, embedding_str = row
            embedding = json.loads(embedding_str)
            metadata = json.loads(metadata_str)

            score = self._cosine_similarity(query_embedding, embedding)

            results.append(
                SearchResult(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    score=score,
                )
            )

        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def delete(self, ids: list[str]) -> int:
        """Delete documents by ID."""
        if not ids:
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        placeholders = ",".join("?" * len(ids))
        cursor.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", ids)

        count = cursor.rowcount
        conn.commit()
        conn.close()

        logger.debug("deleted_documents", count=count)
        return count

    async def get_by_tenant(self, tenant_id: str) -> list[VectorDocument]:
        """Get all documents for a tenant."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, content, metadata, embedding FROM documents
            WHERE json_extract(metadata, '$.tenant_id') = ?
            """,
            (tenant_id,),
        )

        rows = cursor.fetchall()
        conn.close()

        documents = []
        for row in rows:
            doc_id, content, metadata_str, embedding_str = row
            documents.append(
                VectorDocument(
                    id=doc_id,
                    content=content,
                    metadata=json.loads(metadata_str),
                    embedding=json.loads(embedding_str),
                )
            )

        return documents

    def generate_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for a search query."""
        return self._generate_embedding(query)


# TODO: Implement Pinecone adapter
# class PineconeAdapter(VectorDBAdapter):
#     """
#     Pinecone vector database adapter.
#
#     Usage:
#         from pinecone import Pinecone
#         pc = Pinecone(api_key=settings.pinecone_api_key)
#         index = pc.Index(settings.pinecone_index_name)
#
#         # Upsert
#         index.upsert(vectors=[
#             {"id": doc.id, "values": doc.embedding, "metadata": doc.metadata}
#             for doc in documents
#         ])
#
#         # Search
#         results = index.query(
#             vector=query_embedding,
#             top_k=top_k,
#             filter=filter_metadata,
#             include_metadata=True,
#         )
#     """
#     pass


def create_vector_adapter(provider: str | None = None) -> VectorDBAdapter:
    """
    Factory function to create a vector DB adapter.

    Args:
        provider: Vector DB provider name ("local", "pinecone")
                 If None, uses setting from config.

    Returns:
        Configured VectorDBAdapter instance
    """
    settings = get_settings()
    provider = provider or settings.vector_db_provider

    if provider == "local":
        return LocalVectorAdapter(settings.vector_db_path)

    # TODO: Implement Pinecone adapter
    # if provider == "pinecone":
    #     return PineconeAdapter(
    #         settings.pinecone_api_key,
    #         settings.pinecone_index_name,
    #     )

    logger.warning(f"Unknown vector DB provider: {provider}, using local")
    return LocalVectorAdapter(settings.vector_db_path)
