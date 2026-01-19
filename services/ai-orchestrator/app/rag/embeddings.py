"""Embedding adapters for generating vector representations."""

import hashlib
from abc import ABC, abstractmethod
from typing import Optional

import structlog
from openai import AsyncOpenAI

from app.config import get_settings

logger = structlog.get_logger()


class EmbeddingAdapter(ABC):
    """Abstract base class for embedding adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get adapter name."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    async def close(self) -> None:
        """Clean up resources."""
        pass


class OpenAIEmbeddings(EmbeddingAdapter):
    """
    OpenAI embeddings adapter.

    Uses text-embedding-3-small or text-embedding-3-large models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.model = model or settings.embedding_model
        self._dimensions = settings.embedding_dimensions

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.logger = logger.bind(adapter="openai_embeddings")

    @property
    def name(self) -> str:
        return "openai"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self._dimensions,
            )
            return response.data[0].embedding

        except Exception as e:
            self.logger.error("Embedding failed", error=str(e))
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        try:
            # OpenAI supports batch embedding
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self._dimensions,
            )

            # Sort by index to maintain order
            embeddings = sorted(response.data, key=lambda x: x.index)
            return [e.embedding for e in embeddings]

        except Exception as e:
            self.logger.error("Batch embedding failed", error=str(e))
            raise

    async def close(self) -> None:
        """Close the OpenAI client."""
        await self.client.close()


class MockEmbeddings(EmbeddingAdapter):
    """
    Mock embeddings for testing.

    Generates deterministic hash-based embeddings.
    """

    def __init__(self, dimensions: int = 1536) -> None:
        self._dimensions = dimensions
        self.logger = logger.bind(adapter="mock_embeddings")

    @property
    def name(self) -> str:
        return "mock"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        """Generate a hash-based embedding."""
        return self._hash_embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate hash-based embeddings for multiple texts."""
        return [self._hash_embed(text) for text in texts]

    def _hash_embed(self, text: str) -> list[float]:
        """Generate a deterministic embedding from text hash."""
        # Create a hash of the text
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # Convert hash to embedding
        embedding = []
        for i in range(self._dimensions):
            # Use different parts of hash for each dimension
            idx = i % len(text_hash)
            char = text_hash[idx]
            # Convert to float between -1 and 1
            value = (int(char, 16) / 15.0) * 2 - 1
            embedding.append(value)

        # Normalize the embedding
        magnitude = sum(v * v for v in embedding) ** 0.5
        if magnitude > 0:
            embedding = [v / magnitude for v in embedding]

        return embedding


class CohereEmbeddings(EmbeddingAdapter):
    """
    Cohere embeddings adapter.

    Uses embed-english-v3.0 or embed-multilingual-v3.0.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "embed-english-v3.0",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._dimensions = 1024  # Cohere v3 embeddings

        # Would need cohere SDK
        self.logger = logger.bind(adapter="cohere_embeddings")

    @property
    def name(self) -> str:
        return "cohere"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        """Generate embedding using Cohere."""
        # Placeholder - would use cohere SDK
        raise NotImplementedError("Cohere embeddings not yet implemented")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate batch embeddings using Cohere."""
        raise NotImplementedError("Cohere embeddings not yet implemented")
