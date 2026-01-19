"""
Embedding Providers Module

This module provides multi-provider support for text embeddings
used in semantic search and retrieval.
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
)


logger = logging.getLogger(__name__)


class EmbeddingModel(str, Enum):
    """Known embedding models."""

    # OpenAI
    OPENAI_ADA_002 = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"

    # Cohere
    COHERE_EMBED_V3 = "embed-english-v3.0"
    COHERE_EMBED_MULTI_V3 = "embed-multilingual-v3.0"

    # Voyage
    VOYAGE_2 = "voyage-2"
    VOYAGE_LARGE_2 = "voyage-large-2"
    VOYAGE_CODE_2 = "voyage-code-2"

    # Google
    GOOGLE_GECKO = "textembedding-gecko"

    # HuggingFace
    SENTENCE_TRANSFORMERS = "sentence-transformers/all-MiniLM-L6-v2"
    BGE_LARGE = "BAAI/bge-large-en-v1.5"
    BGE_BASE = "BAAI/bge-base-en-v1.5"


@dataclass
class EmbeddingModelInfo:
    """Information about an embedding model."""

    model_id: str
    dimensions: int
    max_input_tokens: int
    price_per_1k: float = 0.0

    # Capabilities
    supports_batching: bool = True
    max_batch_size: int = 100


# Model registry
EMBEDDING_MODELS = {
    EmbeddingModel.OPENAI_ADA_002: EmbeddingModelInfo(
        model_id="text-embedding-ada-002",
        dimensions=1536,
        max_input_tokens=8191,
        price_per_1k=0.0001,
    ),
    EmbeddingModel.OPENAI_3_SMALL: EmbeddingModelInfo(
        model_id="text-embedding-3-small",
        dimensions=1536,
        max_input_tokens=8191,
        price_per_1k=0.00002,
    ),
    EmbeddingModel.OPENAI_3_LARGE: EmbeddingModelInfo(
        model_id="text-embedding-3-large",
        dimensions=3072,
        max_input_tokens=8191,
        price_per_1k=0.00013,
    ),
    EmbeddingModel.COHERE_EMBED_V3: EmbeddingModelInfo(
        model_id="embed-english-v3.0",
        dimensions=1024,
        max_input_tokens=512,
        price_per_1k=0.0001,
    ),
    EmbeddingModel.VOYAGE_2: EmbeddingModelInfo(
        model_id="voyage-2",
        dimensions=1024,
        max_input_tokens=4000,
        price_per_1k=0.0001,
    ),
    EmbeddingModel.VOYAGE_LARGE_2: EmbeddingModelInfo(
        model_id="voyage-large-2",
        dimensions=1536,
        max_input_tokens=16000,
        price_per_1k=0.00012,
    ),
}


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers."""

    # Model selection
    model: str = "text-embedding-3-small"

    # Dimensions (for models that support it)
    dimensions: Optional[int] = None

    # Batching
    batch_size: int = 100

    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600

    # Rate limiting
    max_requests_per_minute: int = 3000

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._cache: Dict[str, List[float]] = {}

    @abstractmethod
    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        pass

    def embed_sync(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Synchronous wrapper for embed."""
        return asyncio.run(self.embed(texts))

    def _batch_texts(self, texts: List[str]) -> List[List[str]]:
        """Split texts into batches."""
        batches = []
        for i in range(0, len(texts), self.config.batch_size):
            batches.append(texts[i:i + self.config.batch_size])
        return batches

    def _get_cached(self, text: str) -> Optional[List[float]]:
        """Get cached embedding."""
        if self.config.enable_cache:
            return self._cache.get(text)
        return None

    def _set_cached(self, text: str, embedding: List[float]) -> None:
        """Cache embedding."""
        if self.config.enable_cache:
            self._cache[text] = embedding

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._cache.clear()


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(config)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    async def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self._client

    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        if isinstance(texts, str):
            texts = [texts]

        # Check cache
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            cached = self._get_cached(text)
            if cached:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        # Embed uncached texts
        if texts_to_embed:
            client = await self._get_client()

            new_embeddings = []
            for batch in self._batch_texts(texts_to_embed):
                params = {
                    "input": batch,
                    "model": self.config.model,
                }

                if self.config.dimensions:
                    params["dimensions"] = self.config.dimensions

                response = await client.embeddings.create(**params)

                for item in response.data:
                    new_embeddings.append(item.embedding)

            # Update results and cache
            for i, idx in enumerate(indices_to_embed):
                embeddings[idx] = new_embeddings[i]
                self._set_cached(texts_to_embed[i], new_embeddings[i])

        return embeddings

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        if self.config.dimensions:
            return self.config.dimensions

        model_info = EMBEDDING_MODELS.get(self.config.model)
        if model_info:
            return model_info.dimensions

        # Default for OpenAI
        return 1536


class CohereEmbeddings(EmbeddingProvider):
    """Cohere embedding provider."""

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(config)
        self._api_key = api_key or os.environ.get("COHERE_API_KEY")
        self._client = None

    async def _get_client(self):
        """Get or create Cohere client."""
        if self._client is None:
            try:
                import cohere
                self._client = cohere.AsyncClient(api_key=self._api_key)
            except ImportError:
                raise ImportError("cohere package required. Install with: pip install cohere")
        return self._client

    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings using Cohere."""
        if isinstance(texts, str):
            texts = [texts]

        client = await self._get_client()

        all_embeddings = []
        for batch in self._batch_texts(texts):
            response = await client.embed(
                texts=batch,
                model=self.config.model,
                input_type="search_document",
            )
            all_embeddings.extend(response.embeddings)

        return all_embeddings

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return 1024  # Cohere v3 default


class VoyageEmbeddings(EmbeddingProvider):
    """Voyage AI embedding provider."""

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(config)
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        self._client = None

    async def _get_client(self):
        """Get or create Voyage client."""
        if self._client is None:
            try:
                import voyageai
                self._client = voyageai.AsyncClient(api_key=self._api_key)
            except ImportError:
                raise ImportError("voyageai package required. Install with: pip install voyageai")
        return self._client

    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings using Voyage."""
        if isinstance(texts, str):
            texts = [texts]

        client = await self._get_client()

        all_embeddings = []
        for batch in self._batch_texts(texts):
            response = await client.embed(
                texts=batch,
                model=self.config.model,
            )
            all_embeddings.extend(response.embeddings)

        return all_embeddings

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        model_info = EMBEDDING_MODELS.get(self.config.model)
        return model_info.dimensions if model_info else 1024


class HuggingFaceEmbeddings(EmbeddingProvider):
    """HuggingFace Sentence Transformers embedding provider."""

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        model_name: Optional[str] = None,
    ):
        super().__init__(config)
        self._model_name = model_name or config.model if config else "sentence-transformers/all-MiniLM-L6-v2"
        self._model = None

    def _get_model(self):
        """Get or load model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers package required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings using HuggingFace."""
        if isinstance(texts, str):
            texts = [texts]

        model = self._get_model()

        # Run in thread pool for async compatibility
        embeddings = await asyncio.to_thread(
            model.encode,
            texts,
            convert_to_numpy=True,
        )

        return [emb.tolist() for emb in embeddings]

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        model = self._get_model()
        return model.get_sentence_embedding_dimension()


class GoogleEmbeddings(EmbeddingProvider):
    """Google Vertex AI embedding provider."""

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        project_id: Optional[str] = None,
        location: str = "us-central1",
    ):
        super().__init__(config)
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location
        self._model = None

    async def _get_model(self):
        """Get or create Vertex AI model."""
        if self._model is None:
            try:
                from vertexai.language_models import TextEmbeddingModel
                self._model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
            except ImportError:
                raise ImportError(
                    "google-cloud-aiplatform package required. "
                    "Install with: pip install google-cloud-aiplatform"
                )
        return self._model

    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings using Google Vertex AI."""
        if isinstance(texts, str):
            texts = [texts]

        model = await self._get_model()

        all_embeddings = []
        for batch in self._batch_texts(texts):
            embeddings = await asyncio.to_thread(
                model.get_embeddings,
                batch,
            )
            all_embeddings.extend([e.values for e in embeddings])

        return all_embeddings

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return 768  # Gecko default


class EmbeddingProviderFactory:
    """Factory for creating embedding providers."""

    _providers: Dict[str, Type[EmbeddingProvider]] = {
        "openai": OpenAIEmbeddings,
        "cohere": CohereEmbeddings,
        "voyage": VoyageEmbeddings,
        "huggingface": HuggingFaceEmbeddings,
        "google": GoogleEmbeddings,
    }

    @classmethod
    def register(cls, name: str, provider_class: Type[EmbeddingProvider]) -> None:
        """Register a new provider class."""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def create(
        cls,
        provider: str,
        config: Optional[EmbeddingConfig] = None,
        **kwargs,
    ) -> EmbeddingProvider:
        """Create a provider instance."""
        provider_lower = provider.lower()

        if provider_lower not in cls._providers:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available: {list(cls._providers.keys())}"
            )

        provider_class = cls._providers[provider_lower]
        return provider_class(config=config, **kwargs)

    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers."""
        return list(cls._providers.keys())


# Utility functions

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """Calculate Euclidean distance between two vectors."""
    return sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5


def dot_product(vec1: List[float], vec2: List[float]) -> float:
    """Calculate dot product between two vectors."""
    return sum(a * b for a, b in zip(vec1, vec2))


__all__ = [
    "EmbeddingModel",
    "EmbeddingModelInfo",
    "EMBEDDING_MODELS",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "OpenAIEmbeddings",
    "CohereEmbeddings",
    "VoyageEmbeddings",
    "HuggingFaceEmbeddings",
    "GoogleEmbeddings",
    "EmbeddingProviderFactory",
    "cosine_similarity",
    "euclidean_distance",
    "dot_product",
]
