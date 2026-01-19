"""
Embedding Generation

Generate embeddings for knowledge base:
- Multiple provider support
- Batch processing
- Caching
- Dimensionality reduction
"""

from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import aiohttp
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class EmbeddingProvider(str, Enum):
    """Embedding providers."""
    OPENAI = "openai"
    COHERE = "cohere"
    VOYAGE = "voyage"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    model: str = "text-embedding-3-small"
    dimensions: int = 1536

    # API settings
    api_key: str = ""
    api_base: str = ""

    # Batching
    batch_size: int = 100
    max_concurrent: int = 10

    # Caching
    enable_cache: bool = True
    cache_ttl_hours: int = 24

    # Rate limiting
    rate_limit: int = 1000  # Requests per minute


class EmbeddingCache:
    """Cache for embeddings."""

    def __init__(self, ttl_hours: int = 24):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl_hours = ttl_hours

    def _hash_text(self, text: str, model: str) -> str:
        """Generate cache key."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding."""
        key = self._hash_text(text, model)
        entry = self._cache.get(key)

        if entry:
            # Check TTL
            age = (datetime.utcnow() - entry["timestamp"]).total_seconds() / 3600
            if age < self._ttl_hours:
                return entry["embedding"]
            else:
                del self._cache[key]

        return None

    def set(self, text: str, model: str, embedding: List[float]) -> None:
        """Cache embedding."""
        key = self._hash_text(text, model)
        self._cache[key] = {
            "embedding": embedding,
            "timestamp": datetime.utcnow(),
        }

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()

    def cleanup(self) -> int:
        """Remove expired entries."""
        now = datetime.utcnow()
        expired = []

        for key, entry in self._cache.items():
            age = (now - entry["timestamp"]).total_seconds() / 3600
            if age >= self._ttl_hours:
                expired.append(key)

        for key in expired:
            del self._cache[key]

        return len(expired)


class BaseEmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        pass


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self._base_url = "https://api.openai.com/v1/embeddings"

    async def embed(self, text: str) -> List[float]:
        """Generate single embedding."""
        results = await self.embed_batch([text])
        return results[0] if results else []

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "input": texts,
            "model": self.model,
        }

        if self.dimensions:
            payload["dimensions"] = self.dimensions

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        logger.error(f"OpenAI API error: {error}")
                        return []

                    data = await response.json()
                    embeddings = [item["embedding"] for item in data["data"]]
                    return embeddings

        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            return []


class CohereEmbeddingProvider(BaseEmbeddingProvider):
    """Cohere embedding provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "embed-english-v3.0",
        input_type: str = "search_document",
    ):
        self.api_key = api_key
        self.model = model
        self.input_type = input_type
        self._base_url = "https://api.cohere.ai/v1/embed"

    async def embed(self, text: str) -> List[float]:
        """Generate single embedding."""
        results = await self.embed_batch([text])
        return results[0] if results else []

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "texts": texts,
            "model": self.model,
            "input_type": self.input_type,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        logger.error(f"Cohere API error: {error}")
                        return []

                    data = await response.json()
                    return data.get("embeddings", [])

        except Exception as e:
            logger.error(f"Cohere embedding error: {e}")
            return []


class VoyageEmbeddingProvider(BaseEmbeddingProvider):
    """Voyage AI embedding provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-large-2",
    ):
        self.api_key = api_key
        self.model = model
        self._base_url = "https://api.voyageai.com/v1/embeddings"

    async def embed(self, text: str) -> List[float]:
        """Generate single embedding."""
        results = await self.embed_batch([text])
        return results[0] if results else []

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "input": texts,
            "model": self.model,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        logger.error(f"Voyage API error: {error}")
                        return []

                    data = await response.json()
                    return [item["embedding"] for item in data.get("data", [])]

        except Exception as e:
            logger.error(f"Voyage embedding error: {e}")
            return []


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.

    Requires sentence-transformers package.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.error("sentence-transformers not installed")
                raise

    async def embed(self, text: str) -> List[float]:
        """Generate single embedding."""
        self._load_model()
        embedding = self._model.encode(text)
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings."""
        self._load_model()
        embeddings = self._model.encode(texts)
        return [e.tolist() for e in embeddings]


class EmbeddingService:
    """
    Unified embedding service.

    Features:
    - Multiple provider support
    - Caching
    - Batch processing
    - Rate limiting
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._provider = self._create_provider()
        self._cache = EmbeddingCache(ttl_hours=config.cache_ttl_hours) if config.enable_cache else None
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._rate_limiter: List[float] = []

    def _create_provider(self) -> BaseEmbeddingProvider:
        """Create embedding provider."""
        if self.config.provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbeddingProvider(
                api_key=self.config.api_key,
                model=self.config.model,
                dimensions=self.config.dimensions,
            )
        elif self.config.provider == EmbeddingProvider.COHERE:
            return CohereEmbeddingProvider(
                api_key=self.config.api_key,
                model=self.config.model,
            )
        elif self.config.provider == EmbeddingProvider.VOYAGE:
            return VoyageEmbeddingProvider(
                api_key=self.config.api_key,
                model=self.config.model,
            )
        elif self.config.provider == EmbeddingProvider.LOCAL:
            return LocalEmbeddingProvider(
                model_name=self.config.model,
            )
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        # Check cache
        if self._cache:
            cached = self._cache.get(text, self.config.model)
            if cached:
                return cached

        # Rate limit
        await self._wait_for_rate_limit()

        # Generate
        async with self._semaphore:
            embedding = await self._provider.embed(text)

        # Cache result
        if self._cache and embedding:
            self._cache.set(text, self.config.model, embedding)

        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        results = [None] * len(texts)
        to_embed = []
        to_embed_indices = []

        # Check cache
        for i, text in enumerate(texts):
            if self._cache:
                cached = self._cache.get(text, self.config.model)
                if cached:
                    results[i] = cached
                    continue

            to_embed.append(text)
            to_embed_indices.append(i)

        # Generate missing embeddings
        if to_embed:
            # Process in batches
            batch_size = self.config.batch_size

            for batch_start in range(0, len(to_embed), batch_size):
                batch_end = min(batch_start + batch_size, len(to_embed))
                batch_texts = to_embed[batch_start:batch_end]
                batch_indices = to_embed_indices[batch_start:batch_end]

                await self._wait_for_rate_limit()

                async with self._semaphore:
                    batch_embeddings = await self._provider.embed_batch(batch_texts)

                # Store results
                for j, embedding in enumerate(batch_embeddings):
                    idx = batch_indices[j]
                    results[idx] = embedding

                    # Cache
                    if self._cache and embedding:
                        self._cache.set(batch_texts[j], self.config.model, embedding)

        return results

    async def _wait_for_rate_limit(self) -> None:
        """Wait for rate limit window."""
        import time
        now = time.time()
        window_start = now - 60  # 1 minute window

        # Clean old entries
        self._rate_limiter = [t for t in self._rate_limiter if t > window_start]

        # Check limit
        if len(self._rate_limiter) >= self.config.rate_limit:
            # Wait until oldest entry expires
            wait_time = self._rate_limiter[0] - window_start
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        # Record request
        self._rate_limiter.append(now)

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        if self._cache:
            self._cache.clear()

    def get_sync_embed_func(self) -> Callable[[str], List[float]]:
        """Get synchronous embedding function for knowledge base."""
        def sync_embed(text: str) -> List[float]:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create new loop in thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.embed(text))
                    return future.result()
            else:
                return loop.run_until_complete(self.embed(text))

        return sync_embed


class EmbeddingComparison:
    """
    Utilities for comparing embeddings.

    Features:
    - Similarity metrics
    - Clustering
    - Dimensionality reduction
    """

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        import math
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    @staticmethod
    def euclidean_distance(a: List[float], b: List[float]) -> float:
        """Calculate Euclidean distance."""
        import math
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def dot_product(a: List[float], b: List[float]) -> float:
        """Calculate dot product."""
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def normalize(embedding: List[float]) -> List[float]:
        """Normalize embedding to unit length."""
        import math
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm == 0:
            return embedding
        return [x / norm for x in embedding]

    @staticmethod
    def average_embeddings(embeddings: List[List[float]]) -> List[float]:
        """Calculate average of multiple embeddings."""
        if not embeddings:
            return []

        dim = len(embeddings[0])
        result = [0.0] * dim

        for emb in embeddings:
            for i, val in enumerate(emb):
                result[i] += val

        n = len(embeddings)
        return [x / n for x in result]

    @staticmethod
    def reduce_dimensions(
        embeddings: List[List[float]],
        target_dims: int = 2,
    ) -> List[List[float]]:
        """
        Reduce embedding dimensions using PCA.

        Note: Requires numpy/scipy for full implementation.
        This is a simplified version.
        """
        try:
            import numpy as np
            from sklearn.decomposition import PCA

            arr = np.array(embeddings)
            pca = PCA(n_components=target_dims)
            reduced = pca.fit_transform(arr)
            return reduced.tolist()
        except ImportError:
            logger.warning("numpy/sklearn not available for dimensionality reduction")
            # Return first target_dims dimensions
            return [emb[:target_dims] for emb in embeddings]


# Factory functions
def create_embedding_service(
    provider: EmbeddingProvider = EmbeddingProvider.OPENAI,
    api_key: str = "",
    model: str = "text-embedding-3-small",
    **kwargs,
) -> EmbeddingService:
    """Create embedding service."""
    config = EmbeddingConfig(
        provider=provider,
        api_key=api_key,
        model=model,
        **kwargs,
    )
    return EmbeddingService(config)


def create_openai_embeddings(
    api_key: str,
    model: str = "text-embedding-3-small",
) -> EmbeddingService:
    """Create OpenAI embedding service."""
    return create_embedding_service(
        provider=EmbeddingProvider.OPENAI,
        api_key=api_key,
        model=model,
    )


def create_local_embeddings(
    model_name: str = "all-MiniLM-L6-v2",
) -> EmbeddingService:
    """Create local embedding service."""
    return create_embedding_service(
        provider=EmbeddingProvider.LOCAL,
        model=model_name,
    )
