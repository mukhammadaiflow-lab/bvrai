"""
Embedding Generation System

Text embeddings for semantic understanding:
- Multiple embedding models
- Similarity computation
- Vector indexing
- Semantic search
- Clustering support
"""

from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import math
import hashlib
import heapq
import logging

logger = logging.getLogger(__name__)


class EmbeddingModelType(str, Enum):
    """Embedding model types."""
    HASH = "hash"
    TFIDF = "tfidf"
    WORD2VEC = "word2vec"
    FASTTEXT = "fasttext"
    BERT = "bert"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    OPENAI = "openai"
    CUSTOM = "custom"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_type: EmbeddingModelType = EmbeddingModelType.HASH
    model_name: str = "default"
    dimensions: int = 384
    normalize: bool = True
    pooling_strategy: str = "mean"  # mean, max, cls
    max_sequence_length: int = 512
    batch_size: int = 32
    cache_embeddings: bool = True
    device: str = "cpu"


@dataclass
class TextEmbedding:
    """Text embedding result."""
    text: str
    embedding: List[float]
    model: str
    dimensions: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.embedding) != self.dimensions:
            raise ValueError(f"Embedding dimension mismatch: {len(self.embedding)} != {self.dimensions}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "embedding": self.embedding,
            "model": self.model,
            "dimensions": self.dimensions,
            "metadata": self.metadata,
        }

    def cosine_similarity(self, other: "TextEmbedding") -> float:
        """Compute cosine similarity with another embedding."""
        return cosine_similarity(self.embedding, other.embedding)

    def euclidean_distance(self, other: "TextEmbedding") -> float:
        """Compute Euclidean distance to another embedding."""
        return euclidean_distance(self.embedding, other.embedding)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Compute Euclidean distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def normalize_vector(vector: List[float]) -> List[float]:
    """L2 normalize a vector."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


class EmbeddingModel(ABC):
    """Abstract base for embedding models."""

    @abstractmethod
    async def embed(self, text: str) -> TextEmbedding:
        """Generate embedding for text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[TextEmbedding]:
        """Generate embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        pass


class HashEmbeddingModel(EmbeddingModel):
    """
    Hash-based embedding model.

    Fast, deterministic embeddings using feature hashing.
    Good for development and testing.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig(model_type=EmbeddingModelType.HASH)
        self._cache: Dict[str, TextEmbedding] = {}

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    async def embed(self, text: str) -> TextEmbedding:
        """Generate hash-based embedding."""
        # Check cache
        cache_key = self._cache_key(text)
        if self.config.cache_embeddings and cache_key in self._cache:
            return self._cache[cache_key]

        # Generate embedding using feature hashing
        embedding = self._generate_hash_embedding(text)

        if self.config.normalize:
            embedding = normalize_vector(embedding)

        result = TextEmbedding(
            text=text,
            embedding=embedding,
            model=f"hash-{self.config.dimensions}",
            dimensions=self.config.dimensions,
        )

        if self.config.cache_embeddings:
            self._cache[cache_key] = result

        return result

    def _generate_hash_embedding(self, text: str) -> List[float]:
        """Generate embedding using feature hashing."""
        embedding = [0.0] * self.config.dimensions

        # Tokenize
        words = text.lower().split()

        # Hash each word to dimension
        for word in words:
            # Use multiple hash functions for better distribution
            for i in range(3):
                h = hashlib.sha256(f"{word}_{i}".encode())
                idx = int(h.hexdigest()[:8], 16) % self.config.dimensions
                sign = 1 if int(h.hexdigest()[8:10], 16) % 2 == 0 else -1
                embedding[idx] += sign * 1.0

        # Also include character n-grams
        text_lower = text.lower()
        for n in [2, 3]:
            for i in range(len(text_lower) - n + 1):
                ngram = text_lower[i:i + n]
                h = hashlib.sha256(f"ngram_{ngram}".encode())
                idx = int(h.hexdigest()[:8], 16) % self.config.dimensions
                sign = 1 if int(h.hexdigest()[8:10], 16) % 2 == 0 else -1
                embedding[idx] += sign * 0.5

        return embedding

    def _cache_key(self, text: str) -> str:
        """Generate cache key."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    async def embed_batch(self, texts: List[str]) -> List[TextEmbedding]:
        """Generate embeddings for multiple texts."""
        return await asyncio.gather(*[self.embed(text) for text in texts])


class TFIDFEmbeddingModel(EmbeddingModel):
    """
    TF-IDF based embedding model.

    Uses term frequency-inverse document frequency weighting.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig(model_type=EmbeddingModelType.TFIDF)
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._doc_count = 0
        self._is_fitted = False

    @property
    def dimensions(self) -> int:
        return len(self._vocabulary) if self._is_fitted else self.config.dimensions

    async def fit(self, documents: List[str]) -> None:
        """Fit the model on documents."""
        # Build vocabulary
        word_doc_count: Dict[str, int] = {}

        for doc in documents:
            words = set(doc.lower().split())
            for word in words:
                word_doc_count[word] = word_doc_count.get(word, 0) + 1

        self._doc_count = len(documents)

        # Filter vocabulary (keep top N by document frequency)
        sorted_words = sorted(
            word_doc_count.items(),
            key=lambda x: -x[1]
        )[:self.config.dimensions]

        self._vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_words)}

        # Compute IDF
        for word, count in word_doc_count.items():
            if word in self._vocabulary:
                self._idf[word] = math.log(self._doc_count / (count + 1)) + 1

        self._is_fitted = True
        logger.info(f"TF-IDF model fitted with {len(self._vocabulary)} terms")

    async def embed(self, text: str) -> TextEmbedding:
        """Generate TF-IDF embedding."""
        if not self._is_fitted:
            # Fit on single document
            await self.fit([text])

        embedding = [0.0] * len(self._vocabulary)

        # Compute TF
        words = text.lower().split()
        word_counts: Dict[str, int] = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Compute TF-IDF
        for word, count in word_counts.items():
            if word in self._vocabulary:
                idx = self._vocabulary[word]
                tf = count / len(words) if words else 0
                idf = self._idf.get(word, 1.0)
                embedding[idx] = tf * idf

        if self.config.normalize:
            embedding = normalize_vector(embedding)

        return TextEmbedding(
            text=text,
            embedding=embedding,
            model="tfidf",
            dimensions=len(embedding),
        )

    async def embed_batch(self, texts: List[str]) -> List[TextEmbedding]:
        """Generate embeddings for multiple texts."""
        if not self._is_fitted:
            await self.fit(texts)
        return await asyncio.gather(*[self.embed(text) for text in texts])


class Word2VecEmbeddingModel(EmbeddingModel):
    """
    Word2Vec-style embedding model.

    Simplified implementation using learned word vectors.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig(
            model_type=EmbeddingModelType.WORD2VEC,
            dimensions=100
        )
        self._word_vectors: Dict[str, List[float]] = {}
        self._is_loaded = False

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    async def load_vectors(self, path: Optional[str] = None) -> None:
        """Load pre-trained word vectors."""
        # In production, this would load actual word vectors
        # Using random initialization for demo
        logger.info("Initializing Word2Vec vectors (demo mode)")
        self._is_loaded = True

    def _get_word_vector(self, word: str) -> List[float]:
        """Get or create word vector."""
        word_lower = word.lower()
        if word_lower not in self._word_vectors:
            # Generate deterministic random vector
            h = hashlib.sha256(word_lower.encode())
            self._word_vectors[word_lower] = [
                (int(h.hexdigest()[i * 2:i * 2 + 2], 16) / 255.0 - 0.5)
                for i in range(self.config.dimensions)
            ]
        return self._word_vectors[word_lower]

    async def embed(self, text: str) -> TextEmbedding:
        """Generate Word2Vec embedding."""
        words = text.lower().split()

        if not words:
            embedding = [0.0] * self.config.dimensions
        else:
            # Average word vectors
            word_embeddings = [self._get_word_vector(w) for w in words]
            embedding = [
                sum(we[i] for we in word_embeddings) / len(word_embeddings)
                for i in range(self.config.dimensions)
            ]

        if self.config.normalize:
            embedding = normalize_vector(embedding)

        return TextEmbedding(
            text=text,
            embedding=embedding,
            model="word2vec",
            dimensions=self.config.dimensions,
        )

    async def embed_batch(self, texts: List[str]) -> List[TextEmbedding]:
        """Generate embeddings for multiple texts."""
        return await asyncio.gather(*[self.embed(text) for text in texts])


class SentenceEmbeddingModel(EmbeddingModel):
    """
    Sentence embedding model.

    Wrapper for sentence transformer models.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
            model_name="all-MiniLM-L6-v2",
            dimensions=384
        )
        self._model = None
        self._fallback = HashEmbeddingModel(EmbeddingConfig(dimensions=self.config.dimensions))

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    async def load_model(self) -> None:
        """Load sentence transformer model."""
        # In production, this would load the actual model
        logger.info(f"Loading sentence transformer: {self.config.model_name}")
        # Placeholder - actual loading would happen here

    async def embed(self, text: str) -> TextEmbedding:
        """Generate sentence embedding."""
        if self._model is None:
            # Use fallback
            result = await self._fallback.embed(text)
            result.model = f"sentence-transformer-{self.config.model_name}-fallback"
            return result

        # In production, this would use the actual model
        # Placeholder implementation
        return await self._fallback.embed(text)

    async def embed_batch(self, texts: List[str]) -> List[TextEmbedding]:
        """Generate embeddings for multiple texts."""
        return await asyncio.gather(*[self.embed(text) for text in texts])


class EmbeddingGenerator:
    """
    Unified embedding generator.

    Provides consistent interface for multiple embedding models.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        model: Optional[EmbeddingModel] = None,
    ):
        self.config = config or EmbeddingConfig()
        self._model = model or self._create_model()
        self._cache: Dict[str, TextEmbedding] = {}

    def _create_model(self) -> EmbeddingModel:
        """Create model based on config."""
        model_map = {
            EmbeddingModelType.HASH: HashEmbeddingModel,
            EmbeddingModelType.TFIDF: TFIDFEmbeddingModel,
            EmbeddingModelType.WORD2VEC: Word2VecEmbeddingModel,
            EmbeddingModelType.SENTENCE_TRANSFORMER: SentenceEmbeddingModel,
        }
        cls = model_map.get(self.config.model_type, HashEmbeddingModel)
        return cls(self.config)

    async def embed(self, text: str) -> TextEmbedding:
        """Generate embedding for text."""
        if self.config.cache_embeddings:
            cache_key = hashlib.sha256(text.encode()).hexdigest()[:32]
            if cache_key in self._cache:
                return self._cache[cache_key]

        result = await self._model.embed(text)

        if self.config.cache_embeddings:
            self._cache[cache_key] = result

        return result

    async def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> List[TextEmbedding]:
        """Generate embeddings for multiple texts."""
        results = []

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_results = await self._model.embed_batch(batch)
            results.extend(batch_results)

            if show_progress:
                logger.info(f"Embedded {min(i + self.config.batch_size, len(texts))}/{len(texts)}")

        return results

    def similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between texts (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.async_similarity(text1, text2)
        )

    async def async_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between texts."""
        emb1, emb2 = await asyncio.gather(
            self.embed(text1),
            self.embed(text2)
        )
        return emb1.cosine_similarity(emb2)

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._model.dimensions

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._cache.clear()


@dataclass
class SearchResult:
    """Search result with score."""
    text: str
    embedding: TextEmbedding
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimilaritySearch:
    """
    Similarity search over embeddings.

    Enables semantic search using vector similarity.
    """

    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        metric: str = "cosine",
    ):
        self.generator = embedding_generator or EmbeddingGenerator()
        self.metric = metric
        self._index: List[TextEmbedding] = []
        self._metadata: List[Dict[str, Any]] = []

    async def add(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add text to index."""
        embedding = await self.generator.embed(text)
        self._index.append(embedding)
        self._metadata.append(metadata or {})
        return len(self._index) - 1

    async def add_batch(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Add multiple texts to index."""
        embeddings = await self.generator.embed_batch(texts)
        metadata_list = metadata_list or [{} for _ in texts]

        indices = []
        for emb, meta in zip(embeddings, metadata_list):
            self._index.append(emb)
            self._metadata.append(meta)
            indices.append(len(self._index) - 1)

        return indices

    async def search(
        self,
        query: str,
        k: int = 10,
        threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search for similar texts."""
        if not self._index:
            return []

        query_embedding = await self.generator.embed(query)

        # Compute similarities
        scores = []
        for i, doc_embedding in enumerate(self._index):
            if self.metric == "cosine":
                score = query_embedding.cosine_similarity(doc_embedding)
            else:  # euclidean
                score = -query_embedding.euclidean_distance(doc_embedding)
            scores.append((score, i))

        # Filter by threshold
        if threshold is not None:
            scores = [(s, i) for s, i in scores if s >= threshold]

        # Get top-k
        top_k = heapq.nlargest(k, scores, key=lambda x: x[0])

        # Build results
        results = []
        for rank, (score, idx) in enumerate(top_k):
            results.append(SearchResult(
                text=self._index[idx].text,
                embedding=self._index[idx],
                score=score,
                rank=rank + 1,
                metadata=self._metadata[idx],
            ))

        return results

    async def search_by_vector(
        self,
        vector: List[float],
        k: int = 10,
    ) -> List[SearchResult]:
        """Search by raw vector."""
        if not self._index:
            return []

        # Create temporary embedding
        query_embedding = TextEmbedding(
            text="",
            embedding=vector,
            model="query",
            dimensions=len(vector),
        )

        scores = []
        for i, doc_embedding in enumerate(self._index):
            score = query_embedding.cosine_similarity(doc_embedding)
            scores.append((score, i))

        top_k = heapq.nlargest(k, scores, key=lambda x: x[0])

        results = []
        for rank, (score, idx) in enumerate(top_k):
            results.append(SearchResult(
                text=self._index[idx].text,
                embedding=self._index[idx],
                score=score,
                rank=rank + 1,
                metadata=self._metadata[idx],
            ))

        return results

    def remove(self, index: int) -> bool:
        """Remove text from index."""
        if 0 <= index < len(self._index):
            del self._index[index]
            del self._metadata[index]
            return True
        return False

    def clear(self) -> None:
        """Clear the index."""
        self._index.clear()
        self._metadata.clear()

    def __len__(self) -> int:
        """Get index size."""
        return len(self._index)


class SemanticClusterer:
    """
    Semantic clustering using embeddings.

    Groups similar texts together.
    """

    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        num_clusters: int = 5,
    ):
        self.generator = embedding_generator or EmbeddingGenerator()
        self.num_clusters = num_clusters
        self._centroids: List[List[float]] = []

    async def cluster(
        self,
        texts: List[str],
        max_iterations: int = 100,
    ) -> Dict[str, Any]:
        """Cluster texts using k-means."""
        if len(texts) < self.num_clusters:
            # Each text is its own cluster
            return {
                "clusters": [[i] for i in range(len(texts))],
                "centroids": [],
                "labels": list(range(len(texts))),
            }

        # Generate embeddings
        embeddings = await self.generator.embed_batch(texts)
        vectors = [e.embedding for e in embeddings]

        # Initialize centroids (k-means++)
        self._centroids = self._init_centroids(vectors)

        # K-means iterations
        labels = [0] * len(vectors)
        for iteration in range(max_iterations):
            # Assign points to nearest centroid
            new_labels = []
            for vector in vectors:
                distances = [
                    euclidean_distance(vector, centroid)
                    for centroid in self._centroids
                ]
                new_labels.append(distances.index(min(distances)))

            # Check convergence
            if new_labels == labels:
                break
            labels = new_labels

            # Update centroids
            for k in range(self.num_clusters):
                cluster_vectors = [v for v, l in zip(vectors, labels) if l == k]
                if cluster_vectors:
                    self._centroids[k] = [
                        sum(v[d] for v in cluster_vectors) / len(cluster_vectors)
                        for d in range(len(vectors[0]))
                    ]

        # Build clusters
        clusters = [[] for _ in range(self.num_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(i)

        return {
            "clusters": clusters,
            "centroids": self._centroids,
            "labels": labels,
            "texts_by_cluster": {
                k: [texts[i] for i in cluster]
                for k, cluster in enumerate(clusters)
            },
        }

    def _init_centroids(self, vectors: List[List[float]]) -> List[List[float]]:
        """Initialize centroids using k-means++."""
        import random

        centroids = [random.choice(vectors)]

        while len(centroids) < self.num_clusters:
            # Compute distances to nearest centroid
            distances = []
            for vector in vectors:
                min_dist = min(
                    euclidean_distance(vector, c) for c in centroids
                )
                distances.append(min_dist ** 2)

            # Choose next centroid proportionally to distance
            total = sum(distances)
            if total == 0:
                centroids.append(random.choice(vectors))
            else:
                r = random.uniform(0, total)
                cumsum = 0
                for i, d in enumerate(distances):
                    cumsum += d
                    if cumsum >= r:
                        centroids.append(vectors[i])
                        break

        return centroids

    async def predict(self, text: str) -> int:
        """Predict cluster for new text."""
        if not self._centroids:
            raise ValueError("Model not fitted. Call cluster() first.")

        embedding = await self.generator.embed(text)
        distances = [
            euclidean_distance(embedding.embedding, centroid)
            for centroid in self._centroids
        ]
        return distances.index(min(distances))


class EmbeddingIndex:
    """
    Persistent embedding index.

    Supports serialization and efficient retrieval.
    """

    def __init__(
        self,
        generator: Optional[EmbeddingGenerator] = None,
        index_path: Optional[str] = None,
    ):
        self.generator = generator or EmbeddingGenerator()
        self.index_path = index_path
        self._embeddings: Dict[str, TextEmbedding] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    async def add(
        self,
        key: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add text with key."""
        embedding = await self.generator.embed(text)
        self._embeddings[key] = embedding
        self._metadata[key] = metadata or {}

    async def get(self, key: str) -> Optional[TextEmbedding]:
        """Get embedding by key."""
        return self._embeddings.get(key)

    async def find_similar(
        self,
        key: str,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Find similar items to key."""
        if key not in self._embeddings:
            return []

        query_emb = self._embeddings[key]
        scores = []

        for other_key, other_emb in self._embeddings.items():
            if other_key != key:
                score = query_emb.cosine_similarity(other_emb)
                scores.append((other_key, score))

        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    def remove(self, key: str) -> bool:
        """Remove item by key."""
        if key in self._embeddings:
            del self._embeddings[key]
            del self._metadata[key]
            return True
        return False

    def keys(self) -> List[str]:
        """Get all keys."""
        return list(self._embeddings.keys())

    def __len__(self) -> int:
        """Get index size."""
        return len(self._embeddings)

    async def save(self, path: Optional[str] = None) -> None:
        """Save index to file."""
        import json

        save_path = path or self.index_path
        if not save_path:
            raise ValueError("No path specified")

        data = {
            "embeddings": {
                k: {
                    "text": v.text,
                    "embedding": v.embedding,
                    "model": v.model,
                    "dimensions": v.dimensions,
                }
                for k, v in self._embeddings.items()
            },
            "metadata": self._metadata,
        }

        with open(save_path, "w") as f:
            json.dump(data, f)

        logger.info(f"Saved {len(self)} embeddings to {save_path}")

    async def load(self, path: Optional[str] = None) -> None:
        """Load index from file."""
        import json

        load_path = path or self.index_path
        if not load_path:
            raise ValueError("No path specified")

        with open(load_path) as f:
            data = json.load(f)

        self._embeddings = {
            k: TextEmbedding(
                text=v["text"],
                embedding=v["embedding"],
                model=v["model"],
                dimensions=v["dimensions"],
            )
            for k, v in data["embeddings"].items()
        }
        self._metadata = data.get("metadata", {})

        logger.info(f"Loaded {len(self)} embeddings from {load_path}")


# Factory function
def create_embedding_model(
    model_type: str = "hash",
    **kwargs,
) -> EmbeddingModel:
    """Create embedding model by type."""
    config = EmbeddingConfig(
        model_type=EmbeddingModelType(model_type),
        **kwargs
    )

    models = {
        "hash": HashEmbeddingModel,
        "tfidf": TFIDFEmbeddingModel,
        "word2vec": Word2VecEmbeddingModel,
        "sentence_transformer": SentenceEmbeddingModel,
    }

    cls = models.get(model_type, HashEmbeddingModel)
    return cls(config)
