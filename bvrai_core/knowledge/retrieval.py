"""
Retrieval Module

This module provides various retrieval strategies for RAG including
semantic search, hybrid search, and reranking.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from .base import (
    RetrievalResult,
    SearchFilter,
    SearchQuery,
)
from .embeddings import EmbeddingProvider
from .vectorstore import VectorStore


logger = logging.getLogger(__name__)


class RetrievalMode(str, Enum):
    """Retrieval modes."""
    SEMANTIC = "semantic"       # Vector similarity only
    KEYWORD = "keyword"         # BM25/keyword only
    HYBRID = "hybrid"           # Combined semantic + keyword
    MULTI_QUERY = "multi_query" # Multiple query variants


@dataclass
class RetrieverConfig:
    """Configuration for retrievers."""

    # Default search parameters
    top_k: int = 5
    min_score: float = 0.0

    # Retrieval mode
    mode: RetrievalMode = RetrievalMode.SEMANTIC

    # Hybrid search weights
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3

    # Reranking
    enable_reranking: bool = False
    rerank_top_k: int = 20
    rerank_model: Optional[str] = None

    # Query expansion
    enable_query_expansion: bool = False
    num_query_expansions: int = 3

    # MMR (Maximal Marginal Relevance)
    enable_mmr: bool = False
    mmr_lambda: float = 0.5

    # Context window
    include_surrounding_chunks: bool = False

    # Deduplication
    deduplicate: bool = True
    dedup_threshold: float = 0.95


class Retriever(ABC):
    """Abstract base class for retrievers."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        config: Optional[RetrieverConfig] = None,
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.config = config or RetrieverConfig()

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[SearchFilter] = None,
    ) -> List[RetrievalResult]:
        """Retrieve relevant chunks for a query."""
        pass

    async def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[SearchFilter] = None,
    ) -> List[Tuple[RetrievalResult, float]]:
        """Retrieve with explicit scores."""
        results = await self.retrieve(query, top_k, filters)
        return [(r, r.score) for r in results]


class SemanticRetriever(Retriever):
    """Simple semantic retriever using vector similarity."""

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[SearchFilter] = None,
    ) -> List[RetrievalResult]:
        """Retrieve using semantic search."""
        top_k = top_k or self.config.top_k

        # Generate query embedding
        embeddings = await self.embedding_provider.embed(query)
        query_embedding = embeddings[0]

        # Search vector store
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

        # Filter by minimum score
        if self.config.min_score > 0:
            results = [r for r in results if r.score >= self.config.min_score]

        return results


class HybridRetriever(Retriever):
    """
    Hybrid retriever combining semantic and keyword search.

    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        keyword_search_fn: Optional[Callable[[str, int], List[RetrievalResult]]] = None,
        config: Optional[RetrieverConfig] = None,
    ):
        super().__init__(vector_store, embedding_provider, config)
        self._keyword_search_fn = keyword_search_fn

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[SearchFilter] = None,
    ) -> List[RetrievalResult]:
        """Retrieve using hybrid search."""
        top_k = top_k or self.config.top_k

        # Semantic search
        embeddings = await self.embedding_provider.embed(query)
        query_embedding = embeddings[0]

        semantic_results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more for fusion
            filters=filters,
        )

        # Keyword search (if available)
        keyword_results = []
        if self._keyword_search_fn:
            keyword_results = await asyncio.to_thread(
                self._keyword_search_fn,
                query,
                top_k * 2,
            )

        # Fuse results using RRF
        if keyword_results:
            results = self._reciprocal_rank_fusion(
                semantic_results,
                keyword_results,
            )
        else:
            results = semantic_results

        # Apply deduplication
        if self.config.deduplicate:
            results = self._deduplicate(results)

        return results[:top_k]

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        k: int = 60,  # RRF constant
    ) -> List[RetrievalResult]:
        """Combine results using Reciprocal Rank Fusion."""
        scores: Dict[str, float] = {}
        results_map: Dict[str, RetrievalResult] = {}

        # Score semantic results
        for rank, result in enumerate(semantic_results):
            rrf_score = self.config.semantic_weight / (k + rank + 1)
            scores[result.chunk_id] = scores.get(result.chunk_id, 0) + rrf_score
            results_map[result.chunk_id] = result
            result.semantic_score = result.score

        # Score keyword results
        for rank, result in enumerate(keyword_results):
            rrf_score = self.config.keyword_weight / (k + rank + 1)
            scores[result.chunk_id] = scores.get(result.chunk_id, 0) + rrf_score
            if result.chunk_id not in results_map:
                results_map[result.chunk_id] = result
            results_map[result.chunk_id].keyword_score = result.score

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        results = []
        for chunk_id in sorted_ids:
            result = results_map[chunk_id]
            result.score = scores[chunk_id]
            results.append(result)

        return results

    def _deduplicate(
        self,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Remove near-duplicate results."""
        if not results:
            return results

        unique_results = [results[0]]

        for result in results[1:]:
            is_duplicate = False
            for unique in unique_results:
                # Simple content similarity check
                if self._content_similarity(result.content, unique.content) > self.config.dedup_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_results.append(result)

        return unique_results

    def _content_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple content similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


class MultiQueryRetriever(Retriever):
    """
    Retriever that generates multiple query variants for better recall.

    Uses an LLM to generate query variations.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        query_generator: Optional[Callable[[str, int], List[str]]] = None,
        config: Optional[RetrieverConfig] = None,
    ):
        super().__init__(vector_store, embedding_provider, config)
        self._query_generator = query_generator

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[SearchFilter] = None,
    ) -> List[RetrievalResult]:
        """Retrieve using multiple query variants."""
        top_k = top_k or self.config.top_k

        # Generate query variations
        queries = [query]
        if self._query_generator and self.config.enable_query_expansion:
            try:
                additional_queries = await asyncio.to_thread(
                    self._query_generator,
                    query,
                    self.config.num_query_expansions,
                )
                queries.extend(additional_queries)
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")

        # Retrieve for each query
        all_results: Dict[str, RetrievalResult] = {}
        scores: Dict[str, List[float]] = {}

        for q in queries:
            embeddings = await self.embedding_provider.embed(q)
            query_embedding = embeddings[0]

            results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
            )

            for result in results:
                if result.chunk_id not in all_results:
                    all_results[result.chunk_id] = result
                    scores[result.chunk_id] = []
                scores[result.chunk_id].append(result.score)

        # Combine scores (average or max)
        for chunk_id, result in all_results.items():
            result.score = max(scores[chunk_id])  # Use max score

        # Sort by score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.score,
            reverse=True,
        )

        return sorted_results[:top_k]


class RerankedRetriever(Retriever):
    """
    Retriever with cross-encoder reranking for improved precision.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        reranker: Optional[Callable[[str, List[str]], List[float]]] = None,
        config: Optional[RetrieverConfig] = None,
    ):
        super().__init__(vector_store, embedding_provider, config)
        self._reranker = reranker
        self._default_reranker = None

    async def _get_reranker(self):
        """Get or create default reranker."""
        if self._reranker:
            return self._reranker

        if self._default_reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                model_name = self.config.rerank_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
                self._default_reranker = CrossEncoder(model_name)
            except ImportError:
                logger.warning("sentence-transformers not installed, reranking disabled")
                return None

        return self._default_reranker

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[SearchFilter] = None,
    ) -> List[RetrievalResult]:
        """Retrieve with reranking."""
        top_k = top_k or self.config.top_k
        initial_k = self.config.rerank_top_k if self.config.enable_reranking else top_k

        # Initial retrieval
        embeddings = await self.embedding_provider.embed(query)
        query_embedding = embeddings[0]

        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=initial_k,
            filters=filters,
        )

        # Rerank if enabled
        if self.config.enable_reranking and results:
            results = await self._rerank(query, results)

        return results[:top_k]

    async def _rerank(
        self,
        query: str,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Rerank results using cross-encoder."""
        reranker = await self._get_reranker()
        if not reranker:
            return results

        # Prepare pairs for reranking
        pairs = [(query, r.content) for r in results]

        # Get rerank scores
        if callable(reranker):
            scores = await asyncio.to_thread(
                reranker,
                query,
                [r.content for r in results],
            )
        else:
            # CrossEncoder model
            scores = await asyncio.to_thread(reranker.predict, pairs)

        # Update scores
        for result, score in zip(results, scores):
            result.rerank_score = float(score)
            # Combine with original score
            result.score = 0.3 * result.score + 0.7 * float(score)

        # Sort by new score
        results.sort(key=lambda x: x.score, reverse=True)

        return results


class MMRRetriever(Retriever):
    """
    Retriever using Maximal Marginal Relevance for diversity.

    Balances relevance with diversity in results.
    """

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[SearchFilter] = None,
    ) -> List[RetrievalResult]:
        """Retrieve with MMR for diversity."""
        top_k = top_k or self.config.top_k
        fetch_k = top_k * 4  # Fetch more for MMR selection

        # Get initial results
        embeddings = await self.embedding_provider.embed(query)
        query_embedding = embeddings[0]

        initial_results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            filters=filters,
        )

        if not initial_results:
            return []

        # Get embeddings for all results
        result_contents = [r.content for r in initial_results]
        result_embeddings = await self.embedding_provider.embed(result_contents)

        # Apply MMR
        selected = await self._mmr_select(
            query_embedding,
            initial_results,
            result_embeddings,
            top_k,
        )

        return selected

    async def _mmr_select(
        self,
        query_embedding: List[float],
        results: List[RetrievalResult],
        embeddings: List[List[float]],
        k: int,
    ) -> List[RetrievalResult]:
        """Select results using MMR algorithm."""
        from .embeddings import cosine_similarity

        lambda_param = self.config.mmr_lambda
        selected_indices: List[int] = []
        remaining_indices = list(range(len(results)))

        # Calculate query similarities
        query_sims = [
            cosine_similarity(query_embedding, emb)
            for emb in embeddings
        ]

        for _ in range(min(k, len(results))):
            if not remaining_indices:
                break

            mmr_scores = []

            for idx in remaining_indices:
                # Relevance to query
                query_sim = query_sims[idx]

                # Maximum similarity to already selected
                max_sim_to_selected = 0.0
                if selected_indices:
                    for sel_idx in selected_indices:
                        sim = cosine_similarity(
                            embeddings[idx],
                            embeddings[sel_idx],
                        )
                        max_sim_to_selected = max(max_sim_to_selected, sim)

                # MMR score
                mmr = lambda_param * query_sim - (1 - lambda_param) * max_sim_to_selected
                mmr_scores.append((idx, mmr))

            # Select highest MMR score
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [results[i] for i in selected_indices]


# Convenience function for creating retrievers
def create_retriever(
    vector_store: VectorStore,
    embedding_provider: EmbeddingProvider,
    mode: RetrievalMode = RetrievalMode.SEMANTIC,
    config: Optional[RetrieverConfig] = None,
    **kwargs,
) -> Retriever:
    """Create a retriever for the specified mode."""
    config = config or RetrieverConfig(mode=mode)

    retrievers = {
        RetrievalMode.SEMANTIC: SemanticRetriever,
        RetrievalMode.HYBRID: HybridRetriever,
        RetrievalMode.MULTI_QUERY: MultiQueryRetriever,
    }

    # Use reranked retriever if reranking is enabled
    if config.enable_reranking:
        return RerankedRetriever(
            vector_store,
            embedding_provider,
            config=config,
            **kwargs,
        )

    # Use MMR retriever if MMR is enabled
    if config.enable_mmr:
        return MMRRetriever(
            vector_store,
            embedding_provider,
            config=config,
        )

    retriever_class = retrievers.get(mode, SemanticRetriever)
    return retriever_class(
        vector_store,
        embedding_provider,
        config=config,
        **kwargs,
    )


__all__ = [
    "RetrievalMode",
    "RetrieverConfig",
    "Retriever",
    "SemanticRetriever",
    "HybridRetriever",
    "MultiQueryRetriever",
    "RerankedRetriever",
    "MMRRetriever",
    "create_retriever",
]
