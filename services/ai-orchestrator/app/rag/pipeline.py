"""RAG Pipeline for retrieval-augmented generation."""

from dataclasses import dataclass, field
from typing import Optional

import structlog

from app.config import get_settings
from app.rag.embeddings import EmbeddingAdapter, OpenAIEmbeddings, MockEmbeddings
from app.rag.retriever import VectorRetriever, QdrantRetriever, MockRetriever, SearchResults
from app.rag.chunker import TextChunker, Document, Chunk, ChunkingStrategy

logger = structlog.get_logger()


@dataclass
class RAGResult:
    """Result from RAG pipeline."""
    query: str
    context: str  # Combined context from retrieved documents
    sources: list[dict]  # Source information
    search_results: SearchResults

    @property
    def has_context(self) -> bool:
        """Check if any context was retrieved."""
        return bool(self.context.strip())

    @property
    def source_count(self) -> int:
        """Get number of sources used."""
        return len(self.sources)


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    # Retrieval settings
    top_k: int = 5
    score_threshold: float = 0.7
    max_context_length: int = 2000

    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE

    # Context formatting
    context_template: str = "Context:\n{context}\n\n"
    source_separator: str = "\n---\n"

    # Filtering
    filter_duplicates: bool = True
    min_chunk_length: int = 50


class RAGPipeline:
    """
    RAG Pipeline for retrieval-augmented generation.

    Handles:
    - Document ingestion and chunking
    - Vector embedding and storage
    - Similarity search and retrieval
    - Context formatting for LLM
    """

    def __init__(
        self,
        embedding_adapter: Optional[EmbeddingAdapter] = None,
        retriever: Optional[VectorRetriever] = None,
        config: Optional[RAGConfig] = None,
    ) -> None:
        settings = get_settings()
        self.config = config or RAGConfig()

        # Initialize embedding adapter
        if embedding_adapter:
            self.embeddings = embedding_adapter
        elif settings.llm_provider == "mock":
            self.embeddings = MockEmbeddings()
        else:
            self.embeddings = OpenAIEmbeddings()

        # Initialize retriever
        if retriever:
            self.retriever = retriever
        elif settings.llm_provider == "mock":
            self.retriever = MockRetriever(self.embeddings)
        else:
            self.retriever = QdrantRetriever(self.embeddings)

        # Initialize chunker
        self.chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            strategy=self.config.chunking_strategy,
            min_chunk_size=self.config.min_chunk_length,
        )

        self.logger = logger.bind(component="rag_pipeline")

    async def ingest_document(
        self,
        text: str,
        metadata: Optional[dict] = None,
        source: Optional[str] = None,
    ) -> list[str]:
        """
        Ingest a single document.

        Args:
            text: Document text
            metadata: Optional metadata
            source: Optional source identifier

        Returns:
            List of chunk IDs
        """
        document = Document(
            id=source or f"doc_{hash(text[:100])}",
            content=text,
            metadata=metadata or {},
            source=source,
        )

        return await self.ingest_documents([document])

    async def ingest_documents(self, documents: list[Document]) -> list[str]:
        """
        Ingest multiple documents.

        Args:
            documents: List of documents to ingest

        Returns:
            List of chunk IDs
        """
        # Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            return []

        # Prepare for indexing
        texts = [c.content for c in all_chunks]
        metadatas = [c.metadata for c in all_chunks]
        ids = [c.id for c in all_chunks]

        # Add to retriever
        added_ids = await self.retriever.add_documents(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

        self.logger.info(
            "Ingested documents",
            document_count=len(documents),
            chunk_count=len(all_chunks),
        )

        return added_ids

    async def retrieve(
        self,
        query: str,
        filter_metadata: Optional[dict] = None,
    ) -> RAGResult:
        """
        Retrieve relevant context for a query.

        Args:
            query: Search query
            filter_metadata: Optional metadata filters

        Returns:
            RAGResult with context and sources
        """
        # Search for relevant chunks
        search_results = await self.retriever.search(
            query=query,
            top_k=self.config.top_k,
            filter_metadata=filter_metadata,
        )

        # Filter by score threshold
        relevant_results = [
            r for r in search_results.results
            if r.score >= self.config.score_threshold
        ]

        # Filter duplicates if enabled
        if self.config.filter_duplicates:
            seen_texts = set()
            unique_results = []
            for r in relevant_results:
                # Simple dedup by first 100 chars
                text_key = r.text[:100].lower()
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    unique_results.append(r)
            relevant_results = unique_results

        # Build context
        context_parts = []
        sources = []
        total_length = 0

        for result in relevant_results:
            # Check if adding this would exceed max length
            if total_length + len(result.text) > self.config.max_context_length:
                break

            context_parts.append(result.text)
            total_length += len(result.text)

            sources.append({
                "id": result.id,
                "score": result.score,
                "text_preview": result.text[:200],
                **result.metadata,
            })

        context = self.config.source_separator.join(context_parts)

        self.logger.debug(
            "Retrieved context",
            query=query[:50],
            results_found=len(search_results.results),
            results_used=len(context_parts),
            context_length=len(context),
        )

        return RAGResult(
            query=query,
            context=context,
            sources=sources,
            search_results=search_results,
        )

    async def query_with_context(
        self,
        query: str,
        filter_metadata: Optional[dict] = None,
    ) -> tuple[str, RAGResult]:
        """
        Get query with prepended context.

        Returns:
            Tuple of (augmented_prompt, rag_result)
        """
        rag_result = await self.retrieve(query, filter_metadata)

        if rag_result.has_context:
            augmented_prompt = self.config.context_template.format(
                context=rag_result.context
            ) + query
        else:
            augmented_prompt = query

        return augmented_prompt, rag_result

    async def delete_by_source(self, source: str) -> int:
        """
        Delete all chunks from a specific source.

        Args:
            source: Source identifier

        Returns:
            Number of chunks deleted
        """
        # Search for all chunks from this source
        results = await self.retriever.search(
            query="",  # Empty query to get all
            top_k=1000,
            filter_metadata={"source": source},
        )

        if not results.results:
            return 0

        ids = [r.id for r in results.results]
        return await self.retriever.delete_documents(ids)

    async def close(self) -> None:
        """Clean up resources."""
        await self.embeddings.close()
        await self.retriever.close()


class AgentRAGPipeline(RAGPipeline):
    """
    RAG Pipeline scoped to a specific agent.

    All operations are filtered by agent_id.
    """

    def __init__(
        self,
        agent_id: str,
        embedding_adapter: Optional[EmbeddingAdapter] = None,
        retriever: Optional[VectorRetriever] = None,
        config: Optional[RAGConfig] = None,
    ) -> None:
        super().__init__(embedding_adapter, retriever, config)
        self.agent_id = agent_id
        self.logger = logger.bind(
            component="agent_rag_pipeline",
            agent_id=agent_id,
        )

    async def ingest_document(
        self,
        text: str,
        metadata: Optional[dict] = None,
        source: Optional[str] = None,
    ) -> list[str]:
        """Ingest document with agent_id in metadata."""
        metadata = metadata or {}
        metadata["agent_id"] = self.agent_id
        return await super().ingest_document(text, metadata, source)

    async def retrieve(
        self,
        query: str,
        filter_metadata: Optional[dict] = None,
    ) -> RAGResult:
        """Retrieve with agent_id filter."""
        filter_metadata = filter_metadata or {}
        filter_metadata["agent_id"] = self.agent_id
        return await super().retrieve(query, filter_metadata)

    async def delete_all(self) -> int:
        """Delete all documents for this agent."""
        return await self.delete_by_source(f"agent:{self.agent_id}")
