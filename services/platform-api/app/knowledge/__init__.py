"""
Knowledge Base Module

RAG-powered knowledge system:
- Document management
- Vector embeddings
- Semantic search
- Context retrieval
"""

from app.knowledge.service import KnowledgeService
from app.knowledge.routes import router

# Engine
from app.knowledge.engine import (
    KnowledgeBase,
    KnowledgeBaseConfig,
    KnowledgeBaseManager,
    Document,
    DocumentMetadata,
    DocumentChunk,
    DocumentType,
    ChunkingStrategy,
    EmbeddingModel,
    DocumentChunker,
    SearchResult,
    VectorStore,
    InMemoryVectorStore,
    get_knowledge_base_manager,
    create_document,
    create_faq_document,
)

# Embeddings
from app.knowledge.embeddings import (
    EmbeddingService,
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingCache,
    EmbeddingComparison,
    BaseEmbeddingProvider,
    OpenAIEmbeddingProvider,
    CohereEmbeddingProvider,
    VoyageEmbeddingProvider,
    LocalEmbeddingProvider,
    create_embedding_service,
    create_openai_embeddings,
    create_local_embeddings,
)

__all__ = [
    # Service
    "KnowledgeService",
    "router",
    # Engine
    "KnowledgeBase",
    "KnowledgeBaseConfig",
    "KnowledgeBaseManager",
    "Document",
    "DocumentMetadata",
    "DocumentChunk",
    "DocumentType",
    "ChunkingStrategy",
    "EmbeddingModel",
    "DocumentChunker",
    "SearchResult",
    "VectorStore",
    "InMemoryVectorStore",
    "get_knowledge_base_manager",
    "create_document",
    "create_faq_document",
    # Embeddings
    "EmbeddingService",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "EmbeddingCache",
    "EmbeddingComparison",
    "BaseEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "CohereEmbeddingProvider",
    "VoyageEmbeddingProvider",
    "LocalEmbeddingProvider",
    "create_embedding_service",
    "create_openai_embeddings",
    "create_local_embeddings",
]
