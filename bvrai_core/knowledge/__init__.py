"""
Builder Engine Knowledge Base System

A comprehensive knowledge management and RAG system that provides:
- Multi-format document ingestion (PDF, DOCX, TXT, HTML, etc.)
- Intelligent text chunking strategies
- Multi-provider embeddings (OpenAI, Cohere, Voyage, etc.)
- Vector store integrations (Pinecone, Weaviate, Qdrant, Chroma)
- Semantic search and retrieval
- Hybrid search (dense + sparse)
- Knowledge graph capabilities

Designed for building intelligent AI agents with domain-specific knowledge.
"""

from .base import (
    DocumentType,
    DocumentMetadata,
    ChunkMetadata,
    RetrievalResult,
    SearchQuery,
    SearchFilter,
)
from .documents import (
    Document,
    DocumentChunk,
    DocumentProcessor,
    PDFProcessor,
    DocxProcessor,
    TextProcessor,
    HTMLProcessor,
    MarkdownProcessor,
)
from .chunking import (
    ChunkingStrategy,
    ChunkingConfig,
    TextChunker,
    SentenceChunker,
    SemanticChunker,
    RecursiveChunker,
)
from .embeddings import (
    EmbeddingProvider,
    EmbeddingConfig,
    OpenAIEmbeddings,
    CohereEmbeddings,
    VoyageEmbeddings,
    HuggingFaceEmbeddings,
    EmbeddingProviderFactory,
)
from .vectorstore import (
    VectorStore,
    VectorStoreConfig,
    PineconeStore,
    QdrantStore,
    ChromaStore,
    InMemoryStore,
    VectorStoreFactory,
)
from .retrieval import (
    Retriever,
    RetrieverConfig,
    HybridRetriever,
    MultiQueryRetriever,
    RerankedRetriever,
)
from .ingestion import (
    IngestionPipeline,
    IngestionConfig,
    IngestionResult,
)

__all__ = [
    # Base
    "DocumentType",
    "DocumentMetadata",
    "ChunkMetadata",
    "RetrievalResult",
    "SearchQuery",
    "SearchFilter",
    # Documents
    "Document",
    "DocumentChunk",
    "DocumentProcessor",
    "PDFProcessor",
    "DocxProcessor",
    "TextProcessor",
    "HTMLProcessor",
    "MarkdownProcessor",
    # Chunking
    "ChunkingStrategy",
    "ChunkingConfig",
    "TextChunker",
    "SentenceChunker",
    "SemanticChunker",
    "RecursiveChunker",
    # Embeddings
    "EmbeddingProvider",
    "EmbeddingConfig",
    "OpenAIEmbeddings",
    "CohereEmbeddings",
    "VoyageEmbeddings",
    "HuggingFaceEmbeddings",
    "EmbeddingProviderFactory",
    # Vector Store
    "VectorStore",
    "VectorStoreConfig",
    "PineconeStore",
    "QdrantStore",
    "ChromaStore",
    "InMemoryStore",
    "VectorStoreFactory",
    # Retrieval
    "Retriever",
    "RetrieverConfig",
    "HybridRetriever",
    "MultiQueryRetriever",
    "RerankedRetriever",
    # Ingestion
    "IngestionPipeline",
    "IngestionConfig",
    "IngestionResult",
]
