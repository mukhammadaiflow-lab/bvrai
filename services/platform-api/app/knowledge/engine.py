"""
Knowledge Base Engine

RAG (Retrieval-Augmented Generation) system:
- Document storage
- Vector embeddings
- Semantic search
- Context retrieval
"""

from typing import Optional, Dict, Any, List, Callable, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import asyncio
import hashlib
import json
import logging
import re

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Types of documents."""
    TEXT = "text"
    FAQ = "faq"
    ARTICLE = "article"
    POLICY = "policy"
    PROCEDURE = "procedure"
    SCRIPT = "script"
    PRODUCT = "product"
    CUSTOM = "custom"


class ChunkingStrategy(str, Enum):
    """Document chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"


class EmbeddingModel(str, Enum):
    """Embedding model types."""
    OPENAI_ADA = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"
    COHERE = "embed-english-v3.0"
    VOYAGE = "voyage-large-2"
    LOCAL = "local"


@dataclass
class DocumentMetadata:
    """Document metadata."""
    title: str = ""
    description: str = ""
    author: str = ""
    source: str = ""
    source_url: str = ""
    language: str = "en"
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Document:
    """Knowledge base document."""
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    doc_type: DocumentType = DocumentType.TEXT
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)

    # Organization
    tenant_id: str = ""
    knowledge_base_id: str = ""

    # Processing
    is_processed: bool = False
    chunk_count: int = 0
    content_hash: str = ""

    # Status
    is_active: bool = True

    def __post_init__(self):
        if not self.content_hash and self.content:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "doc_type": self.doc_type.value,
            "metadata": {
                "title": self.metadata.title,
                "description": self.metadata.description,
                "tags": self.metadata.tags,
                "source": self.metadata.source,
            },
            "is_processed": self.is_processed,
            "chunk_count": self.chunk_count,
            "is_active": self.is_active,
        }


@dataclass
class DocumentChunk:
    """Chunk of a document."""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    content: str = ""
    chunk_index: int = 0

    # Embedding
    embedding: Optional[List[float]] = None
    embedding_model: str = ""

    # Context
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    # Metadata
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """Search result from knowledge base."""
    chunk: DocumentChunk
    score: float
    document: Optional[Document] = None
    highlights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "content": self.chunk.content,
            "score": self.score,
            "highlights": self.highlights,
            "metadata": self.metadata,
        }


@dataclass
class KnowledgeBaseConfig:
    """Knowledge base configuration."""
    kb_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Embedding
    embedding_model: EmbeddingModel = EmbeddingModel.OPENAI_3_SMALL
    embedding_dimensions: int = 1536

    # Chunking
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 500  # tokens
    chunk_overlap: int = 50

    # Search
    default_top_k: int = 5
    similarity_threshold: float = 0.7
    rerank_enabled: bool = False

    # Metadata
    tenant_id: str = ""
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class DocumentChunker:
    """
    Chunks documents into smaller pieces.

    Strategies:
    - Fixed size
    - Sentence-based
    - Paragraph-based
    - Semantic chunking
    - Recursive splitting
    """

    def __init__(self, config: KnowledgeBaseConfig):
        self.config = config

    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk document based on strategy."""
        strategy = self.config.chunking_strategy

        if strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(document)
        elif strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_sentences(document)
        elif strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_paragraphs(document)
        elif strategy == ChunkingStrategy.RECURSIVE:
            return self._chunk_recursive(document)
        else:
            return self._chunk_fixed_size(document)

    def _chunk_fixed_size(self, document: Document) -> List[DocumentChunk]:
        """Split into fixed-size chunks."""
        chunks = []
        content = document.content
        chunk_size = self.config.chunk_size * 4  # Approximate chars per token
        overlap = self.config.chunk_overlap * 4

        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))

            # Find word boundary
            if end < len(content):
                space_pos = content.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos

            chunk_text = content[start:end].strip()

            if chunk_text:
                chunks.append(DocumentChunk(
                    document_id=document.document_id,
                    content=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    token_count=len(chunk_text) // 4,
                ))
                chunk_index += 1

            start = end - overlap if end < len(content) else end

        # Link chunks
        self._link_chunks(chunks)

        return chunks

    def _chunk_sentences(self, document: Document) -> List[DocumentChunk]:
        """Split by sentences."""
        chunks = []
        content = document.content

        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', content)

        current_chunk = ""
        current_start = 0
        chunk_index = 0
        max_tokens = self.config.chunk_size

        for sentence in sentences:
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(potential_chunk) // 4 > max_tokens and current_chunk:
                # Save current chunk
                chunks.append(DocumentChunk(
                    document_id=document.document_id,
                    content=current_chunk.strip(),
                    chunk_index=chunk_index,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    token_count=len(current_chunk) // 4,
                ))
                chunk_index += 1
                current_start += len(current_chunk)
                current_chunk = sentence
            else:
                current_chunk = potential_chunk

        # Add remaining
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                document_id=document.document_id,
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=len(content),
                token_count=len(current_chunk) // 4,
            ))

        self._link_chunks(chunks)
        return chunks

    def _chunk_paragraphs(self, document: Document) -> List[DocumentChunk]:
        """Split by paragraphs."""
        chunks = []
        content = document.content

        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', content)

        current_chunk = ""
        current_start = 0
        chunk_index = 0
        max_tokens = self.config.chunk_size

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            potential_chunk = current_chunk + "\n\n" + para if current_chunk else para

            if len(potential_chunk) // 4 > max_tokens and current_chunk:
                chunks.append(DocumentChunk(
                    document_id=document.document_id,
                    content=current_chunk.strip(),
                    chunk_index=chunk_index,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    token_count=len(current_chunk) // 4,
                ))
                chunk_index += 1
                current_start += len(current_chunk)
                current_chunk = para
            else:
                current_chunk = potential_chunk

        if current_chunk.strip():
            chunks.append(DocumentChunk(
                document_id=document.document_id,
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=len(content),
                token_count=len(current_chunk) // 4,
            ))

        self._link_chunks(chunks)
        return chunks

    def _chunk_recursive(self, document: Document) -> List[DocumentChunk]:
        """Recursively split by multiple separators."""
        separators = ["\n\n", "\n", ". ", " "]
        return self._recursive_split(
            document,
            document.content,
            separators,
            0,
        )

    def _recursive_split(
        self,
        document: Document,
        text: str,
        separators: List[str],
        start_char: int,
    ) -> List[DocumentChunk]:
        """Recursively split text."""
        chunks = []
        max_size = self.config.chunk_size * 4

        if len(text) <= max_size or not separators:
            if text.strip():
                chunks.append(DocumentChunk(
                    document_id=document.document_id,
                    content=text.strip(),
                    chunk_index=0,  # Will be updated later
                    start_char=start_char,
                    end_char=start_char + len(text),
                    token_count=len(text) // 4,
                ))
            return chunks

        separator = separators[0]
        parts = text.split(separator)

        current_chunk = ""
        current_start = start_char

        for part in parts:
            potential = current_chunk + separator + part if current_chunk else part

            if len(potential) <= max_size:
                current_chunk = potential
            else:
                if current_chunk:
                    if len(current_chunk) <= max_size:
                        chunks.append(DocumentChunk(
                            document_id=document.document_id,
                            content=current_chunk.strip(),
                            chunk_index=0,
                            start_char=current_start,
                            end_char=current_start + len(current_chunk),
                            token_count=len(current_chunk) // 4,
                        ))
                    else:
                        # Recurse with next separator
                        sub_chunks = self._recursive_split(
                            document,
                            current_chunk,
                            separators[1:],
                            current_start,
                        )
                        chunks.extend(sub_chunks)

                    current_start += len(current_chunk) + len(separator)

                current_chunk = part

        # Handle remaining
        if current_chunk:
            if len(current_chunk) <= max_size:
                chunks.append(DocumentChunk(
                    document_id=document.document_id,
                    content=current_chunk.strip(),
                    chunk_index=0,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    token_count=len(current_chunk) // 4,
                ))
            else:
                sub_chunks = self._recursive_split(
                    document,
                    current_chunk,
                    separators[1:],
                    current_start,
                )
                chunks.extend(sub_chunks)

        # Update indices
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        self._link_chunks(chunks)
        return chunks

    def _link_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Link chunks to each other."""
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.previous_chunk_id = chunks[i - 1].chunk_id
            if i < len(chunks) - 1:
                chunk.next_chunk_id = chunks[i + 1].chunk_id


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    async def add(self, chunks: List[DocumentChunk]) -> None:
        """Add chunks with embeddings."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks."""
        pass

    @abstractmethod
    async def delete(self, document_id: str) -> None:
        """Delete document chunks."""
        pass


class InMemoryVectorStore(VectorStore):
    """In-memory vector store implementation."""

    def __init__(self):
        self._chunks: Dict[str, DocumentChunk] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._by_document: Dict[str, Set[str]] = {}

    async def add(self, chunks: List[DocumentChunk]) -> None:
        """Add chunks."""
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            if chunk.embedding:
                self._embeddings[chunk.chunk_id] = chunk.embedding

            if chunk.document_id not in self._by_document:
                self._by_document[chunk.document_id] = set()
            self._by_document[chunk.document_id].add(chunk.chunk_id)

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search by cosine similarity."""
        results = []

        for chunk_id, embedding in self._embeddings.items():
            chunk = self._chunks.get(chunk_id)
            if not chunk:
                continue

            # Apply filters
            if filter_dict:
                if "document_id" in filter_dict:
                    if chunk.document_id != filter_dict["document_id"]:
                        continue

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, embedding)
            results.append((chunk, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def delete(self, document_id: str) -> None:
        """Delete document chunks."""
        chunk_ids = self._by_document.pop(document_id, set())
        for chunk_id in chunk_ids:
            self._chunks.pop(chunk_id, None)
            self._embeddings.pop(chunk_id, None)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        import math
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)


class KnowledgeBase:
    """
    Complete knowledge base system.

    Features:
    - Document management
    - Chunking and embedding
    - Semantic search
    - Context retrieval
    """

    def __init__(
        self,
        config: KnowledgeBaseConfig,
        vector_store: Optional[VectorStore] = None,
        embedding_func: Optional[Callable[[str], List[float]]] = None,
    ):
        self.config = config
        self._vector_store = vector_store or InMemoryVectorStore()
        self._embedding_func = embedding_func
        self._chunker = DocumentChunker(config)
        self._documents: Dict[str, Document] = {}

    async def add_document(self, document: Document) -> Document:
        """Add document to knowledge base."""
        document.knowledge_base_id = self.config.kb_id

        # Check for duplicate
        if document.content_hash in [d.content_hash for d in self._documents.values()]:
            logger.warning(f"Duplicate document content: {document.document_id}")

        # Chunk document
        chunks = self._chunker.chunk(document)

        # Generate embeddings
        if self._embedding_func:
            for chunk in chunks:
                chunk.embedding = self._embedding_func(chunk.content)
                chunk.embedding_model = self.config.embedding_model.value

        # Store chunks
        await self._vector_store.add(chunks)

        # Update document
        document.is_processed = True
        document.chunk_count = len(chunks)
        self._documents[document.document_id] = document

        logger.info(f"Added document {document.document_id} with {len(chunks)} chunks")
        return document

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_document: bool = True,
    ) -> List[SearchResult]:
        """Search knowledge base."""
        k = top_k or self.config.default_top_k

        # Generate query embedding
        if not self._embedding_func:
            logger.error("No embedding function configured")
            return []

        query_embedding = self._embedding_func(query)

        # Search vector store
        results = await self._vector_store.search(
            query_embedding,
            top_k=k,
            filter_dict=filter_dict,
        )

        # Filter by threshold
        threshold = self.config.similarity_threshold
        filtered_results = [
            (chunk, score) for chunk, score in results
            if score >= threshold
        ]

        # Build search results
        search_results = []
        for chunk, score in filtered_results:
            doc = self._documents.get(chunk.document_id) if include_document else None

            # Generate highlights
            highlights = self._generate_highlights(query, chunk.content)

            search_results.append(SearchResult(
                chunk=chunk,
                score=score,
                document=doc,
                highlights=highlights,
                metadata={
                    "document_title": doc.metadata.title if doc else "",
                    "chunk_index": chunk.chunk_index,
                },
            ))

        return search_results

    def _generate_highlights(self, query: str, content: str) -> List[str]:
        """Generate text highlights for search results."""
        highlights = []
        query_words = set(query.lower().split())

        sentences = re.split(r'(?<=[.!?])\s+', content)
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = query_words & sentence_words
            if overlap:
                highlights.append(sentence[:200])

        return highlights[:3]

    async def get_context(
        self,
        query: str,
        max_tokens: int = 2000,
        include_adjacent: bool = True,
    ) -> str:
        """Get relevant context for a query."""
        results = await self.search(query, top_k=10)

        context_parts = []
        total_tokens = 0

        for result in results:
            chunk_content = result.chunk.content
            chunk_tokens = len(chunk_content) // 4

            if total_tokens + chunk_tokens > max_tokens:
                break

            # Include adjacent chunks for context
            if include_adjacent:
                prev_id = result.chunk.previous_chunk_id
                next_id = result.chunk.next_chunk_id

                # This would need access to chunk storage
                # Simplified here
                pass

            context_parts.append(chunk_content)
            total_tokens += chunk_tokens

        return "\n\n".join(context_parts)

    def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self._documents.get(document_id)

    def list_documents(
        self,
        doc_type: Optional[DocumentType] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Document]:
        """List documents with optional filters."""
        docs = list(self._documents.values())

        if doc_type:
            docs = [d for d in docs if d.doc_type == doc_type]

        if tags:
            docs = [d for d in docs if any(t in d.metadata.tags for t in tags)]

        return docs

    async def delete_document(self, document_id: str) -> bool:
        """Delete document and its chunks."""
        if document_id not in self._documents:
            return False

        await self._vector_store.delete(document_id)
        del self._documents[document_id]

        logger.info(f"Deleted document: {document_id}")
        return True

    async def update_document(self, document: Document) -> Document:
        """Update document (re-process)."""
        await self.delete_document(document.document_id)
        return await self.add_document(document)

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        docs = list(self._documents.values())
        return {
            "kb_id": self.config.kb_id,
            "name": self.config.name,
            "total_documents": len(docs),
            "total_chunks": sum(d.chunk_count for d in docs),
            "by_type": {
                t.value: sum(1 for d in docs if d.doc_type == t)
                for t in DocumentType
            },
            "embedding_model": self.config.embedding_model.value,
        }


class KnowledgeBaseManager:
    """
    Manages multiple knowledge bases.

    Features:
    - Multi-tenant support
    - Knowledge base lifecycle
    - Cross-KB search
    """

    def __init__(self):
        self._knowledge_bases: Dict[str, KnowledgeBase] = {}
        self._embedding_providers: Dict[str, Callable[[str], List[float]]] = {}

    def register_embedding_provider(
        self,
        model: EmbeddingModel,
        provider: Callable[[str], List[float]],
    ) -> None:
        """Register embedding provider."""
        self._embedding_providers[model.value] = provider

    def create_knowledge_base(
        self,
        config: KnowledgeBaseConfig,
        vector_store: Optional[VectorStore] = None,
    ) -> KnowledgeBase:
        """Create new knowledge base."""
        embedding_func = self._embedding_providers.get(config.embedding_model.value)

        kb = KnowledgeBase(
            config=config,
            vector_store=vector_store,
            embedding_func=embedding_func,
        )

        self._knowledge_bases[config.kb_id] = kb
        logger.info(f"Created knowledge base: {config.kb_id}")
        return kb

    def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBase]:
        """Get knowledge base by ID."""
        return self._knowledge_bases.get(kb_id)

    def list_knowledge_bases(
        self,
        tenant_id: Optional[str] = None,
    ) -> List[KnowledgeBase]:
        """List knowledge bases."""
        kbs = list(self._knowledge_bases.values())

        if tenant_id:
            kbs = [kb for kb in kbs if kb.config.tenant_id == tenant_id]

        return kbs

    def delete_knowledge_base(self, kb_id: str) -> bool:
        """Delete knowledge base."""
        return self._knowledge_bases.pop(kb_id, None) is not None

    async def search_all(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Search across all knowledge bases."""
        all_results = []

        for kb in self.list_knowledge_bases(tenant_id):
            results = await kb.search(query, top_k=top_k)
            for result in results:
                result.metadata["knowledge_base_id"] = kb.config.kb_id
                result.metadata["knowledge_base_name"] = kb.config.name
            all_results.extend(results)

        # Sort by score
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "total_knowledge_bases": len(self._knowledge_bases),
            "knowledge_bases": [
                kb.get_stats() for kb in self._knowledge_bases.values()
            ],
        }


# Singleton manager instance
_manager_instance: Optional[KnowledgeBaseManager] = None


def get_knowledge_base_manager() -> KnowledgeBaseManager:
    """Get singleton knowledge base manager."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = KnowledgeBaseManager()
    return _manager_instance


# Helper functions
def create_document(
    content: str,
    title: str = "",
    doc_type: DocumentType = DocumentType.TEXT,
    tags: Optional[List[str]] = None,
    **kwargs,
) -> Document:
    """Create a document."""
    return Document(
        content=content,
        doc_type=doc_type,
        metadata=DocumentMetadata(
            title=title,
            tags=tags or [],
            **kwargs,
        ),
    )


def create_faq_document(
    question: str,
    answer: str,
    tags: Optional[List[str]] = None,
) -> Document:
    """Create FAQ document."""
    return Document(
        content=f"Q: {question}\nA: {answer}",
        doc_type=DocumentType.FAQ,
        metadata=DocumentMetadata(
            title=question[:100],
            tags=tags or ["faq"],
        ),
    )
