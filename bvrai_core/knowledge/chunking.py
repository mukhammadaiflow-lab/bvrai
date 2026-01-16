"""
Text Chunking Module

This module provides various strategies for chunking documents
into smaller pieces suitable for embedding and retrieval.
"""

import logging
import re
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
    ChunkMetadata,
    generate_chunk_id,
    estimate_tokens,
)
from .documents import Document, DocumentChunk


logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    # Fixed size chunks
    FIXED = "fixed"

    # Sentence-based chunking
    SENTENCE = "sentence"

    # Paragraph-based chunking
    PARAGRAPH = "paragraph"

    # Semantic chunking (topic-based)
    SEMANTIC = "semantic"

    # Recursive (hierarchical) chunking
    RECURSIVE = "recursive"

    # Section-based (using document structure)
    SECTION = "section"

    # Sliding window
    SLIDING_WINDOW = "sliding_window"


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""

    # Strategy
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE

    # Size limits
    chunk_size: int = 1000        # Target chunk size in characters
    chunk_overlap: int = 200      # Overlap between chunks
    min_chunk_size: int = 100     # Minimum chunk size
    max_chunk_size: int = 2000    # Maximum chunk size

    # Token-based sizing (alternative to character-based)
    use_token_counting: bool = False
    max_tokens: int = 500
    token_overlap: int = 50

    # Separator preferences
    separators: List[str] = field(default_factory=lambda: [
        "\n\n\n",      # Triple newline (section break)
        "\n\n",        # Double newline (paragraph)
        "\n",          # Single newline
        ". ",          # Sentence
        "! ",
        "? ",
        "; ",          # Clause
        ", ",          # Phrase
        " ",           # Word
    ])

    # Sentence-based options
    min_sentences: int = 1
    max_sentences: int = 10

    # Preserve structure
    preserve_paragraphs: bool = True
    preserve_sentences: bool = True
    respect_section_boundaries: bool = True

    # Metadata enrichment
    add_section_context: bool = True
    include_document_title: bool = False


class TextChunker(ABC):
    """Abstract base class for text chunkers."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    @abstractmethod
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk a document into smaller pieces."""
        pass

    def _create_chunk(
        self,
        content: str,
        document: Document,
        chunk_index: int,
        total_chunks: int,
        start_char: int,
        end_char: int,
        **extra_metadata,
    ) -> DocumentChunk:
        """Create a DocumentChunk with metadata."""
        metadata = ChunkMetadata(
            document_id=document.id,
            document_source=document.metadata.source,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            start_char=start_char,
            end_char=end_char,
            chunking_strategy=self.__class__.__name__,
            overlap_chars=self.config.chunk_overlap,
        )

        # Add extra metadata
        for key, value in extra_metadata.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
            else:
                metadata.custom[key] = value

        # Inherit document metadata
        metadata.custom.update({
            "document_title": document.metadata.title,
            "document_source": document.metadata.source,
            "document_tags": document.metadata.tags,
        })

        chunk_id = generate_chunk_id(document.id, chunk_index)

        return DocumentChunk(
            id=chunk_id,
            content=content,
            metadata=metadata,
            document_id=document.id,
        )


class FixedChunker(TextChunker):
    """Simple fixed-size chunker with overlap."""

    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk document into fixed-size pieces."""
        text = document.content
        chunks = []

        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = min(start + chunk_size, len(text))

            # Extract chunk
            chunk_text = text[start:end]

            # Try to end at a word boundary
            if end < len(text) and not text[end].isspace():
                last_space = chunk_text.rfind(' ')
                if last_space > chunk_size // 2:
                    chunk_text = chunk_text[:last_space]
                    end = start + last_space

            chunks.append((chunk_text.strip(), start, end))

            # Move to next position with overlap
            start = end - overlap
            if start >= len(text) - overlap:
                break

            chunk_index += 1

        # Create DocumentChunk objects
        total_chunks = len(chunks)
        return [
            self._create_chunk(
                content=content,
                document=document,
                chunk_index=i,
                total_chunks=total_chunks,
                start_char=start,
                end_char=end,
            )
            for i, (content, start, end) in enumerate(chunks)
        ]


class SentenceChunker(TextChunker):
    """Sentence-based chunker that preserves sentence boundaries."""

    # Sentence splitting pattern
    SENTENCE_PATTERN = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])|'  # Standard sentence end
        r'(?<=[.!?])\s*\n+|'        # Sentence end with newline
        r'\n{2,}'                    # Paragraph breaks
    )

    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk document by sentences."""
        text = document.content

        # Split into sentences
        sentences = self._split_sentences(text)

        chunks = []
        current_chunk = []
        current_size = 0
        chunk_start = 0
        current_position = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # Check if adding sentence exceeds limits
            if current_size + sentence_size > self.config.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunk_end = current_position
                chunks.append((chunk_text, chunk_start, chunk_end))

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk,
                    self.config.chunk_overlap,
                )
                current_chunk = overlap_sentences
                current_size = sum(len(s) for s in current_chunk)
                chunk_start = chunk_end - current_size

            current_chunk.append(sentence)
            current_size += sentence_size
            current_position += sentence_size

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append((chunk_text, chunk_start, len(text)))

        # Create DocumentChunk objects
        total_chunks = len(chunks)
        return [
            self._create_chunk(
                content=content,
                document=document,
                chunk_index=i,
                total_chunks=total_chunks,
                start_char=start,
                end_char=end,
            )
            for i, (content, start, end) in enumerate(chunks)
        ]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = self.SENTENCE_PATTERN.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_sentences(
        self,
        sentences: List[str],
        target_overlap: int,
    ) -> List[str]:
        """Get sentences for overlap."""
        overlap_sentences = []
        current_size = 0

        for sentence in reversed(sentences):
            if current_size + len(sentence) > target_overlap:
                break
            overlap_sentences.insert(0, sentence)
            current_size += len(sentence)

        return overlap_sentences


class RecursiveChunker(TextChunker):
    """
    Recursive chunker that splits text hierarchically.

    Tries larger separators first, then falls back to smaller ones
    if chunks are still too large.
    """

    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk document recursively."""
        text = document.content
        raw_chunks = self._recursive_split(text, self.config.separators)

        # Merge small chunks and split large ones
        processed_chunks = self._process_chunks(raw_chunks)

        # Track positions
        positioned_chunks = self._add_positions(document.content, processed_chunks)

        # Create DocumentChunk objects
        total_chunks = len(positioned_chunks)
        return [
            self._create_chunk(
                content=content,
                document=document,
                chunk_index=i,
                total_chunks=total_chunks,
                start_char=start,
                end_char=end,
            )
            for i, (content, start, end) in enumerate(positioned_chunks)
        ]

    def _recursive_split(
        self,
        text: str,
        separators: List[str],
    ) -> List[str]:
        """Recursively split text using separators."""
        if not text.strip():
            return []

        # If text is small enough, return as-is
        if len(text) <= self.config.chunk_size:
            return [text]

        # Try each separator in order
        for i, separator in enumerate(separators):
            if separator in text:
                splits = text.split(separator)
                splits = [s for s in splits if s.strip()]

                if len(splits) > 1:
                    # Recursively process each split
                    result = []
                    remaining_separators = separators[i:]

                    for split in splits:
                        if len(split) > self.config.chunk_size:
                            result.extend(
                                self._recursive_split(split, remaining_separators)
                            )
                        else:
                            result.append(split)

                    return result

        # No suitable separator found, do hard split
        return self._hard_split(text)

    def _hard_split(self, text: str) -> List[str]:
        """Hard split when no suitable separator found."""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.config.chunk_size, len(text))

            # Try to split at word boundary
            if end < len(text):
                space_pos = text.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos

            chunks.append(text[start:end].strip())
            start = end

        return chunks

    def _process_chunks(self, chunks: List[str]) -> List[str]:
        """Process chunks: merge small, ensure overlap."""
        processed = []
        buffer = ""

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            if len(buffer) + len(chunk) <= self.config.chunk_size:
                buffer = f"{buffer} {chunk}".strip() if buffer else chunk
            else:
                if buffer:
                    processed.append(buffer)
                buffer = chunk

        if buffer:
            processed.append(buffer)

        # Add overlap
        if self.config.chunk_overlap > 0:
            processed = self._add_overlap(processed)

        return processed

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between chunks."""
        if len(chunks) <= 1:
            return chunks

        result = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]

            # Get overlap from end of previous chunk
            overlap_text = prev_chunk[-self.config.chunk_overlap:]

            # Find a good breaking point
            space_pos = overlap_text.find(' ')
            if space_pos > 0:
                overlap_text = overlap_text[space_pos:].strip()

            result.append(f"{overlap_text} {current_chunk}".strip())

        return result

    def _add_positions(
        self,
        original_text: str,
        chunks: List[str],
    ) -> List[Tuple[str, int, int]]:
        """Add character positions to chunks."""
        positioned = []
        search_start = 0

        for chunk in chunks:
            # Find chunk in original text
            # Use first significant portion to find position
            search_text = chunk[:min(100, len(chunk))].replace('\n', ' ').strip()

            pos = original_text.find(search_text, search_start)
            if pos == -1:
                pos = search_start  # Fallback

            end_pos = pos + len(chunk)
            positioned.append((chunk, pos, end_pos))
            search_start = pos + 1

        return positioned


class SemanticChunker(TextChunker):
    """
    Semantic chunker that groups text by topic similarity.

    Requires an embedding function to compute similarities.
    """

    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        similarity_threshold: float = 0.5,
    ):
        super().__init__(config)
        self._embedding_fn = embedding_fn
        self._similarity_threshold = similarity_threshold

    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk document semantically."""
        if not self._embedding_fn:
            # Fall back to recursive chunking
            logger.warning("No embedding function provided, falling back to recursive chunking")
            return RecursiveChunker(self.config).chunk(document)

        text = document.content

        # First, split into sentences
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return [self._create_chunk(
                content=text,
                document=document,
                chunk_index=0,
                total_chunks=1,
                start_char=0,
                end_char=len(text),
            )]

        # Get embeddings for each sentence
        embeddings = [self._embedding_fn(s) for s in sentences]

        # Group sentences by similarity
        groups = self._group_by_similarity(sentences, embeddings)

        # Create chunks from groups
        positioned_chunks = self._groups_to_chunks(text, groups)

        total_chunks = len(positioned_chunks)
        return [
            self._create_chunk(
                content=content,
                document=document,
                chunk_index=i,
                total_chunks=total_chunks,
                start_char=start,
                end_char=end,
            )
            for i, (content, start, end) in enumerate(positioned_chunks)
        ]

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """Calculate cosine similarity between vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _group_by_similarity(
        self,
        sentences: List[str],
        embeddings: List[List[float]],
    ) -> List[List[str]]:
        """Group sentences by semantic similarity."""
        groups = []
        current_group = [sentences[0]]
        current_embedding = embeddings[0]

        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(current_embedding, embeddings[i])

            # Check if should start new group
            group_size = sum(len(s) for s in current_group)

            if (similarity < self._similarity_threshold or
                group_size >= self.config.chunk_size):
                groups.append(current_group)
                current_group = [sentences[i]]
                current_embedding = embeddings[i]
            else:
                current_group.append(sentences[i])
                # Update running average embedding
                current_embedding = [
                    (a + b) / 2 for a, b in zip(current_embedding, embeddings[i])
                ]

        if current_group:
            groups.append(current_group)

        return groups

    def _groups_to_chunks(
        self,
        original_text: str,
        groups: List[List[str]],
    ) -> List[Tuple[str, int, int]]:
        """Convert sentence groups to positioned chunks."""
        chunks = []
        search_start = 0

        for group in groups:
            content = ' '.join(group)

            # Find position
            first_sentence = group[0][:50] if group else ""
            pos = original_text.find(first_sentence, search_start)
            if pos == -1:
                pos = search_start

            end_pos = pos + len(content)
            chunks.append((content, pos, end_pos))
            search_start = pos + 1

        return chunks


class SlidingWindowChunker(TextChunker):
    """Sliding window chunker with configurable stride."""

    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        stride: Optional[int] = None,
    ):
        super().__init__(config)
        self._stride = stride or (config.chunk_size - config.chunk_overlap if config else 800)

    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Chunk document using sliding window."""
        text = document.content
        chunks = []

        window_size = self.config.chunk_size
        stride = self._stride

        start = 0
        while start < len(text):
            end = min(start + window_size, len(text))
            chunk_text = text[start:end]

            # Try to end at word boundary
            if end < len(text) and not text[end].isspace():
                last_space = chunk_text.rfind(' ')
                if last_space > window_size // 2:
                    chunk_text = chunk_text[:last_space]
                    end = start + last_space

            chunks.append((chunk_text.strip(), start, end))

            start += stride
            if start + stride > len(text):
                break

        total_chunks = len(chunks)
        return [
            self._create_chunk(
                content=content,
                document=document,
                chunk_index=i,
                total_chunks=total_chunks,
                start_char=start_pos,
                end_char=end_pos,
            )
            for i, (content, start_pos, end_pos) in enumerate(chunks)
        ]


# Factory for creating chunkers
def create_chunker(
    strategy: ChunkingStrategy,
    config: Optional[ChunkingConfig] = None,
    **kwargs,
) -> TextChunker:
    """Create a chunker for the specified strategy."""
    config = config or ChunkingConfig(strategy=strategy)

    chunkers = {
        ChunkingStrategy.FIXED: FixedChunker,
        ChunkingStrategy.SENTENCE: SentenceChunker,
        ChunkingStrategy.RECURSIVE: RecursiveChunker,
        ChunkingStrategy.SEMANTIC: SemanticChunker,
        ChunkingStrategy.SLIDING_WINDOW: SlidingWindowChunker,
    }

    chunker_class = chunkers.get(strategy, RecursiveChunker)
    return chunker_class(config, **kwargs)


__all__ = [
    "ChunkingStrategy",
    "ChunkingConfig",
    "TextChunker",
    "FixedChunker",
    "SentenceChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SlidingWindowChunker",
    "create_chunker",
]
