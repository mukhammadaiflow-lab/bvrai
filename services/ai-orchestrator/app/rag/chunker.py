"""Text chunking for document processing."""

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import structlog

logger = structlog.get_logger()


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"


@dataclass
class Document:
    """A document to be chunked."""
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    source: Optional[str] = None

    @classmethod
    def from_text(cls, text: str, source: Optional[str] = None) -> "Document":
        """Create a document from text."""
        return cls(
            id=str(uuid.uuid4()),
            content=text,
            source=source,
        )


@dataclass
class Chunk:
    """A chunk of a document."""
    id: str
    content: str
    document_id: str
    index: int  # Position in document
    metadata: dict = field(default_factory=dict)

    # Character positions
    start_char: int = 0
    end_char: int = 0

    @property
    def length(self) -> int:
        """Get chunk length in characters."""
        return len(self.content)

    @property
    def word_count(self) -> int:
        """Get word count."""
        return len(self.content.split())


class TextChunker:
    """
    Text chunking for document processing.

    Splits documents into overlapping chunks for embedding and retrieval.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE,
        min_chunk_size: int = 100,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.min_chunk_size = min_chunk_size

        self.logger = logger.bind(
            component="chunker",
            strategy=strategy.value,
        )

    def chunk_document(self, document: Document) -> list[Chunk]:
        """
        Chunk a document using the configured strategy.

        Args:
            document: Document to chunk

        Returns:
            List of chunks
        """
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(document)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_by_sentence(document)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_by_paragraph(document)
        else:
            return self._chunk_fixed_size(document)

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk multiple documents."""
        chunks = []
        for doc in documents:
            chunks.extend(self.chunk_document(doc))
        return chunks

    def _chunk_fixed_size(self, document: Document) -> list[Chunk]:
        """Chunk by fixed character size with overlap."""
        text = document.content
        chunks = []

        start = 0
        index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to end at a word boundary
            if end < len(text):
                # Look for last space before end
                last_space = text.rfind(" ", start, end)
                if last_space > start + self.min_chunk_size:
                    end = last_space

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        id=f"{document.id}_{index}",
                        content=chunk_text,
                        document_id=document.id,
                        index=index,
                        start_char=start,
                        end_char=end,
                        metadata={
                            **document.metadata,
                            "source": document.source,
                        },
                    )
                )
                index += 1

            # Move start with overlap
            start = end - self.chunk_overlap
            if start <= 0 or end >= len(text):
                break

        return chunks

    def _chunk_by_sentence(self, document: Document) -> list[Chunk]:
        """Chunk by sentences, combining until chunk_size is reached."""
        text = document.content

        # Split into sentences
        sentences = self._split_sentences(text)

        chunks = []
        current_chunk = []
        current_size = 0
        current_start = 0
        index = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence exceeds chunk_size, save current chunk
            if current_size + sentence_len > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)

                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(
                        Chunk(
                            id=f"{document.id}_{index}",
                            content=chunk_text,
                            document_id=document.id,
                            index=index,
                            start_char=current_start,
                            end_char=current_start + len(chunk_text),
                            metadata={
                                **document.metadata,
                                "source": document.source,
                            },
                        )
                    )
                    index += 1

                # Keep overlap (last few sentences)
                overlap_sentences = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break

                current_chunk = overlap_sentences
                current_size = sum(len(s) for s in current_chunk)
                current_start = current_start + len(chunk_text) - overlap_size

            current_chunk.append(sentence)
            current_size += sentence_len

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        id=f"{document.id}_{index}",
                        content=chunk_text,
                        document_id=document.id,
                        index=index,
                        start_char=current_start,
                        end_char=current_start + len(chunk_text),
                        metadata={
                            **document.metadata,
                            "source": document.source,
                        },
                    )
                )

        return chunks

    def _chunk_by_paragraph(self, document: Document) -> list[Chunk]:
        """Chunk by paragraphs."""
        text = document.content

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", text)

        chunks = []
        current_chunk = []
        current_size = 0
        index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)

            # If paragraph alone is too big, use sentence chunking
            if para_len > self.chunk_size:
                # Save current chunk first
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(
                            Chunk(
                                id=f"{document.id}_{index}",
                                content=chunk_text,
                                document_id=document.id,
                                index=index,
                                metadata={
                                    **document.metadata,
                                    "source": document.source,
                                },
                            )
                        )
                        index += 1
                    current_chunk = []
                    current_size = 0

                # Chunk the large paragraph by sentences
                para_doc = Document(
                    id=f"{document.id}_para",
                    content=para,
                    metadata=document.metadata,
                    source=document.source,
                )
                para_chunks = self._chunk_by_sentence(para_doc)
                for pc in para_chunks:
                    pc.id = f"{document.id}_{index}"
                    pc.document_id = document.id
                    pc.index = index
                    chunks.append(pc)
                    index += 1

                continue

            # If adding this paragraph exceeds chunk_size, save current
            if current_size + para_len > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(
                        Chunk(
                            id=f"{document.id}_{index}",
                            content=chunk_text,
                            document_id=document.id,
                            index=index,
                            metadata={
                                **document.metadata,
                                "source": document.source,
                            },
                        )
                    )
                    index += 1
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += para_len

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        id=f"{document.id}_{index}",
                        content=chunk_text,
                        document_id=document.id,
                        index=index,
                        metadata={
                            **document.metadata,
                            "source": document.source,
                        },
                    )
                )

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        # Could use nltk or spacy for better results
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]


class RecursiveChunker(TextChunker):
    """
    Recursive text chunker that tries multiple separators.

    Similar to LangChain's RecursiveCharacterTextSplitter.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[list[str]] = None,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap)

        self.separators = separators or [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            " ",     # Words
            "",      # Characters
        ]

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Recursively chunk a document."""
        chunks = self._recursive_split(document.content, self.separators)

        result = []
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text) >= self.min_chunk_size:
                result.append(
                    Chunk(
                        id=f"{document.id}_{i}",
                        content=chunk_text,
                        document_id=document.id,
                        index=i,
                        metadata={
                            **document.metadata,
                            "source": document.source,
                        },
                    )
                )

        return result

    def _recursive_split(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """Recursively split text using separators."""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator:
            splits = text.split(separator)
        else:
            # Character-level split
            splits = list(text)

        chunks = []
        current_chunk = []
        current_size = 0

        for split in splits:
            split_size = len(split)

            if current_size + split_size > self.chunk_size:
                if current_chunk:
                    combined = separator.join(current_chunk)
                    if len(combined) > self.chunk_size and remaining_separators:
                        # Recursively split if still too large
                        sub_chunks = self._recursive_split(
                            combined,
                            remaining_separators,
                        )
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(combined)

                    current_chunk = []
                    current_size = 0

            current_chunk.append(split)
            current_size += split_size + len(separator)

        # Handle last chunk
        if current_chunk:
            combined = separator.join(current_chunk)
            if len(combined) > self.chunk_size and remaining_separators:
                sub_chunks = self._recursive_split(combined, remaining_separators)
                chunks.extend(sub_chunks)
            else:
                chunks.append(combined)

        return chunks
