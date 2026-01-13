"""
LLM Streaming Utilities

This module provides utilities for handling streaming LLM responses
including buffering, event processing, and sentence detection.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from .base import StreamChunk, LLMRole


logger = logging.getLogger(__name__)


class StreamEventType(str, Enum):
    """Types of streaming events."""

    # Content events
    TOKEN = "token"                 # Individual token received
    WORD = "word"                   # Complete word formed
    SENTENCE = "sentence"           # Complete sentence formed
    PARAGRAPH = "paragraph"         # Complete paragraph formed

    # Tool events
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"

    # Stream lifecycle
    STREAM_START = "stream_start"
    STREAM_END = "stream_end"
    STREAM_ERROR = "stream_error"

    # Timing events
    FIRST_TOKEN = "first_token"
    LATENCY = "latency"


@dataclass
class StreamEvent:
    """Event emitted during streaming."""

    type: StreamEventType
    data: Any = None
    timestamp: float = field(default_factory=time.time)

    # For content events
    content: str = ""
    accumulated_content: str = ""

    # For tool events
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None

    # Metadata
    index: int = 0


@dataclass
class StreamBuffer:
    """Buffer for accumulating streamed content."""

    # Accumulated content
    content: str = ""
    tokens: List[str] = field(default_factory=list)
    words: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)

    # Current partial content
    current_word: str = ""
    current_sentence: str = ""

    # Tool call accumulation
    tool_calls: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Timing
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    start_time: float = field(default_factory=time.time)

    # Statistics
    token_count: int = 0

    def add_token(self, token: str) -> List[StreamEvent]:
        """
        Add a token and return any events triggered.
        """
        events = []
        now = time.time()

        # Track first token
        if self.first_token_time is None:
            self.first_token_time = now
            events.append(StreamEvent(
                type=StreamEventType.FIRST_TOKEN,
                data={"latency_ms": (now - self.start_time) * 1000},
                timestamp=now,
            ))

        self.last_token_time = now
        self.token_count += 1

        # Token event
        events.append(StreamEvent(
            type=StreamEventType.TOKEN,
            content=token,
            accumulated_content=self.content + token,
            timestamp=now,
        ))

        # Add to content
        self.content += token
        self.tokens.append(token)

        # Process for words
        word_events = self._process_words(token)
        events.extend(word_events)

        # Process for sentences
        sentence_events = self._process_sentences(token)
        events.extend(sentence_events)

        return events

    def _process_words(self, token: str) -> List[StreamEvent]:
        """Process token for word boundaries."""
        events = []

        self.current_word += token

        # Check for word boundaries (whitespace, punctuation)
        word_boundary_pattern = r'[\s\.,!?;:\)\]\}\'\"]+$'

        if re.search(word_boundary_pattern, self.current_word):
            # Extract the word
            word = self.current_word.strip()
            if word:
                self.words.append(word)
                events.append(StreamEvent(
                    type=StreamEventType.WORD,
                    content=word,
                    accumulated_content=self.content,
                ))
            self.current_word = ""

        return events

    def _process_sentences(self, token: str) -> List[StreamEvent]:
        """Process token for sentence boundaries."""
        events = []

        self.current_sentence += token

        # Check for sentence boundaries
        sentence_end_pattern = r'[.!?]+[\s\"\'\)]*$'

        if re.search(sentence_end_pattern, self.current_sentence):
            sentence = self.current_sentence.strip()
            if sentence:
                self.sentences.append(sentence)
                events.append(StreamEvent(
                    type=StreamEventType.SENTENCE,
                    content=sentence,
                    accumulated_content=self.content,
                ))
            self.current_sentence = ""

        return events

    def add_tool_call_delta(
        self,
        tool_call_id: str,
        name: Optional[str] = None,
        arguments_delta: str = "",
    ) -> Optional[StreamEvent]:
        """Add a tool call delta."""
        if tool_call_id not in self.tool_calls:
            self.tool_calls[tool_call_id] = {
                "id": tool_call_id,
                "name": "",
                "arguments": "",
            }
            return StreamEvent(
                type=StreamEventType.TOOL_CALL_START,
                tool_call_id=tool_call_id,
            )

        tc = self.tool_calls[tool_call_id]

        if name:
            tc["name"] = name

        if arguments_delta:
            tc["arguments"] += arguments_delta

        return StreamEvent(
            type=StreamEventType.TOOL_CALL_DELTA,
            tool_call_id=tool_call_id,
            tool_name=tc["name"],
            content=arguments_delta,
        )

    def finalize(self) -> List[StreamEvent]:
        """Finalize the buffer and return any remaining events."""
        events = []

        # Emit remaining word
        if self.current_word.strip():
            self.words.append(self.current_word.strip())
            events.append(StreamEvent(
                type=StreamEventType.WORD,
                content=self.current_word.strip(),
                accumulated_content=self.content,
            ))

        # Emit remaining sentence
        if self.current_sentence.strip():
            self.sentences.append(self.current_sentence.strip())
            events.append(StreamEvent(
                type=StreamEventType.SENTENCE,
                content=self.current_sentence.strip(),
                accumulated_content=self.content,
            ))

        # Emit tool call completions
        for tc_id, tc in self.tool_calls.items():
            events.append(StreamEvent(
                type=StreamEventType.TOOL_CALL_END,
                tool_call_id=tc_id,
                tool_name=tc.get("name"),
                content=tc.get("arguments", ""),
            ))

        # Stream end event
        events.append(StreamEvent(
            type=StreamEventType.STREAM_END,
            data={
                "total_content": self.content,
                "token_count": self.token_count,
                "duration_ms": (time.time() - self.start_time) * 1000,
                "tokens_per_second": (
                    self.token_count / (time.time() - self.start_time)
                    if time.time() > self.start_time else 0
                ),
            },
        ))

        return events

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        duration = time.time() - self.start_time
        return {
            "token_count": self.token_count,
            "word_count": len(self.words),
            "sentence_count": len(self.sentences),
            "content_length": len(self.content),
            "duration_seconds": duration,
            "tokens_per_second": self.token_count / duration if duration > 0 else 0,
            "time_to_first_token_ms": (
                (self.first_token_time - self.start_time) * 1000
                if self.first_token_time else None
            ),
            "tool_calls": len(self.tool_calls),
        }


class StreamProcessor:
    """
    Processes streaming LLM responses and emits events.
    """

    def __init__(
        self,
        on_token: Optional[Callable[[str], None]] = None,
        on_word: Optional[Callable[[str], None]] = None,
        on_sentence: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[Dict[str, Any]], None]] = None,
        min_sentence_length: int = 5,
    ):
        self.on_token = on_token
        self.on_word = on_word
        self.on_sentence = on_sentence
        self.on_tool_call = on_tool_call
        self.min_sentence_length = min_sentence_length

        self._buffer: Optional[StreamBuffer] = None

    async def process_stream(
        self,
        stream: AsyncIterator[StreamChunk],
    ) -> AsyncIterator[StreamEvent]:
        """
        Process a stream of chunks and yield events.
        """
        self._buffer = StreamBuffer()

        # Emit stream start
        yield StreamEvent(type=StreamEventType.STREAM_START)

        try:
            async for chunk in stream:
                # Process content
                if chunk.content:
                    events = self._buffer.add_token(chunk.content)
                    for event in events:
                        yield event
                        await self._emit_callback(event)

                # Process tool calls
                if chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        event = self._buffer.add_tool_call_delta(
                            tool_call_id=tc.get("id", ""),
                            name=tc.get("function", {}).get("name"),
                            arguments_delta=tc.get("function", {}).get("arguments", ""),
                        )
                        if event:
                            yield event

                # Check for stream end
                if chunk.is_finished:
                    break

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.STREAM_ERROR,
                data={"error": str(e)},
            )
            raise

        # Finalize
        final_events = self._buffer.finalize()
        for event in final_events:
            yield event
            await self._emit_callback(event)

    async def _emit_callback(self, event: StreamEvent) -> None:
        """Emit callback for event."""
        try:
            if event.type == StreamEventType.TOKEN and self.on_token:
                if asyncio.iscoroutinefunction(self.on_token):
                    await self.on_token(event.content)
                else:
                    self.on_token(event.content)

            elif event.type == StreamEventType.WORD and self.on_word:
                if asyncio.iscoroutinefunction(self.on_word):
                    await self.on_word(event.content)
                else:
                    self.on_word(event.content)

            elif event.type == StreamEventType.SENTENCE and self.on_sentence:
                if len(event.content) >= self.min_sentence_length:
                    if asyncio.iscoroutinefunction(self.on_sentence):
                        await self.on_sentence(event.content)
                    else:
                        self.on_sentence(event.content)

            elif event.type == StreamEventType.TOOL_CALL_END and self.on_tool_call:
                tc = self._buffer.tool_calls.get(event.tool_call_id, {})
                if asyncio.iscoroutinefunction(self.on_tool_call):
                    await self.on_tool_call(tc)
                else:
                    self.on_tool_call(tc)

        except Exception as e:
            logger.error(f"Callback error: {e}")

    def get_accumulated_content(self) -> str:
        """Get accumulated content."""
        return self._buffer.content if self._buffer else ""

    def get_sentences(self) -> List[str]:
        """Get accumulated sentences."""
        return self._buffer.sentences if self._buffer else []

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """Get accumulated tool calls."""
        return list(self._buffer.tool_calls.values()) if self._buffer else []

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self._buffer.get_stats() if self._buffer else {}


async def stream_to_text(
    stream: AsyncIterator[StreamChunk],
) -> Tuple[str, Dict[str, Any]]:
    """
    Consume a stream and return full text plus metadata.
    """
    buffer = StreamBuffer()

    async for chunk in stream:
        if chunk.content:
            buffer.add_token(chunk.content)

    return buffer.content, buffer.get_stats()


async def stream_sentences(
    stream: AsyncIterator[StreamChunk],
    min_length: int = 5,
) -> AsyncIterator[str]:
    """
    Stream sentences from LLM response.

    Useful for TTS integration where you want to start
    speaking as soon as a complete sentence is ready.
    """
    processor = StreamProcessor(min_sentence_length=min_length)

    async for event in processor.process_stream(stream):
        if event.type == StreamEventType.SENTENCE:
            if len(event.content) >= min_length:
                yield event.content


async def stream_words(
    stream: AsyncIterator[StreamChunk],
) -> AsyncIterator[str]:
    """
    Stream individual words from LLM response.
    """
    processor = StreamProcessor()

    async for event in processor.process_stream(stream):
        if event.type == StreamEventType.WORD:
            yield event.content


class SentenceBuffer:
    """
    Buffer that accumulates streaming text and yields complete sentences.

    Designed for TTS integration where you want natural speech boundaries.
    """

    # Sentence-ending patterns
    SENTENCE_END_PATTERNS = [
        r'[.!?]+[\s\"\'\)]*$',  # Standard endings
        r'\n\n+',               # Paragraph breaks
        r':\s*\n',              # Colon followed by newline
    ]

    def __init__(
        self,
        min_sentence_length: int = 10,
        max_buffer_length: int = 500,
        flush_timeout: float = 2.0,
    ):
        self.min_sentence_length = min_sentence_length
        self.max_buffer_length = max_buffer_length
        self.flush_timeout = flush_timeout

        self._buffer = ""
        self._last_flush_time = time.time()

    def add(self, text: str) -> List[str]:
        """
        Add text to buffer and return any complete sentences.
        """
        self._buffer += text
        sentences = []

        while True:
            sentence = self._try_extract_sentence()
            if sentence:
                sentences.append(sentence)
            else:
                break

        # Force flush if buffer too long or timeout
        if self._should_force_flush():
            if self._buffer.strip():
                sentences.append(self._buffer.strip())
                self._buffer = ""

        return sentences

    def _try_extract_sentence(self) -> Optional[str]:
        """Try to extract a complete sentence from buffer."""
        for pattern in self.SENTENCE_END_PATTERNS:
            match = re.search(pattern, self._buffer)
            if match:
                end_pos = match.end()
                sentence = self._buffer[:end_pos].strip()

                if len(sentence) >= self.min_sentence_length:
                    self._buffer = self._buffer[end_pos:].lstrip()
                    self._last_flush_time = time.time()
                    return sentence

        return None

    def _should_force_flush(self) -> bool:
        """Check if we should force flush the buffer."""
        if len(self._buffer) > self.max_buffer_length:
            return True

        if time.time() - self._last_flush_time > self.flush_timeout:
            return True

        return False

    def flush(self) -> Optional[str]:
        """Force flush remaining content."""
        if self._buffer.strip():
            content = self._buffer.strip()
            self._buffer = ""
            return content
        return None


__all__ = [
    "StreamEventType",
    "StreamEvent",
    "StreamBuffer",
    "StreamProcessor",
    "stream_to_text",
    "stream_sentences",
    "stream_words",
    "SentenceBuffer",
]
