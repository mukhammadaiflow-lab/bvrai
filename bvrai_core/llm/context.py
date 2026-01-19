"""
LLM Context Window Management

This module provides sophisticated context window management including:
- Automatic truncation strategies
- Message prioritization
- Token counting and optimization
- Conversation summarization
"""

import asyncio
import logging
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
    LLMMessage,
    LLMRole,
    ModelInfo,
    get_model_info,
)


# Try to import tiktoken for accurate token counting
_tiktoken_available = False
_tiktoken_encodings = {}
try:
    import tiktoken
    _tiktoken_available = True
except ImportError:
    pass


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Estimate token count for text.

    Uses tiktoken for accurate counting when available,
    falls back to a character-based heuristic.

    Args:
        text: The text to count tokens for
        model: The model to use for encoding (affects tokenization)

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    if _tiktoken_available:
        try:
            # Get or create encoding for this model
            if model not in _tiktoken_encodings:
                try:
                    _tiktoken_encodings[model] = tiktoken.encoding_for_model(model)
                except KeyError:
                    # Fall back to cl100k_base for unknown models
                    _tiktoken_encodings[model] = tiktoken.get_encoding("cl100k_base")

            encoding = _tiktoken_encodings[model]
            return len(encoding.encode(text))
        except Exception:
            pass  # Fall through to estimation

    # Improved character-based estimation
    # Average ratio varies by language and content type:
    # - English prose: ~4 chars/token
    # - Code: ~3 chars/token
    # - CJK languages: ~1.5 chars/token

    # Detect if text contains significant CJK characters
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff'
                    or '\u3040' <= c <= '\u309f'  # Hiragana
                    or '\u30a0' <= c <= '\u30ff')  # Katakana

    if cjk_count > len(text) * 0.3:
        # Significant CJK content
        return int(len(text) / 1.5) + 1

    # Check for code-like content (common programming symbols)
    code_indicators = sum(1 for c in text if c in '{}[]();=<>/')
    if code_indicators > len(text) * 0.05:
        # Likely code
        return int(len(text) / 3) + 1

    # Default: English prose
    return int(len(text) / 4) + 1


logger = logging.getLogger(__name__)


class MessagePriority(int, Enum):
    """Priority levels for messages."""
    CRITICAL = 100      # Never remove (system prompts, key context)
    HIGH = 75           # Remove only if necessary
    NORMAL = 50         # Standard messages
    LOW = 25            # Can be removed first
    EPHEMERAL = 0       # Remove immediately when needed


class ContextStrategy(str, Enum):
    """Strategies for managing context when it exceeds limits."""

    # Remove oldest messages first
    FIFO = "fifo"

    # Remove by priority, then by age
    PRIORITY = "priority"

    # Summarize old messages
    SUMMARIZE = "summarize"

    # Keep only recent + high priority
    SLIDING_WINDOW = "sliding_window"

    # Smart selection based on relevance
    SEMANTIC = "semantic"


@dataclass
class MessageMetadata:
    """Metadata for context management."""

    priority: MessagePriority = MessagePriority.NORMAL
    token_count: int = 0
    can_summarize: bool = True
    summary_group: Optional[str] = None  # Group messages for summarization
    relevance_score: float = 1.0
    timestamp: float = 0.0


@dataclass
class ContextWindow:
    """Represents the current context window."""

    messages: List[LLMMessage] = field(default_factory=list)
    metadata: List[MessageMetadata] = field(default_factory=list)

    # Token tracking
    total_tokens: int = 0
    max_tokens: int = 0
    reserved_output_tokens: int = 0

    # Model info
    model_id: Optional[str] = None

    def __post_init__(self):
        if self.model_id:
            model_info = get_model_info(self.model_id)
            if model_info:
                self.max_tokens = model_info.context_window
                self.reserved_output_tokens = min(
                    model_info.max_output_tokens,
                    self.max_tokens // 4,
                )

    @property
    def available_tokens(self) -> int:
        """Get available tokens for new content."""
        return self.max_tokens - self.total_tokens - self.reserved_output_tokens

    @property
    def is_full(self) -> bool:
        """Check if context is at capacity."""
        return self.available_tokens <= 0

    def add_message(
        self,
        message: LLMMessage,
        priority: MessagePriority = MessagePriority.NORMAL,
        token_count: Optional[int] = None,
    ) -> None:
        """Add a message to the context."""
        if token_count is None:
            token_count = self._estimate_tokens(message.content)

        metadata = MessageMetadata(
            priority=priority,
            token_count=token_count,
            timestamp=message.timestamp,
        )

        self.messages.append(message)
        self.metadata.append(metadata)
        self.total_tokens += token_count

    def remove_message(self, index: int) -> Optional[LLMMessage]:
        """Remove a message by index."""
        if 0 <= index < len(self.messages):
            message = self.messages.pop(index)
            metadata = self.metadata.pop(index)
            self.total_tokens -= metadata.token_count
            return message
        return None

    def get_messages(self) -> List[LLMMessage]:
        """Get all messages."""
        return self.messages.copy()

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()
        self.metadata.clear()
        self.total_tokens = 0

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return estimate_tokens(text)


class ContextManager:
    """
    Manages LLM context windows with various optimization strategies.
    """

    def __init__(
        self,
        model_id: str = "gpt-4o",
        strategy: ContextStrategy = ContextStrategy.SLIDING_WINDOW,
        max_context_tokens: Optional[int] = None,
        reserved_output_tokens: int = 4096,
        summarizer: Optional[Callable[[List[LLMMessage]], str]] = None,
    ):
        self.model_id = model_id
        self.strategy = strategy
        self.summarizer = summarizer

        # Get model info
        model_info = get_model_info(model_id)
        if model_info:
            self.max_context_tokens = max_context_tokens or model_info.context_window
        else:
            self.max_context_tokens = max_context_tokens or 8192

        self.reserved_output_tokens = reserved_output_tokens

        # Internal state
        self._window = ContextWindow(
            model_id=model_id,
            max_tokens=self.max_context_tokens,
            reserved_output_tokens=reserved_output_tokens,
        )

        # System messages (always kept)
        self._system_messages: List[Tuple[LLMMessage, int]] = []

        # Conversation history
        self._history: List[Tuple[LLMMessage, MessageMetadata]] = []

        # Summarized content
        self._summaries: List[str] = []

    @property
    def total_tokens(self) -> int:
        """Get total tokens in context."""
        system_tokens = sum(t for _, t in self._system_messages)
        history_tokens = sum(m.token_count for _, m in self._history)
        return system_tokens + history_tokens

    @property
    def available_tokens(self) -> int:
        """Get available tokens for new messages."""
        return self.max_context_tokens - self.total_tokens - self.reserved_output_tokens

    def set_system_prompt(
        self,
        content: str,
        priority: MessagePriority = MessagePriority.CRITICAL,
    ) -> None:
        """Set the system prompt."""
        message = LLMMessage.system(content)
        token_count = self._estimate_tokens(content)
        self._system_messages = [(message, token_count)]

    def add_system_message(
        self,
        content: str,
        priority: MessagePriority = MessagePriority.CRITICAL,
    ) -> None:
        """Add an additional system message."""
        message = LLMMessage.system(content)
        token_count = self._estimate_tokens(content)
        self._system_messages.append((message, token_count))

    def add_message(
        self,
        message: LLMMessage,
        priority: MessagePriority = MessagePriority.NORMAL,
        token_count: Optional[int] = None,
    ) -> None:
        """Add a message to the conversation history."""
        if token_count is None:
            token_count = self._estimate_tokens(message.content)

        metadata = MessageMetadata(
            priority=priority,
            token_count=token_count,
            timestamp=message.timestamp,
        )

        self._history.append((message, metadata))

        # Check if we need to truncate
        if self.available_tokens < 0:
            self._apply_strategy()

    def add_user_message(
        self,
        content: str,
        images: Optional[List[str]] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> None:
        """Add a user message."""
        message = LLMMessage.user(content, images=images)
        self.add_message(message, priority)

    def add_assistant_message(
        self,
        content: str,
        tool_calls: Optional[List[Dict]] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> None:
        """Add an assistant message."""
        message = LLMMessage.assistant(content, tool_calls=tool_calls)
        self.add_message(message, priority)

    def add_tool_result(
        self,
        content: str,
        tool_call_id: str,
        tool_name: str,
        priority: MessagePriority = MessagePriority.HIGH,
    ) -> None:
        """Add a tool result message."""
        message = LLMMessage.tool(content, tool_call_id, tool_name)
        self.add_message(message, priority)

    def get_messages(self) -> List[LLMMessage]:
        """Get all messages for LLM request."""
        messages = []

        # Add system messages first
        for msg, _ in self._system_messages:
            messages.append(msg)

        # Add summaries if any
        if self._summaries:
            summary_content = "\n\n".join([
                "Previous conversation summary:",
                *self._summaries,
            ])
            messages.append(LLMMessage.system(summary_content))

        # Add conversation history
        for msg, _ in self._history:
            messages.append(msg)

        return messages

    def _apply_strategy(self) -> None:
        """Apply the configured truncation strategy."""
        if self.strategy == ContextStrategy.FIFO:
            self._apply_fifo()
        elif self.strategy == ContextStrategy.PRIORITY:
            self._apply_priority()
        elif self.strategy == ContextStrategy.SLIDING_WINDOW:
            self._apply_sliding_window()
        elif self.strategy == ContextStrategy.SUMMARIZE:
            self._apply_summarize()
        elif self.strategy == ContextStrategy.SEMANTIC:
            self._apply_semantic()

    def _apply_fifo(self) -> None:
        """Remove oldest messages first."""
        while self.available_tokens < 0 and len(self._history) > 1:
            # Keep at least the most recent exchange
            removed = self._history.pop(0)
            logger.debug(f"FIFO removed message: {removed[0].content[:50]}...")

    def _apply_priority(self) -> None:
        """Remove by priority, then by age."""
        while self.available_tokens < 0 and len(self._history) > 1:
            # Find lowest priority message
            min_priority = min(m.priority for _, m in self._history)
            candidates = [
                (i, msg, meta) for i, (msg, meta) in enumerate(self._history)
                if meta.priority == min_priority
            ]

            if candidates:
                # Remove oldest among lowest priority
                idx = candidates[0][0]
                removed = self._history.pop(idx)
                logger.debug(
                    f"Priority removed message (priority={min_priority}): "
                    f"{removed[0].content[:50]}..."
                )

    def _apply_sliding_window(self) -> None:
        """Keep recent messages plus high-priority ones."""
        # First, separate high-priority messages
        high_priority = []
        normal_messages = []

        for msg, meta in self._history:
            if meta.priority >= MessagePriority.HIGH:
                high_priority.append((msg, meta))
            else:
                normal_messages.append((msg, meta))

        # Calculate tokens for high-priority
        high_priority_tokens = sum(m.token_count for _, m in high_priority)

        # Calculate remaining budget for normal messages
        system_tokens = sum(t for _, t in self._system_messages)
        remaining_tokens = (
            self.max_context_tokens
            - self.reserved_output_tokens
            - system_tokens
            - high_priority_tokens
        )

        # Keep recent messages within budget
        kept_normal = []
        current_tokens = 0

        for msg, meta in reversed(normal_messages):
            if current_tokens + meta.token_count <= remaining_tokens:
                kept_normal.insert(0, (msg, meta))
                current_tokens += meta.token_count
            else:
                logger.debug(f"Sliding window dropped: {msg.content[:50]}...")

        # Rebuild history
        self._history = high_priority + kept_normal

    def _apply_summarize(self) -> None:
        """Summarize old messages."""
        if not self.summarizer:
            # Fall back to sliding window
            self._apply_sliding_window()
            return

        # Find messages to summarize (keep recent ones)
        keep_count = min(10, len(self._history) // 2)
        to_summarize = self._history[:-keep_count]
        to_keep = self._history[-keep_count:]

        if to_summarize:
            # Generate summary
            messages_to_summarize = [msg for msg, _ in to_summarize]
            try:
                summary = self.summarizer(messages_to_summarize)
                self._summaries.append(summary)
                self._history = to_keep
                logger.debug(f"Summarized {len(to_summarize)} messages")
            except Exception as e:
                logger.error(f"Summarization failed: {e}")
                self._apply_sliding_window()

    def _apply_semantic(self) -> None:
        """Remove messages based on semantic relevance."""
        # This would require embeddings - fall back to priority for now
        self._apply_priority()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return estimate_tokens(text, self.model_id)

    def clear_history(self) -> None:
        """Clear conversation history but keep system prompts."""
        self._history.clear()
        self._summaries.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get context statistics."""
        return {
            "model_id": self.model_id,
            "strategy": self.strategy.value,
            "max_tokens": self.max_context_tokens,
            "reserved_output": self.reserved_output_tokens,
            "total_tokens": self.total_tokens,
            "available_tokens": self.available_tokens,
            "system_messages": len(self._system_messages),
            "history_messages": len(self._history),
            "summaries": len(self._summaries),
            "utilization": self.total_tokens / self.max_context_tokens if self.max_context_tokens else 0,
        }

    def export_history(self) -> List[Dict[str, Any]]:
        """Export conversation history."""
        return [
            {
                "message": msg.to_dict(),
                "metadata": {
                    "priority": meta.priority.value,
                    "token_count": meta.token_count,
                    "timestamp": meta.timestamp,
                },
            }
            for msg, meta in self._history
        ]

    def import_history(
        self,
        history: List[Dict[str, Any]],
        clear_existing: bool = True,
    ) -> None:
        """Import conversation history."""
        if clear_existing:
            self._history.clear()

        for item in history:
            msg_data = item["message"]
            meta_data = item.get("metadata", {})

            message = LLMMessage(
                role=LLMRole(msg_data["role"]),
                content=msg_data.get("content", ""),
                tool_calls=msg_data.get("tool_calls"),
                tool_call_id=msg_data.get("tool_call_id"),
                name=msg_data.get("name"),
            )

            metadata = MessageMetadata(
                priority=MessagePriority(meta_data.get("priority", MessagePriority.NORMAL)),
                token_count=meta_data.get("token_count", self._estimate_tokens(message.content)),
                timestamp=meta_data.get("timestamp", 0.0),
            )

            self._history.append((message, metadata))


def create_conversation_summarizer(
    llm_provider: "BaseLLMProvider",
    model: str = "gpt-4o-mini",
) -> Callable[[List[LLMMessage]], str]:
    """
    Create a conversation summarizer function.

    Args:
        llm_provider: LLM provider to use for summarization
        model: Model to use

    Returns:
        Summarizer function
    """
    import asyncio
    from .base import LLMConfig

    async def summarize_async(messages: List[LLMMessage]) -> str:
        # Build conversation text
        conversation = []
        for msg in messages:
            role = msg.role.value.upper()
            conversation.append(f"{role}: {msg.content}")

        prompt = f"""Summarize the following conversation concisely,
preserving key information, decisions, and context:

{chr(10).join(conversation)}

Summary:"""

        response = await llm_provider.complete(
            messages=[LLMMessage.user(prompt)],
            config=LLMConfig(
                model=model,
                max_tokens=500,
                temperature=0.3,
            ),
        )

        return response.content

    def summarize(messages: List[LLMMessage]) -> str:
        """
        Synchronous wrapper for summarize_async.

        Handles the case when called from an already running event loop.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running - safe to use asyncio.run
            return asyncio.run(summarize_async(messages))

        # Event loop is running - need to handle differently
        # Create a new thread to run the coroutine
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, summarize_async(messages))
            return future.result()

    return summarize


__all__ = [
    "MessagePriority",
    "ContextStrategy",
    "MessageMetadata",
    "ContextWindow",
    "ContextManager",
    "create_conversation_summarizer",
    "estimate_tokens",
]
