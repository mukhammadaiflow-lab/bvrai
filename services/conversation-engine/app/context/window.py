"""Context window management for LLM interactions."""

import structlog
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from app.context.manager import Message, MessageRole


logger = structlog.get_logger()


class TruncationStrategy(str, Enum):
    """Strategy for truncating context when it exceeds limits."""
    SLIDING = "sliding"  # Remove oldest messages
    SUMMARIZE = "summarize"  # Summarize older messages
    IMPORTANT = "important"  # Keep important messages, remove others
    HYBRID = "hybrid"  # Combination of strategies


@dataclass
class WindowConfig:
    """Configuration for context window."""
    max_tokens: int = 4000
    max_messages: int = 50
    reserved_tokens: int = 1000  # Reserved for response
    system_tokens: int = 500  # Reserved for system prompt
    strategy: TruncationStrategy = TruncationStrategy.SLIDING

    # For important strategy
    keep_first_n: int = 2  # Keep first N user messages
    keep_last_n: int = 10  # Keep last N messages

    # For summarize strategy
    summarize_threshold: int = 20  # Summarize when this many messages
    summary_max_tokens: int = 200


class ContextWindow:
    """
    Manages context window for LLM interactions.

    Handles:
    - Token counting and limiting
    - Message selection strategies
    - Context compression
    - Important message preservation
    """

    def __init__(self, config: Optional[WindowConfig] = None):
        self.config = config or WindowConfig()

    def fit_messages(
        self,
        messages: List[Message],
        system_prompt: str = "",
        available_tokens: Optional[int] = None,
    ) -> List[Message]:
        """
        Fit messages into available token budget.

        Args:
            messages: All conversation messages
            system_prompt: System prompt (for token calculation)
            available_tokens: Override available tokens

        Returns:
            Messages that fit in context window
        """
        if not messages:
            return []

        # Calculate available tokens
        total_available = available_tokens or self.config.max_tokens
        system_tokens = self._estimate_tokens(system_prompt)
        available = total_available - system_tokens - self.config.reserved_tokens

        if available <= 0:
            logger.warning("no_tokens_available", system_tokens=system_tokens)
            return messages[-3:]  # Keep at least last 3

        # Apply strategy
        if self.config.strategy == TruncationStrategy.SLIDING:
            return self._sliding_window(messages, available)
        elif self.config.strategy == TruncationStrategy.IMPORTANT:
            return self._important_window(messages, available)
        elif self.config.strategy == TruncationStrategy.HYBRID:
            return self._hybrid_window(messages, available)
        else:
            return self._sliding_window(messages, available)

    def _sliding_window(
        self,
        messages: List[Message],
        available_tokens: int,
    ) -> List[Message]:
        """Simple sliding window - keep most recent messages."""
        result = []
        total_tokens = 0

        # Work backwards from most recent
        for msg in reversed(messages):
            msg_tokens = msg.token_estimate()
            if total_tokens + msg_tokens > available_tokens:
                break
            result.insert(0, msg)
            total_tokens += msg_tokens

        return result

    def _important_window(
        self,
        messages: List[Message],
        available_tokens: int,
    ) -> List[Message]:
        """Keep important messages (first few + recent)."""
        if len(messages) <= self.config.keep_first_n + self.config.keep_last_n:
            return self._sliding_window(messages, available_tokens)

        # Get first N user messages
        first_messages = []
        user_count = 0
        for msg in messages:
            first_messages.append(msg)
            if msg.role == MessageRole.USER:
                user_count += 1
                if user_count >= self.config.keep_first_n:
                    break

        # Get last N messages
        last_messages = messages[-self.config.keep_last_n:]

        # Combine without duplicates
        seen_times = set()
        combined = []

        for msg in first_messages + last_messages:
            msg_id = (msg.timestamp, msg.content[:50])
            if msg_id not in seen_times:
                combined.append(msg)
                seen_times.add(msg_id)

        # Check token budget
        total_tokens = sum(m.token_estimate() for m in combined)
        if total_tokens <= available_tokens:
            return combined

        # If still too many, fall back to sliding
        return self._sliding_window(combined, available_tokens)

    def _hybrid_window(
        self,
        messages: List[Message],
        available_tokens: int,
    ) -> List[Message]:
        """
        Hybrid approach:
        1. Keep first user message (for context)
        2. Keep all function calls/results from current turn
        3. Fill remaining with recent messages
        """
        if len(messages) <= 5:
            return self._sliding_window(messages, available_tokens)

        must_include = []
        remaining = []

        # First user message
        for msg in messages:
            if msg.role == MessageRole.USER:
                must_include.append(msg)
                break

        # Recent function calls/results (last 5)
        for msg in messages[-5:]:
            if msg.role in (MessageRole.FUNCTION, MessageRole.TOOL):
                if msg not in must_include:
                    must_include.append(msg)

        # Calculate tokens used by must-include
        must_tokens = sum(m.token_estimate() for m in must_include)
        remaining_tokens = available_tokens - must_tokens

        # Fill with recent messages
        for msg in reversed(messages):
            if msg in must_include:
                continue
            msg_tokens = msg.token_estimate()
            if remaining_tokens - msg_tokens < 0:
                break
            remaining.insert(0, msg)
            remaining_tokens -= msg_tokens

        # Combine and sort by timestamp
        all_messages = must_include + remaining
        all_messages.sort(key=lambda m: m.timestamp)

        return all_messages

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if not text:
            return 0
        # Rough approximation: 4 chars per token
        return len(text) // 4 + 1

    def get_window_stats(
        self,
        messages: List[Message],
        system_prompt: str = "",
    ) -> Dict[str, Any]:
        """Get statistics about current context window."""
        total_tokens = sum(m.token_estimate() for m in messages)
        system_tokens = self._estimate_tokens(system_prompt)

        return {
            "message_count": len(messages),
            "total_tokens": total_tokens,
            "system_tokens": system_tokens,
            "available_tokens": self.config.max_tokens - total_tokens - system_tokens,
            "utilization_percent": round(
                (total_tokens + system_tokens) / self.config.max_tokens * 100, 2
            ),
        }

    def should_summarize(self, messages: List[Message]) -> bool:
        """Check if messages should be summarized."""
        return len(messages) >= self.config.summarize_threshold

    def get_messages_to_summarize(
        self,
        messages: List[Message],
    ) -> List[Message]:
        """Get messages that should be summarized (older ones)."""
        if len(messages) < self.config.summarize_threshold:
            return []

        # Keep last N messages, summarize the rest
        cutoff = len(messages) - self.config.keep_last_n
        return messages[:cutoff]


class TokenCounter:
    """
    Token counter for different LLM providers.

    Uses tiktoken for OpenAI, approximations for others.
    """

    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self._encoding = None

        # Try to load tiktoken
        try:
            import tiktoken
            self._encoding = tiktoken.encoding_for_model(model)
        except Exception:
            pass  # Fall back to approximation

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0

        if self._encoding:
            return len(self._encoding.encode(text))

        # Approximation for other models
        return len(text) // 4 + 1

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens in message list (OpenAI format)."""
        total = 0

        for msg in messages:
            # Message overhead
            total += 4  # role, content structure

            # Content
            content = msg.get("content", "")
            if content:
                total += self.count(content)

            # Name if present
            if "name" in msg:
                total += self.count(msg["name"]) + 1

            # Function call
            if "function_call" in msg:
                total += self.count(str(msg["function_call"]))

        # Reply priming
        total += 2

        return total

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        if self.count(text) <= max_tokens:
            return text

        # Binary search for cutoff point
        low, high = 0, len(text)

        while low < high:
            mid = (low + high + 1) // 2
            if self.count(text[:mid]) <= max_tokens:
                low = mid
            else:
                high = mid - 1

        return text[:low] + "..."
