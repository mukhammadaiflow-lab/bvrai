"""
Conversation Context Management Module

This module provides context management for agent conversations,
including message history, windowing, and context compression.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
)

from .base import (
    Message,
    ConversationContext,
    AgentConfig,
    IntentCategory,
    SentimentLevel,
    KnowledgeChunk,
)


logger = logging.getLogger(__name__)


class TokenCounter(ABC):
    """Abstract base class for token counting."""

    @abstractmethod
    def count(self, text: str) -> int:
        """Count tokens in text."""
        pass

    @abstractmethod
    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens in messages."""
        pass


class SimpleTokenCounter(TokenCounter):
    """Simple token counter based on character/word estimation."""

    def __init__(self, chars_per_token: float = 4.0):
        """
        Initialize counter.

        Args:
            chars_per_token: Average characters per token
        """
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        """Count tokens in text."""
        return max(1, int(len(text) / self.chars_per_token))

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens in messages."""
        total = 0
        for msg in messages:
            # Add overhead for message structure
            total += 4
            content = msg.get("content", "")
            if content:
                total += self.count(content)
            # Function calls
            if "function_call" in msg:
                total += self.count(str(msg["function_call"]))
        return total


class TiktokenCounter(TokenCounter):
    """Token counter using tiktoken library."""

    def __init__(self, model: str = "gpt-4"):
        """
        Initialize counter.

        Args:
            model: Model name for encoding
        """
        self.model = model
        self._encoding = None

    def _get_encoding(self):
        """Get tiktoken encoding (lazy load)."""
        if self._encoding is None:
            try:
                import tiktoken
                self._encoding = tiktoken.encoding_for_model(self.model)
            except ImportError:
                logger.warning("tiktoken not installed, using simple counter")
                return None
            except KeyError:
                # Fall back to cl100k_base for unknown models
                import tiktoken
                self._encoding = tiktoken.get_encoding("cl100k_base")
        return self._encoding

    def count(self, text: str) -> int:
        """Count tokens in text."""
        encoding = self._get_encoding()
        if encoding is None:
            return max(1, len(text) // 4)
        return len(encoding.encode(text))

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """Count tokens in messages."""
        encoding = self._get_encoding()
        if encoding is None:
            return sum(
                4 + len(msg.get("content", "")) // 4
                for msg in messages
            )

        total = 0
        for msg in messages:
            # Message overhead
            total += 4
            for key, value in msg.items():
                if isinstance(value, str):
                    total += len(encoding.encode(value))
                elif isinstance(value, dict):
                    total += len(encoding.encode(str(value)))

        # Final overhead
        total += 2

        return total


@dataclass
class ContextWindow:
    """Manages the context window for LLM input."""

    max_tokens: int = 8000
    reserved_output_tokens: int = 1000
    system_prompt_tokens: int = 0

    # Messages
    system_message: Optional[Message] = None
    messages: List[Message] = field(default_factory=list)

    # Knowledge context
    knowledge_context: str = ""
    knowledge_tokens: int = 0

    # Token counter
    token_counter: TokenCounter = field(default_factory=SimpleTokenCounter)

    @property
    def available_tokens(self) -> int:
        """Get available tokens for messages."""
        return (
            self.max_tokens -
            self.reserved_output_tokens -
            self.system_prompt_tokens -
            self.knowledge_tokens
        )

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt."""
        self.system_message = Message(role="system", content=prompt)
        self.system_prompt_tokens = self.token_counter.count(prompt) + 4

    def add_message(self, message: Message) -> bool:
        """
        Add a message to the window.

        Returns True if message was added, False if would exceed limit.
        """
        msg_tokens = self.token_counter.count(message.content) + 4
        message.tokens = msg_tokens

        current_tokens = sum(m.tokens for m in self.messages)
        if current_tokens + msg_tokens > self.available_tokens:
            return False

        self.messages.append(message)
        return True

    def add_knowledge(self, chunks: List[KnowledgeChunk], max_tokens: int = 2000) -> None:
        """Add knowledge context."""
        context_parts = []
        tokens_used = 0

        for chunk in chunks:
            chunk_tokens = self.token_counter.count(chunk.content)
            if tokens_used + chunk_tokens > max_tokens:
                break

            context_parts.append(f"[{chunk.source}]: {chunk.content}")
            tokens_used += chunk_tokens

        self.knowledge_context = "\n\n".join(context_parts)
        self.knowledge_tokens = tokens_used

    def get_messages_for_llm(self) -> List[Dict[str, Any]]:
        """Get messages formatted for LLM."""
        result = []

        # Add system message with knowledge context
        if self.system_message:
            system_content = self.system_message.content
            if self.knowledge_context:
                system_content += f"\n\n## Relevant Information\n{self.knowledge_context}"
            result.append({"role": "system", "content": system_content})

        # Add conversation messages
        for msg in self.messages:
            msg_dict = {"role": msg.role, "content": msg.content}
            if msg.function_name:
                msg_dict["name"] = msg.function_name
            if msg.function_call_id:
                msg_dict["tool_call_id"] = msg.function_call_id
            result.append(msg_dict)

        return result

    def get_token_usage(self) -> Dict[str, int]:
        """Get token usage breakdown."""
        message_tokens = sum(m.tokens for m in self.messages)
        return {
            "max_tokens": self.max_tokens,
            "reserved_output": self.reserved_output_tokens,
            "system_prompt": self.system_prompt_tokens,
            "knowledge": self.knowledge_tokens,
            "messages": message_tokens,
            "available": self.available_tokens - message_tokens,
        }


class ContextCompressor:
    """Compresses conversation context when it exceeds limits."""

    def __init__(
        self,
        token_counter: TokenCounter,
        summarizer: Optional[Callable[[str], Coroutine[Any, Any, str]]] = None,
    ):
        """
        Initialize compressor.

        Args:
            token_counter: Token counter
            summarizer: Optional async function to summarize text
        """
        self.token_counter = token_counter
        self.summarizer = summarizer

    async def compress(
        self,
        messages: List[Message],
        target_tokens: int,
        preserve_recent: int = 4,
    ) -> List[Message]:
        """
        Compress messages to fit within target tokens.

        Args:
            messages: Messages to compress
            target_tokens: Target token count
            preserve_recent: Number of recent messages to preserve

        Returns:
            Compressed messages
        """
        if not messages:
            return []

        current_tokens = sum(m.tokens for m in messages)
        if current_tokens <= target_tokens:
            return messages

        # Split into older and recent messages
        if len(messages) <= preserve_recent:
            # Can't compress further
            return messages[-preserve_recent:]

        older_messages = messages[:-preserve_recent]
        recent_messages = messages[-preserve_recent:]

        # Try summarization if available
        if self.summarizer:
            summary = await self._summarize_messages(older_messages)
            summary_message = Message(
                role="assistant",
                content=f"[Previous conversation summary: {summary}]",
            )
            summary_message.tokens = self.token_counter.count(summary_message.content)

            result = [summary_message] + recent_messages
            result_tokens = sum(m.tokens for m in result)

            if result_tokens <= target_tokens:
                return result

        # Fall back to truncation
        return self._truncate_messages(messages, target_tokens)

    async def _summarize_messages(self, messages: List[Message]) -> str:
        """Summarize a list of messages."""
        conversation_text = "\n".join(
            f"{m.role}: {m.content}" for m in messages
        )

        if self.summarizer:
            return await self.summarizer(conversation_text)

        # Simple extractive summary
        return self._simple_summary(messages)

    def _simple_summary(self, messages: List[Message]) -> str:
        """Create a simple extractive summary."""
        key_points = []

        for msg in messages:
            content = msg.content
            # Extract first sentence as key point
            sentences = content.split(". ")
            if sentences:
                key_points.append(sentences[0])

        return ". ".join(key_points[:5])

    def _truncate_messages(
        self,
        messages: List[Message],
        target_tokens: int,
    ) -> List[Message]:
        """Truncate messages to fit target tokens."""
        result = []
        tokens_used = 0

        # Add messages from end (most recent first)
        for msg in reversed(messages):
            if tokens_used + msg.tokens > target_tokens:
                break
            result.insert(0, msg)
            tokens_used += msg.tokens

        return result


class ConversationMemory:
    """
    Manages conversation memory with short-term and long-term storage.

    Provides:
    - Message history management
    - Context window management
    - Memory compression
    - Semantic search over history
    """

    def __init__(
        self,
        max_short_term_messages: int = 50,
        context_window_tokens: int = 8000,
        token_counter: Optional[TokenCounter] = None,
    ):
        """
        Initialize memory.

        Args:
            max_short_term_messages: Max messages in short-term memory
            context_window_tokens: Context window size
            token_counter: Token counter to use
        """
        self.max_short_term_messages = max_short_term_messages
        self.token_counter = token_counter or TiktokenCounter()

        # Short-term memory (recent messages)
        self._short_term: List[Message] = []

        # Long-term memory (summarized/archived)
        self._long_term_summaries: List[str] = []

        # Context window
        self.context_window = ContextWindow(
            max_tokens=context_window_tokens,
            token_counter=self.token_counter,
        )

        # Compressor
        self._compressor = ContextCompressor(self.token_counter)

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt."""
        self.context_window.set_system_prompt(prompt)

    def add_user_message(self, content: str) -> Message:
        """Add a user message."""
        return self._add_message("user", content)

    def add_assistant_message(self, content: str) -> Message:
        """Add an assistant message."""
        return self._add_message("assistant", content)

    def add_function_result(
        self,
        content: str,
        function_name: str,
        call_id: str,
    ) -> Message:
        """Add a function result message."""
        msg = Message(
            role="function",
            content=content,
            function_name=function_name,
            function_call_id=call_id,
        )
        msg.tokens = self.token_counter.count(content) + 10
        self._short_term.append(msg)
        self._manage_memory()
        return msg

    def _add_message(self, role: str, content: str) -> Message:
        """Add a message to memory."""
        msg = Message(role=role, content=content)
        msg.tokens = self.token_counter.count(content) + 4
        self._short_term.append(msg)
        self._manage_memory()
        return msg

    def _manage_memory(self) -> None:
        """Manage memory limits."""
        if len(self._short_term) > self.max_short_term_messages:
            # Move older messages to long-term memory
            to_archive = self._short_term[:-self.max_short_term_messages // 2]
            self._short_term = self._short_term[-self.max_short_term_messages // 2:]

            # Create summary of archived messages
            summary = self._create_summary(to_archive)
            self._long_term_summaries.append(summary)

    def _create_summary(self, messages: List[Message]) -> str:
        """Create a summary of messages."""
        key_points = []
        for msg in messages:
            if msg.role == "user":
                key_points.append(f"User asked: {msg.content[:100]}")
            elif msg.role == "assistant":
                key_points.append(f"Agent responded: {msg.content[:100]}")
        return " | ".join(key_points[:5])

    def get_context_for_llm(
        self,
        knowledge_chunks: Optional[List[KnowledgeChunk]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get formatted context for LLM.

        Args:
            knowledge_chunks: Optional knowledge context

        Returns:
            List of messages for LLM
        """
        # Add knowledge if provided
        if knowledge_chunks:
            self.context_window.add_knowledge(knowledge_chunks)

        # Update context window with current messages
        self.context_window.messages = self._short_term.copy()

        return self.context_window.get_messages_for_llm()

    async def compress_if_needed(self, target_tokens: int) -> None:
        """Compress memory if needed."""
        current_tokens = sum(m.tokens for m in self._short_term)
        if current_tokens > target_tokens:
            self._short_term = await self._compressor.compress(
                self._short_term,
                target_tokens,
            )

    def get_last_messages(self, n: int = 5) -> List[Message]:
        """Get last n messages."""
        return self._short_term[-n:]

    def get_last_user_message(self) -> Optional[Message]:
        """Get last user message."""
        for msg in reversed(self._short_term):
            if msg.role == "user":
                return msg
        return None

    def get_message_count(self) -> int:
        """Get total message count."""
        return len(self._short_term)

    def get_token_count(self) -> int:
        """Get total token count."""
        return sum(m.tokens for m in self._short_term)

    def clear(self) -> None:
        """Clear all memory."""
        self._short_term.clear()
        self._long_term_summaries.clear()
        self.context_window.messages.clear()
        self.context_window.knowledge_context = ""
        self.context_window.knowledge_tokens = 0


class ConversationTracker:
    """
    Tracks conversation state and extracts insights.

    Provides:
    - Intent detection
    - Sentiment analysis
    - Topic tracking
    - Entity extraction
    """

    def __init__(self):
        """Initialize tracker."""
        self._intents: List[Tuple[datetime, IntentCategory, float]] = []
        self._sentiments: List[Tuple[datetime, SentimentLevel, float]] = []
        self._topics: List[str] = []
        self._entities: Dict[str, List[str]] = {}

    def detect_intent(self, text: str) -> Tuple[IntentCategory, float]:
        """
        Detect intent from text.

        Returns tuple of (intent, confidence).
        """
        text_lower = text.lower()

        # Simple keyword-based detection
        intent_keywords = {
            IntentCategory.GREETING: [
                "hello", "hi", "hey", "good morning", "good afternoon",
                "good evening", "how are you"
            ],
            IntentCategory.GOODBYE: [
                "goodbye", "bye", "thanks for your help", "have a good day",
                "talk to you later"
            ],
            IntentCategory.QUESTION: [
                "what", "when", "where", "why", "how", "can you tell me",
                "do you know", "is there", "are there"
            ],
            IntentCategory.REQUEST: [
                "i need", "i want", "can you", "could you", "please",
                "i would like", "help me"
            ],
            IntentCategory.COMPLAINT: [
                "problem", "issue", "frustrated", "disappointed", "unhappy",
                "not working", "broken", "terrible"
            ],
            IntentCategory.CONFIRMATION: [
                "yes", "yeah", "correct", "that's right", "exactly",
                "absolutely", "sure", "okay", "ok"
            ],
            IntentCategory.NEGATION: [
                "no", "nope", "not", "don't", "won't", "can't", "incorrect",
                "wrong", "that's not"
            ],
            IntentCategory.SCHEDULING: [
                "schedule", "appointment", "book", "reserve", "available",
                "time slot", "when can"
            ],
            IntentCategory.TRANSFER: [
                "speak to someone", "human", "representative", "manager",
                "supervisor", "transfer me"
            ],
        }

        best_intent = IntentCategory.UNKNOWN
        best_confidence = 0.0

        for intent, keywords in intent_keywords.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                confidence = min(1.0, matches / 3)
                if confidence > best_confidence:
                    best_intent = intent
                    best_confidence = confidence

        self._intents.append((datetime.utcnow(), best_intent, best_confidence))
        return best_intent, best_confidence

    def analyze_sentiment(self, text: str) -> Tuple[SentimentLevel, float]:
        """
        Analyze sentiment of text.

        Returns tuple of (sentiment, confidence).
        """
        text_lower = text.lower()

        positive_words = [
            "thank", "great", "wonderful", "excellent", "perfect", "happy",
            "pleased", "good", "love", "appreciate", "helpful"
        ]
        negative_words = [
            "terrible", "awful", "hate", "frustrated", "angry", "disappointed",
            "bad", "poor", "worst", "horrible", "annoyed", "upset"
        ]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        score = positive_count - negative_count

        if score >= 3:
            sentiment = SentimentLevel.VERY_POSITIVE
            confidence = 0.9
        elif score >= 1:
            sentiment = SentimentLevel.POSITIVE
            confidence = 0.7
        elif score <= -3:
            sentiment = SentimentLevel.VERY_NEGATIVE
            confidence = 0.9
        elif score <= -1:
            sentiment = SentimentLevel.NEGATIVE
            confidence = 0.7
        else:
            sentiment = SentimentLevel.NEUTRAL
            confidence = 0.5

        self._sentiments.append((datetime.utcnow(), sentiment, confidence))
        return sentiment, confidence

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text."""
        import re

        entities: Dict[str, List[str]] = {}

        # Phone numbers
        phone_pattern = r'\b(?:\+1)?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        phones = re.findall(phone_pattern, text)
        if phones:
            entities["phone"] = phones

        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            entities["email"] = emails

        # Dates (simple patterns)
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?,? \d{4}\b',
        ]
        for pattern in date_patterns:
            dates = re.findall(pattern, text, re.IGNORECASE)
            if dates:
                entities.setdefault("date", []).extend(dates)

        # Times
        time_pattern = r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b'
        times = re.findall(time_pattern, text)
        if times:
            entities["time"] = times

        # Money amounts
        money_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
        amounts = re.findall(money_pattern, text)
        if amounts:
            entities["money"] = amounts

        # Update stored entities
        for entity_type, values in entities.items():
            self._entities.setdefault(entity_type, []).extend(values)

        return entities

    def get_dominant_intent(self) -> Optional[IntentCategory]:
        """Get the most common intent."""
        if not self._intents:
            return None

        intent_counts: Dict[IntentCategory, int] = {}
        for _, intent, _ in self._intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        return max(intent_counts, key=intent_counts.get)

    def get_current_sentiment(self) -> SentimentLevel:
        """Get the current sentiment."""
        if not self._sentiments:
            return SentimentLevel.NEUTRAL

        # Weight recent sentiments more heavily
        recent = self._sentiments[-5:]
        sentiment_scores = {
            SentimentLevel.VERY_NEGATIVE: -2,
            SentimentLevel.NEGATIVE: -1,
            SentimentLevel.NEUTRAL: 0,
            SentimentLevel.POSITIVE: 1,
            SentimentLevel.VERY_POSITIVE: 2,
        }

        total_score = sum(
            sentiment_scores[sentiment] * confidence
            for _, sentiment, confidence in recent
        )
        avg_score = total_score / len(recent)

        if avg_score >= 1.5:
            return SentimentLevel.VERY_POSITIVE
        elif avg_score >= 0.5:
            return SentimentLevel.POSITIVE
        elif avg_score <= -1.5:
            return SentimentLevel.VERY_NEGATIVE
        elif avg_score <= -0.5:
            return SentimentLevel.NEGATIVE
        else:
            return SentimentLevel.NEUTRAL

    def get_all_entities(self) -> Dict[str, List[str]]:
        """Get all extracted entities."""
        return dict(self._entities)


__all__ = [
    "TokenCounter",
    "SimpleTokenCounter",
    "TiktokenCounter",
    "ContextWindow",
    "ContextCompressor",
    "ConversationMemory",
    "ConversationTracker",
]
