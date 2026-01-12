"""Context summarizer for long conversations."""

import asyncio
import structlog
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import httpx

from app.context.manager import Message, MessageRole


logger = structlog.get_logger()


@dataclass
class SummaryConfig:
    """Configuration for summarization."""
    max_summary_tokens: int = 200
    min_messages_to_summarize: int = 10
    llm_endpoint: str = "http://ai-orchestrator:8085"
    llm_model: str = "gpt-4o-mini"
    timeout_seconds: float = 30.0


class ContextSummarizer:
    """
    Summarizes conversation context for long conversations.

    Uses LLM to create concise summaries of older messages,
    preserving key information while reducing token count.
    """

    SUMMARIZE_PROMPT = """Summarize the following conversation excerpt concisely.
Focus on:
1. Key information collected (names, dates, preferences)
2. Main topics discussed
3. Any decisions or commitments made
4. Current state of the conversation

Keep the summary under {max_tokens} tokens. Use bullet points.

Conversation:
{conversation}

Summary:"""

    def __init__(self, config: Optional[SummaryConfig] = None):
        self.config = config or SummaryConfig()
        self._http_client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        """Start the summarizer."""
        self._http_client = httpx.AsyncClient(
            timeout=self.config.timeout_seconds
        )

    async def stop(self) -> None:
        """Stop the summarizer."""
        if self._http_client:
            await self._http_client.aclose()

    async def summarize(
        self,
        messages: List[Message],
        existing_summary: Optional[str] = None,
    ) -> str:
        """
        Summarize a list of messages.

        Args:
            messages: Messages to summarize
            existing_summary: Previous summary to incorporate

        Returns:
            Summary text
        """
        if len(messages) < self.config.min_messages_to_summarize:
            return existing_summary or ""

        # Format conversation
        conversation = self._format_messages(messages)

        # Include existing summary
        if existing_summary:
            conversation = f"Previous context: {existing_summary}\n\n{conversation}"

        # Build prompt
        prompt = self.SUMMARIZE_PROMPT.format(
            max_tokens=self.config.max_summary_tokens,
            conversation=conversation,
        )

        # Call LLM
        try:
            summary = await self._call_llm(prompt)
            logger.info(
                "context_summarized",
                message_count=len(messages),
                summary_length=len(summary),
            )
            return summary

        except Exception as e:
            logger.error("summarization_failed", error=str(e))
            # Fall back to simple extractive summary
            return self._extractive_summary(messages, existing_summary)

    async def incremental_summarize(
        self,
        new_messages: List[Message],
        existing_summary: str,
        max_new_messages: int = 20,
    ) -> str:
        """
        Incrementally update an existing summary with new messages.

        More efficient than re-summarizing everything.
        """
        if len(new_messages) <= 3:
            # Too few new messages, just append key info
            additions = self._extract_key_info(new_messages)
            if additions:
                return f"{existing_summary}\nRecent: {additions}"
            return existing_summary

        # Summarize new messages
        new_summary = await self.summarize(new_messages[-max_new_messages:])

        # Merge summaries
        merged = await self._merge_summaries(existing_summary, new_summary)

        return merged

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for summarization."""
        if not self._http_client:
            raise RuntimeError("Summarizer not started")

        response = await self._http_client.post(
            f"{self.config.llm_endpoint}/v1/generate",
            json={
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "model": self.config.llm_model,
                "max_tokens": self.config.max_summary_tokens,
                "temperature": 0.3,  # Low temperature for consistency
            },
        )
        response.raise_for_status()

        data = response.json()
        return data.get("content", "")

    async def _merge_summaries(
        self,
        existing: str,
        new: str,
    ) -> str:
        """Merge two summaries into one."""
        merge_prompt = f"""Merge these two conversation summaries into one concise summary.
Keep the most important information and remove redundancy.
Maximum {self.config.max_summary_tokens} tokens.

Earlier summary:
{existing}

Recent summary:
{new}

Merged summary:"""

        try:
            return await self._call_llm(merge_prompt)
        except Exception:
            # Fall back to simple concatenation
            return f"{existing}\n\n{new}"

    def _format_messages(self, messages: List[Message]) -> str:
        """Format messages for summarization prompt."""
        lines = []

        for msg in messages:
            role = msg.role.value.upper()
            content = msg.content[:500]  # Truncate long messages

            if msg.role == MessageRole.TOOL:
                role = f"FUNCTION[{msg.name}]"

            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _extractive_summary(
        self,
        messages: List[Message],
        existing_summary: Optional[str] = None,
    ) -> str:
        """
        Create extractive summary without LLM.

        Extracts key information patterns.
        """
        parts = []

        if existing_summary:
            parts.append(existing_summary)

        # Extract entities
        entities = self._extract_entities(messages)
        if entities:
            parts.append(f"Collected: {', '.join(entities)}")

        # Extract intents
        intents = self._extract_intents(messages)
        if intents:
            parts.append(f"Topics: {', '.join(intents[:3])}")

        # Recent exchange
        if len(messages) >= 2:
            last_user = None
            last_assistant = None

            for msg in reversed(messages):
                if msg.role == MessageRole.USER and not last_user:
                    last_user = msg.content[:100]
                elif msg.role == MessageRole.ASSISTANT and not last_assistant:
                    last_assistant = msg.content[:100]

                if last_user and last_assistant:
                    break

            if last_user:
                parts.append(f"Last asked: {last_user}")

        return " | ".join(parts)

    def _extract_entities(self, messages: List[Message]) -> List[str]:
        """Extract named entities from messages."""
        import re

        entities = set()

        patterns = [
            # Names (simple pattern)
            r"(?:my name is|I'm|I am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            # Email
            r"[\w.+-]+@[\w-]+\.[\w.-]+",
            # Phone
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            # Dates
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\b",
            # Times
            r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b",
        ]

        for msg in messages:
            if msg.role == MessageRole.USER:
                for pattern in patterns:
                    matches = re.findall(pattern, msg.content)
                    entities.update(str(m) for m in matches if m)

        return list(entities)[:10]  # Limit to 10

    def _extract_intents(self, messages: List[Message]) -> List[str]:
        """Extract likely intents from user messages."""
        intent_keywords = {
            "schedule": ["schedule", "appointment", "book", "reserve", "meeting"],
            "cancel": ["cancel", "remove", "delete", "stop"],
            "inquiry": ["what", "when", "where", "how", "why", "?"],
            "complaint": ["problem", "issue", "wrong", "broken", "not working"],
            "purchase": ["buy", "purchase", "order", "price", "cost"],
            "support": ["help", "support", "assist", "trouble"],
        }

        found_intents = set()

        for msg in messages:
            if msg.role != MessageRole.USER:
                continue

            content_lower = msg.content.lower()

            for intent, keywords in intent_keywords.items():
                if any(kw in content_lower for kw in keywords):
                    found_intents.add(intent)

        return list(found_intents)

    def _extract_key_info(self, messages: List[Message]) -> str:
        """Extract key info from recent messages."""
        key_parts = []

        for msg in messages:
            if msg.role == MessageRole.USER:
                # Keep short user messages
                if len(msg.content) < 50:
                    key_parts.append(f"User: {msg.content}")
            elif msg.role == MessageRole.TOOL:
                # Keep function results
                key_parts.append(f"Action: {msg.name}")

        return "; ".join(key_parts[:3])

    def estimate_compression_ratio(
        self,
        messages: List[Message],
        summary: str,
    ) -> float:
        """Estimate compression ratio achieved."""
        original_tokens = sum(m.token_estimate() for m in messages)
        summary_tokens = len(summary) // 4

        if original_tokens == 0:
            return 0.0

        return round(1 - (summary_tokens / original_tokens), 2)
