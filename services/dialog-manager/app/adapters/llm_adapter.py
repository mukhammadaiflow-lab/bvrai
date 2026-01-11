"""
LLM Adapter - Abstract interface and implementations for LLM providers.

This module provides:
- Abstract LLMAdapter interface
- MockLLMAdapter for testing (rule-based responses)
- Comments for plugging in OpenAI/Anthropic/Claude

TODO: Implement real LLM adapters:
- OpenAIAdapter: Use openai Python client
- AnthropicAdapter: Use anthropic Python client
"""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import structlog

from app.config import get_settings

logger = structlog.get_logger()


@dataclass
class LLMResponse:
    """Response from LLM completion."""

    text: str
    finish_reason: str
    usage: dict[str, int]
    model: str


@dataclass
class LLMMessage:
    """Chat message for LLM."""

    role: str  # "system", "user", "assistant"
    content: str


class LLMAdapter(ABC):
    """
    Abstract base class for LLM adapters.

    Implementations should handle:
    - API authentication
    - Request/response formatting
    - Error handling and retries
    - Token counting
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            messages: List of chat messages (system, user, assistant)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            stop_sequences: Sequences that stop generation

        Returns:
            LLMResponse with generated text and metadata
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the LLM service is available."""
        pass


class MockLLMAdapter(LLMAdapter):
    """
    Mock LLM adapter for testing.

    Uses rule-based pattern matching to generate responses.
    This allows tests to run without API keys.
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.last_messages: list[LLMMessage] = []

        # Rule-based response patterns
        # Pattern: (regex, response_template, action_type, action_params)
        self.patterns: list[tuple[str, str, str | None, dict[str, Any] | None]] = [
            # Booking patterns
            (
                r"(book|schedule|appointment|reservation)",
                "I'd be happy to help you book an appointment. What service would you like, and when works best for you?",
                "initiate_booking",
                {"intent": "booking"},
            ),
            # FAQ/Help patterns
            (
                r"(hours|open|close|when)",
                "Our business hours are Monday through Friday, 9 AM to 6 PM, and Saturday 10 AM to 4 PM. We're closed on Sundays.",
                "lookup_faq",
                {"category": "hours"},
            ),
            # Pricing patterns
            (
                r"(price|cost|how much|fee)",
                "Our pricing varies by service. Would you like me to tell you about our rates for a specific service?",
                "lookup_faq",
                {"category": "pricing"},
            ),
            # Contact patterns
            (
                r"(contact|reach|phone|email|address)",
                "You can reach us by phone at (555) 123-4567 or email at contact@example.com. We're located at 123 Main Street.",
                "lookup_faq",
                {"category": "contact"},
            ),
            # Confirmation patterns
            (
                r"(yes|confirm|correct|that's right|sounds good)",
                "Great! I've noted that down. Is there anything else I can help you with?",
                "confirm_action",
                {},
            ),
            # Negation patterns
            (
                r"(no|cancel|never mind|don't)",
                "No problem at all. Is there something else I can help you with today?",
                "cancel_action",
                {},
            ),
            # Greeting patterns
            (
                r"(hello|hi|hey|good morning|good afternoon)",
                "Hello! Welcome! How can I assist you today?",
                None,
                None,
            ),
            # Thanks patterns
            (
                r"(thank|thanks|appreciate)",
                "You're welcome! Is there anything else I can help you with?",
                None,
                None,
            ),
            # Goodbye patterns
            (
                r"(bye|goodbye|see you|take care)",
                "Goodbye! Thank you for contacting us. Have a great day!",
                "end_conversation",
                {},
            ),
        ]

        # Default response when no pattern matches
        self.default_response = (
            "I understand you're asking about that. "
            "Could you please provide a bit more detail so I can help you better?"
        )

    async def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate a mock response based on pattern matching."""
        self.call_count += 1
        self.last_messages = messages

        # Get the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.role == "user":
                user_message = msg.content.lower()
                break

        # Match patterns
        response_text = self.default_response
        action_type: str | None = None
        action_params: dict[str, Any] | None = None

        for pattern, response, act_type, act_params in self.patterns:
            if re.search(pattern, user_message, re.IGNORECASE):
                response_text = response
                action_type = act_type
                action_params = act_params
                break

        # Format response with action if present
        if action_type:
            # Embed action in response for extraction
            response_text = f"{response_text}\n[ACTION:{action_type}:{action_params}]"

        logger.debug(
            "mock_llm_response",
            user_message=user_message[:50],
            response=response_text[:50],
            action_type=action_type,
        )

        return LLMResponse(
            text=response_text,
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            model="mock-llm",
        )

    async def is_available(self) -> bool:
        """Mock LLM is always available."""
        return True


# TODO: Implement OpenAI adapter
# class OpenAIAdapter(LLMAdapter):
#     """
#     OpenAI LLM adapter using the openai Python client.
#
#     Usage:
#         from openai import AsyncOpenAI
#         client = AsyncOpenAI(api_key=settings.openai_api_key)
#         response = await client.chat.completions.create(
#             model=settings.llm_model,
#             messages=[{"role": m.role, "content": m.content} for m in messages],
#             max_tokens=max_tokens,
#             temperature=temperature,
#         )
#     """
#     pass


# TODO: Implement Anthropic adapter
# class AnthropicAdapter(LLMAdapter):
#     """
#     Anthropic Claude adapter using the anthropic Python client.
#
#     Usage:
#         from anthropic import AsyncAnthropic
#         client = AsyncAnthropic(api_key=settings.anthropic_api_key)
#         response = await client.messages.create(
#             model=settings.llm_model,
#             max_tokens=max_tokens,
#             messages=[{"role": m.role, "content": m.content} for m in messages],
#         )
#     """
#     pass


def create_llm_adapter(provider: str | None = None) -> LLMAdapter:
    """
    Factory function to create an LLM adapter.

    Args:
        provider: LLM provider name ("mock", "openai", "anthropic")
                 If None, uses setting from config.

    Returns:
        Configured LLMAdapter instance
    """
    settings = get_settings()
    provider = provider or settings.llm_provider

    if provider == "mock":
        return MockLLMAdapter()

    # TODO: Implement real adapters
    # if provider == "openai":
    #     return OpenAIAdapter(settings.openai_api_key, settings.llm_model)
    # if provider == "anthropic":
    #     return AnthropicAdapter(settings.anthropic_api_key, settings.llm_model)

    logger.warning(f"Unknown LLM provider: {provider}, using mock")
    return MockLLMAdapter()
