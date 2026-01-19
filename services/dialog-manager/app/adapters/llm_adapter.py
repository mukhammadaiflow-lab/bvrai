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


class OpenAIAdapter(LLMAdapter):
    """
    OpenAI LLM adapter using the openai Python client.

    Supports GPT-4, GPT-4-turbo, GPT-3.5-turbo, and other OpenAI models.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        organization: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.organization = organization
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    organization=self.organization,
                    base_url=self.base_url,
                )
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
        return self._client

    async def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate a completion using OpenAI."""
        client = self._get_client()

        # Convert messages to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                max_tokens=max_tokens or 1024,
                temperature=temperature if temperature is not None else 0.7,
                stop=stop_sequences,
            )

            choice = response.choices[0]
            usage = response.usage

            logger.debug(
                "openai_completion",
                model=self.model,
                tokens=usage.total_tokens if usage else 0,
                finish_reason=choice.finish_reason,
            )

            return LLMResponse(
                text=choice.message.content or "",
                finish_reason=choice.finish_reason or "stop",
                usage={
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                },
                model=response.model,
            )

        except Exception as e:
            logger.error("openai_error", error=str(e))
            raise

    async def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            client = self._get_client()
            # Simple models list check
            await client.models.list()
            return True
        except Exception as e:
            logger.warning("openai_unavailable", error=str(e))
            return False


class AnthropicAdapter(LLMAdapter):
    """
    Anthropic Claude adapter using the anthropic Python client.

    Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku, and other models.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        """Get or create the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = AsyncAnthropic(**kwargs)
            except ImportError:
                raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
        return self._client

    async def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate a completion using Anthropic Claude."""
        client = self._get_client()

        # Extract system message if present
        system_message = None
        chat_messages = []
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                chat_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        try:
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens or 1024,
                "messages": chat_messages,
            }

            if system_message:
                kwargs["system"] = system_message
            if temperature is not None:
                kwargs["temperature"] = temperature
            if stop_sequences:
                kwargs["stop_sequences"] = stop_sequences

            response = await client.messages.create(**kwargs)

            # Extract text from response
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text

            logger.debug(
                "anthropic_completion",
                model=self.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                stop_reason=response.stop_reason,
            )

            return LLMResponse(
                text=text,
                finish_reason=response.stop_reason or "end_turn",
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                model=response.model,
            )

        except Exception as e:
            logger.error("anthropic_error", error=str(e))
            raise

    async def is_available(self) -> bool:
        """Check if Anthropic API is available."""
        try:
            client = self._get_client()
            # Try a minimal request
            await client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}],
            )
            return True
        except Exception as e:
            logger.warning("anthropic_unavailable", error=str(e))
            return False


class GroqAdapter(LLMAdapter):
    """
    Groq LLM adapter for ultra-fast inference.

    Supports Llama, Mixtral, and other models on Groq infrastructure.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-70b-versatile",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        """Get or create the Groq client."""
        if self._client is None:
            try:
                from groq import AsyncGroq
                self._client = AsyncGroq(api_key=self.api_key)
            except ImportError:
                raise RuntimeError("groq package not installed. Run: pip install groq")
        return self._client

    async def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate a completion using Groq."""
        client = self._get_client()

        groq_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=groq_messages,
                max_tokens=max_tokens or 1024,
                temperature=temperature if temperature is not None else 0.7,
                stop=stop_sequences,
            )

            choice = response.choices[0]
            usage = response.usage

            logger.debug(
                "groq_completion",
                model=self.model,
                tokens=usage.total_tokens if usage else 0,
            )

            return LLMResponse(
                text=choice.message.content or "",
                finish_reason=choice.finish_reason or "stop",
                usage={
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                },
                model=response.model,
            )

        except Exception as e:
            logger.error("groq_error", error=str(e))
            raise

    async def is_available(self) -> bool:
        """Check if Groq API is available."""
        try:
            client = self._get_client()
            await client.models.list()
            return True
        except Exception as e:
            logger.warning("groq_unavailable", error=str(e))
            return False


class GeminiAdapter(LLMAdapter):
    """
    Google Gemini adapter using the google-generativeai client.

    Supports Gemini Pro, Gemini Pro Vision, and other models.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-pro",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        """Get or create the Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                raise RuntimeError("google-generativeai not installed. Run: pip install google-generativeai")
        return self._client

    async def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Generate a completion using Google Gemini."""
        import asyncio

        client = self._get_client()

        # Convert messages to Gemini format
        # Gemini uses a different format - combine into conversation
        conversation_text = ""
        for msg in messages:
            if msg.role == "system":
                conversation_text += f"Instructions: {msg.content}\n\n"
            elif msg.role == "user":
                conversation_text += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                conversation_text += f"Assistant: {msg.content}\n"

        try:
            generation_config = {
                "max_output_tokens": max_tokens or 1024,
            }
            if temperature is not None:
                generation_config["temperature"] = temperature
            if stop_sequences:
                generation_config["stop_sequences"] = stop_sequences

            # Gemini's generate_content is sync, run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.generate_content(
                    conversation_text,
                    generation_config=generation_config,
                )
            )

            text = response.text if hasattr(response, 'text') else ""

            # Estimate token usage (Gemini doesn't always provide this)
            prompt_tokens = len(conversation_text.split()) * 1.3  # Rough estimate
            completion_tokens = len(text.split()) * 1.3

            logger.debug(
                "gemini_completion",
                model=self.model,
                text_length=len(text),
            )

            return LLMResponse(
                text=text,
                finish_reason="stop",
                usage={
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(prompt_tokens + completion_tokens),
                },
                model=self.model,
            )

        except Exception as e:
            logger.error("gemini_error", error=str(e))
            raise

    async def is_available(self) -> bool:
        """Check if Gemini API is available."""
        try:
            client = self._get_client()
            return client is not None
        except Exception as e:
            logger.warning("gemini_unavailable", error=str(e))
            return False


def create_llm_adapter(provider: str | None = None) -> LLMAdapter:
    """
    Factory function to create an LLM adapter.

    Args:
        provider: LLM provider name ("mock", "openai", "anthropic", "groq", "gemini")
                 If None, uses setting from config.

    Returns:
        Configured LLMAdapter instance

    Environment variables required per provider:
        - openai: OPENAI_API_KEY
        - anthropic: ANTHROPIC_API_KEY
        - groq: GROQ_API_KEY
        - gemini: GOOGLE_API_KEY
    """
    import os

    settings = get_settings()
    provider = provider or settings.llm_provider

    if provider == "mock":
        return MockLLMAdapter()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY") or getattr(settings, "openai_api_key", None)
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, falling back to mock")
            return MockLLMAdapter()
        model = getattr(settings, "llm_model", "gpt-4-turbo-preview")
        return OpenAIAdapter(api_key=api_key, model=model)

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY") or getattr(settings, "anthropic_api_key", None)
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set, falling back to mock")
            return MockLLMAdapter()
        model = getattr(settings, "llm_model", "claude-3-5-sonnet-20241022")
        return AnthropicAdapter(api_key=api_key, model=model)

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY") or getattr(settings, "groq_api_key", None)
        if not api_key:
            logger.warning("GROQ_API_KEY not set, falling back to mock")
            return MockLLMAdapter()
        model = getattr(settings, "llm_model", "llama-3.1-70b-versatile")
        return GroqAdapter(api_key=api_key, model=model)

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY") or getattr(settings, "google_api_key", None)
        if not api_key:
            logger.warning("GOOGLE_API_KEY not set, falling back to mock")
            return MockLLMAdapter()
        model = getattr(settings, "llm_model", "gemini-1.5-pro")
        return GeminiAdapter(api_key=api_key, model=model)

    logger.warning(f"Unknown LLM provider: {provider}, using mock")
    return MockLLMAdapter()
