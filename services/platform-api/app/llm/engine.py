"""
LLM Engine

Unified LLM service layer:
- Provider management
- Request routing
- Fallback handling
- Caching and rate limiting
"""

from typing import Optional, Dict, Any, List, AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import hashlib
import json
import logging
import time

from .providers import (
    BaseLLMProvider, LLMProvider, Message, MessageRole,
    CompletionConfig, CompletionResponse, StreamChunk,
    create_provider,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM service."""
    primary_provider: LLMProvider = LLMProvider.OPENAI
    fallback_providers: List[LLMProvider] = field(default_factory=list)

    # API keys
    api_keys: Dict[str, str] = field(default_factory=dict)
    provider_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Default model settings
    default_model: str = "gpt-4-turbo"
    default_temperature: float = 0.7
    default_max_tokens: int = 1000

    # Rate limiting
    rate_limit_rpm: int = 100  # Requests per minute
    rate_limit_tpm: int = 100000  # Tokens per minute

    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600

    # Retry settings
    max_retries: int = 3
    retry_delay_ms: int = 1000


class LLMCache:
    """Cache for LLM responses."""

    def __init__(self, ttl_seconds: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl_seconds

    def _make_key(
        self,
        messages: List[Message],
        config: CompletionConfig,
    ) -> str:
        """Generate cache key."""
        content = json.dumps({
            "messages": [m.to_dict() for m in messages],
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def get(
        self,
        messages: List[Message],
        config: CompletionConfig,
    ) -> Optional[CompletionResponse]:
        """Get cached response."""
        key = self._make_key(messages, config)
        entry = self._cache.get(key)

        if entry:
            age = (datetime.utcnow() - entry["timestamp"]).total_seconds()
            if age < self._ttl:
                return entry["response"]
            else:
                del self._cache[key]

        return None

    def set(
        self,
        messages: List[Message],
        config: CompletionConfig,
        response: CompletionResponse,
    ) -> None:
        """Cache response."""
        key = self._make_key(messages, config)
        self._cache[key] = {
            "response": response,
            "timestamp": datetime.utcnow(),
        }

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()


class RateLimiter:
    """Token and request rate limiter."""

    def __init__(
        self,
        rpm_limit: int = 100,
        tpm_limit: int = 100000,
    ):
        self._rpm_limit = rpm_limit
        self._tpm_limit = tpm_limit
        self._requests: List[float] = []
        self._tokens: List[tuple] = []  # (timestamp, token_count)

    async def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """Wait if rate limit would be exceeded."""
        now = time.time()
        window_start = now - 60

        # Clean old entries
        self._requests = [t for t in self._requests if t > window_start]
        self._tokens = [(t, c) for t, c in self._tokens if t > window_start]

        # Check RPM
        if len(self._requests) >= self._rpm_limit:
            wait_time = self._requests[0] - window_start
            if wait_time > 0:
                logger.debug(f"Rate limit (RPM): waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

        # Check TPM
        current_tokens = sum(c for _, c in self._tokens)
        if current_tokens + estimated_tokens > self._tpm_limit:
            wait_time = self._tokens[0][0] - window_start
            if wait_time > 0:
                logger.debug(f"Rate limit (TPM): waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

    def record_request(self, token_count: int = 0) -> None:
        """Record a request."""
        now = time.time()
        self._requests.append(now)
        if token_count > 0:
            self._tokens.append((now, token_count))


class LLMRouter:
    """
    Routes requests to appropriate providers.

    Features:
    - Provider selection
    - Fallback handling
    - Load balancing
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._providers: Dict[LLMProvider, BaseLLMProvider] = {}
        self._health: Dict[LLMProvider, Dict[str, Any]] = {}
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize configured providers."""
        # Primary provider
        self._create_provider(self.config.primary_provider)

        # Fallback providers
        for provider in self.config.fallback_providers:
            self._create_provider(provider)

    def _create_provider(self, provider: LLMProvider) -> None:
        """Create and register provider."""
        api_key = self.config.api_keys.get(provider.value, "")
        extra_config = self.config.provider_configs.get(provider.value, {})

        if api_key:
            self._providers[provider] = create_provider(
                provider,
                api_key,
                **extra_config,
            )
            self._health[provider] = {
                "healthy": True,
                "failures": 0,
                "last_success": None,
                "last_failure": None,
            }

    def get_provider(self, preferred: Optional[LLMProvider] = None) -> BaseLLMProvider:
        """Get healthy provider."""
        # Try preferred provider
        if preferred and preferred in self._providers:
            if self._is_healthy(preferred):
                return self._providers[preferred]

        # Try primary
        if self._is_healthy(self.config.primary_provider):
            return self._providers[self.config.primary_provider]

        # Try fallbacks
        for provider in self.config.fallback_providers:
            if self._is_healthy(provider):
                return self._providers[provider]

        # Return primary even if unhealthy
        return self._providers[self.config.primary_provider]

    def _is_healthy(self, provider: LLMProvider) -> bool:
        """Check if provider is healthy."""
        health = self._health.get(provider, {})
        return health.get("healthy", False)

    def record_success(self, provider: LLMProvider) -> None:
        """Record successful request."""
        if provider in self._health:
            self._health[provider]["healthy"] = True
            self._health[provider]["failures"] = 0
            self._health[provider]["last_success"] = datetime.utcnow()

    def record_failure(self, provider: LLMProvider) -> None:
        """Record failed request."""
        if provider in self._health:
            health = self._health[provider]
            health["failures"] += 1
            health["last_failure"] = datetime.utcnow()

            # Mark unhealthy after 3 consecutive failures
            if health["failures"] >= 3:
                health["healthy"] = False

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all providers."""
        return {
            provider.value: {
                "healthy": health["healthy"],
                "failures": health["failures"],
                "last_success": health["last_success"].isoformat() if health["last_success"] else None,
                "last_failure": health["last_failure"].isoformat() if health["last_failure"] else None,
            }
            for provider, health in self._health.items()
        }


class LLMEngine:
    """
    Main LLM engine.

    Features:
    - Unified API for all providers
    - Caching and rate limiting
    - Retry and fallback
    - Metrics and monitoring
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._router = LLMRouter(config)
        self._cache = LLMCache(ttl_seconds=config.cache_ttl_seconds) if config.enable_cache else None
        self._rate_limiter = RateLimiter(
            rpm_limit=config.rate_limit_rpm,
            tpm_limit=config.rate_limit_tpm,
        )
        self._metrics: Dict[str, Any] = {
            "total_requests": 0,
            "total_tokens": 0,
            "cache_hits": 0,
            "errors": 0,
        }

    async def complete(
        self,
        messages: List[Message],
        config: Optional[CompletionConfig] = None,
        provider: Optional[LLMProvider] = None,
        use_cache: bool = True,
    ) -> CompletionResponse:
        """Generate completion."""
        # Build config
        completion_config = config or CompletionConfig(
            model=self.config.default_model,
            temperature=self.config.default_temperature,
            max_tokens=self.config.default_max_tokens,
        )

        # Check cache
        if use_cache and self._cache:
            cached = self._cache.get(messages, completion_config)
            if cached:
                self._metrics["cache_hits"] += 1
                return cached

        # Rate limit
        estimated_tokens = sum(len(m.content) // 4 for m in messages)
        await self._rate_limiter.wait_if_needed(estimated_tokens)

        # Get provider
        llm_provider = self._router.get_provider(provider)

        # Execute with retry
        response = await self._execute_with_retry(
            llm_provider,
            messages,
            completion_config,
        )

        # Cache response
        if use_cache and self._cache and response:
            self._cache.set(messages, completion_config, response)

        # Update metrics
        self._metrics["total_requests"] += 1
        self._metrics["total_tokens"] += response.total_tokens
        self._rate_limiter.record_request(response.total_tokens)

        return response

    async def _execute_with_retry(
        self,
        provider: BaseLLMProvider,
        messages: List[Message],
        config: CompletionConfig,
    ) -> CompletionResponse:
        """Execute request with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = await provider.complete(messages, config)
                self._router.record_success(LLMProvider(provider.provider_name))
                return response

            except Exception as e:
                last_error = e
                logger.warning(f"LLM request failed (attempt {attempt + 1}): {e}")
                self._router.record_failure(LLMProvider(provider.provider_name))

                # Try fallback provider
                if attempt < self.config.max_retries - 1:
                    provider = self._router.get_provider()
                    await asyncio.sleep(self.config.retry_delay_ms / 1000)

        self._metrics["errors"] += 1
        raise last_error

    async def stream(
        self,
        messages: List[Message],
        config: Optional[CompletionConfig] = None,
        provider: Optional[LLMProvider] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream completion."""
        completion_config = config or CompletionConfig(
            model=self.config.default_model,
            temperature=self.config.default_temperature,
            max_tokens=self.config.default_max_tokens,
            stream=True,
        )

        # Rate limit
        estimated_tokens = sum(len(m.content) // 4 for m in messages)
        await self._rate_limiter.wait_if_needed(estimated_tokens)

        # Get provider
        llm_provider = self._router.get_provider(provider)

        # Stream
        try:
            async for chunk in llm_provider.stream(messages, completion_config):
                yield chunk

            self._router.record_success(LLMProvider(llm_provider.provider_name))
            self._metrics["total_requests"] += 1

        except Exception as e:
            logger.error(f"Stream error: {e}")
            self._router.record_failure(LLMProvider(llm_provider.provider_name))
            self._metrics["errors"] += 1
            raise

    async def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Message]] = None,
        **kwargs,
    ) -> str:
        """Simple chat interface."""
        messages = []

        if system_prompt:
            messages.append(Message(role=MessageRole.SYSTEM, content=system_prompt))

        if history:
            messages.extend(history)

        messages.append(Message(role=MessageRole.USER, content=user_message))

        config = CompletionConfig(**kwargs) if kwargs else None
        response = await self.complete(messages, config)

        return response.content

    async def function_call(
        self,
        user_message: str,
        functions: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute function calling."""
        messages = []

        if system_prompt:
            messages.append(Message(role=MessageRole.SYSTEM, content=system_prompt))

        messages.append(Message(role=MessageRole.USER, content=user_message))

        config = CompletionConfig(
            functions=functions,
            function_call="auto",
            **kwargs,
        )

        response = await self.complete(messages, config)

        if response.function_call:
            return {
                "name": response.function_call.get("name"),
                "arguments": json.loads(response.function_call.get("arguments", "{}")),
            }

        return {"content": response.content}

    def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics."""
        return {
            **self._metrics,
            "providers": self._router.get_health_status(),
        }

    def clear_cache(self) -> None:
        """Clear response cache."""
        if self._cache:
            self._cache.clear()


class ConversationManager:
    """
    Manages conversation state with LLM.

    Features:
    - Message history
    - Context window management
    - Token counting
    """

    def __init__(
        self,
        engine: LLMEngine,
        system_prompt: str = "",
        max_history_tokens: int = 4000,
    ):
        self._engine = engine
        self._system_prompt = system_prompt
        self._max_history_tokens = max_history_tokens
        self._messages: List[Message] = []
        self._total_tokens = 0

    def set_system_prompt(self, prompt: str) -> None:
        """Set system prompt."""
        self._system_prompt = prompt

    def add_user_message(self, content: str) -> None:
        """Add user message."""
        self._messages.append(Message(role=MessageRole.USER, content=content))
        self._trim_history()

    def add_assistant_message(self, content: str) -> None:
        """Add assistant message."""
        self._messages.append(Message(role=MessageRole.ASSISTANT, content=content))
        self._trim_history()

    def _trim_history(self) -> None:
        """Trim history to fit context window."""
        total_tokens = len(self._system_prompt) // 4

        # Calculate tokens in history
        history_tokens = []
        for msg in self._messages:
            tokens = len(msg.content) // 4
            history_tokens.append(tokens)
            total_tokens += tokens

        # Remove oldest messages if needed
        while total_tokens > self._max_history_tokens and len(self._messages) > 2:
            removed_tokens = history_tokens.pop(0)
            self._messages.pop(0)
            total_tokens -= removed_tokens

    def get_messages(self) -> List[Message]:
        """Get all messages including system prompt."""
        messages = []

        if self._system_prompt:
            messages.append(Message(role=MessageRole.SYSTEM, content=self._system_prompt))

        messages.extend(self._messages)
        return messages

    async def send(
        self,
        user_message: str,
        **kwargs,
    ) -> str:
        """Send message and get response."""
        self.add_user_message(user_message)

        messages = self.get_messages()
        response = await self._engine.complete(messages, **kwargs)

        self.add_assistant_message(response.content)
        self._total_tokens += response.total_tokens

        return response.content

    async def stream_send(
        self,
        user_message: str,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Send message and stream response."""
        self.add_user_message(user_message)

        messages = self.get_messages()
        full_response = ""

        async for chunk in self._engine.stream(messages, **kwargs):
            full_response = chunk.content
            yield chunk.delta

        self.add_assistant_message(full_response)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._messages.clear()
        self._total_tokens = 0

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return [
            {"role": m.role.value, "content": m.content}
            for m in self._messages
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        return {
            "message_count": len(self._messages),
            "total_tokens": self._total_tokens,
            "history_size": sum(len(m.content) for m in self._messages),
        }


# Singleton engine instance
_engine_instance: Optional[LLMEngine] = None


def get_llm_engine(config: Optional[LLMConfig] = None) -> LLMEngine:
    """Get singleton LLM engine."""
    global _engine_instance
    if _engine_instance is None:
        if config is None:
            config = LLMConfig()
        _engine_instance = LLMEngine(config)
    return _engine_instance


def create_llm_engine(
    primary_provider: LLMProvider = LLMProvider.OPENAI,
    api_key: str = "",
    **kwargs,
) -> LLMEngine:
    """Create LLM engine with configuration."""
    config = LLMConfig(
        primary_provider=primary_provider,
        api_keys={primary_provider.value: api_key},
        **kwargs,
    )
    return LLMEngine(config)
