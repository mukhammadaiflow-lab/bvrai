"""LLM manager for handling multiple providers."""

import structlog
from typing import Optional, Dict, List, Any, AsyncIterator

from app.llm.base import LLMAdapter, LLMMessage, LLMResponse, StreamChunk
from app.llm.openai import OpenAIAdapter
from app.llm.anthropic import AnthropicAdapter
from app.llm.groq import GroqAdapter


logger = structlog.get_logger()


class LLMManager:
    """
    Manages multiple LLM providers.

    Features:
    - Provider selection
    - Fallback support
    - Load balancing (future)
    - Cost tracking
    """

    def __init__(self):
        self._adapters: Dict[str, LLMAdapter] = {}
        self._default_provider: Optional[str] = None

    async def register_provider(
        self,
        name: str,
        adapter: LLMAdapter,
        set_default: bool = False,
    ) -> None:
        """
        Register an LLM provider.

        Args:
            name: Provider name
            adapter: LLM adapter instance
            set_default: Set as default provider
        """
        await adapter.connect()
        self._adapters[name] = adapter

        if set_default or self._default_provider is None:
            self._default_provider = name

        logger.info(
            "llm_provider_registered",
            name=name,
            model=adapter.model if hasattr(adapter, "model") else "unknown",
        )

    async def unregister_provider(self, name: str) -> bool:
        """Unregister a provider."""
        if name in self._adapters:
            await self._adapters[name].disconnect()
            del self._adapters[name]

            if self._default_provider == name:
                self._default_provider = next(iter(self._adapters), None)

            return True
        return False

    def get_adapter(self, provider: Optional[str] = None) -> LLMAdapter:
        """Get adapter for provider."""
        provider = provider or self._default_provider

        if not provider or provider not in self._adapters:
            raise ValueError(f"Unknown provider: {provider}")

        return self._adapters[provider]

    async def generate(
        self,
        messages: List[LLMMessage],
        provider: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        fallback_providers: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate response with optional fallback.

        Args:
            messages: Conversation messages
            provider: Provider to use
            tools: Available tools
            temperature: Sampling temperature
            max_tokens: Max tokens
            fallback_providers: Providers to try if primary fails
            **kwargs: Additional options

        Returns:
            LLM response
        """
        providers_to_try = [provider or self._default_provider]
        if fallback_providers:
            providers_to_try.extend(fallback_providers)

        last_error = None

        for prov in providers_to_try:
            if prov not in self._adapters:
                continue

            try:
                adapter = self._adapters[prov]
                response = await adapter.generate(
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                return response

            except Exception as e:
                logger.warning(
                    "llm_generate_failed",
                    provider=prov,
                    error=str(e),
                )
                last_error = e

        raise last_error or ValueError("No providers available")

    async def stream(
        self,
        messages: List[LLMMessage],
        provider: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response from LLM."""
        adapter = self.get_adapter(provider)

        async for chunk in adapter.stream(
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ):
            yield chunk

    def list_providers(self) -> List[Dict[str, Any]]:
        """List registered providers."""
        return [
            {
                "name": name,
                "model": adapter.model if hasattr(adapter, "model") else "unknown",
                "supports_streaming": adapter.supports_streaming,
                "supports_tools": adapter.supports_tools,
                "is_default": name == self._default_provider,
            }
            for name, adapter in self._adapters.items()
        ]

    def set_default(self, provider: str) -> bool:
        """Set default provider."""
        if provider in self._adapters:
            self._default_provider = provider
            return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "providers": len(self._adapters),
            "default_provider": self._default_provider,
            "provider_stats": {
                name: adapter.get_statistics()
                for name, adapter in self._adapters.items()
            },
        }

    async def shutdown(self) -> None:
        """Shutdown all providers."""
        for name, adapter in list(self._adapters.items()):
            try:
                await adapter.disconnect()
            except Exception as e:
                logger.error(
                    "llm_shutdown_error",
                    provider=name,
                    error=str(e),
                )

        self._adapters.clear()
        self._default_provider = None


# Global manager instance
llm_manager = LLMManager()


async def setup_providers(
    openai_key: Optional[str] = None,
    anthropic_key: Optional[str] = None,
    groq_key: Optional[str] = None,
    default_provider: str = "openai",
) -> LLMManager:
    """
    Setup LLM providers from API keys.

    Args:
        openai_key: OpenAI API key
        anthropic_key: Anthropic API key
        groq_key: Groq API key
        default_provider: Default provider name

    Returns:
        Configured LLM manager
    """
    if openai_key:
        adapter = OpenAIAdapter(api_key=openai_key)
        await llm_manager.register_provider(
            "openai",
            adapter,
            set_default=(default_provider == "openai"),
        )

    if anthropic_key:
        adapter = AnthropicAdapter(api_key=anthropic_key)
        await llm_manager.register_provider(
            "anthropic",
            adapter,
            set_default=(default_provider == "anthropic"),
        )

    if groq_key:
        adapter = GroqAdapter(api_key=groq_key)
        await llm_manager.register_provider(
            "groq",
            adapter,
            set_default=(default_provider == "groq"),
        )

    return llm_manager
