"""
LLM Orchestrator

Main orchestration layer that provides:
- Multi-provider routing
- Automatic fallback
- Load balancing
- Rate limiting
- Cost optimization
- Request tracking
"""

import asyncio
import logging
import random
import time
from collections import defaultdict
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

from .base import (
    BaseLLMProvider,
    LLMConfig,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMUsage,
    ModelInfo,
    ProviderConfig,
    StreamChunk,
    get_model_info,
)
from .providers import LLMProviderFactory
from .tools import ToolCall, ToolRegistry, ToolResult
from .context import ContextManager


logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """Strategies for routing requests to providers."""

    # Use single provider
    SINGLE = "single"

    # Round-robin across providers
    ROUND_ROBIN = "round_robin"

    # Route based on lowest cost
    COST_OPTIMIZED = "cost_optimized"

    # Route based on lowest latency
    LATENCY_OPTIMIZED = "latency_optimized"

    # Route based on model capabilities
    CAPABILITY_BASED = "capability_based"

    # Random selection
    RANDOM = "random"

    # Weighted distribution
    WEIGHTED = "weighted"


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""

    # Enable fallback on failure
    enabled: bool = True

    # Maximum fallback attempts
    max_attempts: int = 3

    # Errors that trigger fallback
    fallback_on_errors: List[str] = field(default_factory=lambda: [
        "rate_limit",
        "timeout",
        "server_error",
        "connection_error",
        "model_unavailable",
    ])

    # Delay between fallback attempts (ms)
    retry_delay_ms: int = 500

    # Exponential backoff multiplier
    backoff_multiplier: float = 2.0

    # Maximum retry delay (ms)
    max_retry_delay_ms: int = 10000


@dataclass
class ProviderHealth:
    """Health status of a provider."""

    provider: str
    model: str

    # Request tracking
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Latency tracking
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0

    # Error tracking
    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    last_error: Optional[str] = None

    # Circuit breaker state
    circuit_open: bool = False
    circuit_open_until: Optional[float] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return float('inf')
        return self.total_latency_ms / self.successful_requests

    def record_success(self, latency_ms: float) -> None:
        """Record a successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.consecutive_failures = 0

        # Close circuit on success
        if self.circuit_open:
            self.circuit_open = False
            self.circuit_open_until = None

    def record_failure(self, error: str) -> None:
        """Record a failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        self.last_error = error

        # Open circuit after consecutive failures
        if self.consecutive_failures >= 3:
            self.circuit_open = True
            self.circuit_open_until = time.time() + 60  # 1 minute

    def is_available(self) -> bool:
        """Check if provider is available."""
        if not self.circuit_open:
            return True

        # Check if circuit timeout has passed
        if self.circuit_open_until and time.time() > self.circuit_open_until:
            # Allow one request to test
            self.circuit_open = False
            return True

        return False


@dataclass
class RequestContext:
    """Context for a single request."""

    request_id: str
    messages: List[LLMMessage]
    config: LLMConfig

    # Tracking
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0

    # Result
    response: Optional[LLMResponse] = None
    error: Optional[str] = None
    successful_provider: Optional[str] = None


@dataclass
class OrchestratorConfig:
    """Configuration for the LLM orchestrator."""

    # Primary provider/model
    default_provider: str = "openai"
    default_model: str = "gpt-4o-mini"

    # Routing
    routing_strategy: RoutingStrategy = RoutingStrategy.SINGLE

    # Provider weights (for weighted routing)
    provider_weights: Dict[str, float] = field(default_factory=dict)

    # Fallback configuration
    fallback: FallbackConfig = field(default_factory=FallbackConfig)

    # Fallback chain (provider:model pairs)
    fallback_chain: List[Tuple[str, str]] = field(default_factory=list)

    # Rate limiting
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 100000

    # Cost controls
    max_cost_per_request: float = 1.0
    max_cost_per_day: float = 100.0

    # Timeouts
    request_timeout_ms: int = 60000

    # Tool execution
    max_tool_iterations: int = 10
    parallel_tool_execution: bool = True


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100000,
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute

        self._request_tokens = requests_per_minute
        self._token_tokens = tokens_per_minute
        self._last_refill = time.time()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill
        minutes = elapsed / 60

        self._request_tokens = min(
            self.requests_per_minute,
            self._request_tokens + minutes * self.requests_per_minute
        )
        self._token_tokens = min(
            self.tokens_per_minute,
            self._token_tokens + minutes * self.tokens_per_minute
        )
        self._last_refill = now

    async def acquire(self, token_count: int = 0) -> bool:
        """Acquire permission to make a request."""
        self._refill()

        if self._request_tokens < 1:
            return False

        if token_count > 0 and self._token_tokens < token_count:
            return False

        self._request_tokens -= 1
        if token_count > 0:
            self._token_tokens -= token_count

        return True

    async def wait_for_capacity(
        self,
        token_count: int = 0,
        timeout: float = 60.0,
    ) -> bool:
        """Wait until capacity is available."""
        start = time.time()

        while time.time() - start < timeout:
            if await self.acquire(token_count):
                return True
            await asyncio.sleep(0.1)

        return False


class LLMOrchestrator:
    """
    Main LLM orchestration layer.

    Provides intelligent routing, fallback, and tool execution
    across multiple LLM providers.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        context_manager: Optional[ContextManager] = None,
    ):
        self.config = config or OrchestratorConfig()
        self.tool_registry = tool_registry
        self.context_manager = context_manager

        # Providers
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._provider_configs: Dict[str, ProviderConfig] = {}

        # Health tracking
        self._health: Dict[str, ProviderHealth] = {}

        # Rate limiting
        self._rate_limiters: Dict[str, RateLimiter] = {}

        # Cost tracking
        self._daily_cost: float = 0.0
        self._cost_reset_time: float = time.time()

        # Request tracking
        self._request_count: int = 0
        self._round_robin_index: int = 0

        # Usage accumulator
        self._total_usage = LLMUsage()

    def add_provider(
        self,
        name: str,
        provider: Optional[BaseLLMProvider] = None,
        config: Optional[ProviderConfig] = None,
        **kwargs,
    ) -> None:
        """Add a provider to the orchestrator."""
        if provider is None:
            provider = LLMProviderFactory.create(name, config=config, **kwargs)

        self._providers[name] = provider
        self._provider_configs[name] = config or ProviderConfig()

        # Initialize rate limiter
        cfg = self._provider_configs[name]
        self._rate_limiters[name] = RateLimiter(
            requests_per_minute=cfg.max_requests_per_minute,
            tokens_per_minute=cfg.max_tokens_per_minute,
        )

    def remove_provider(self, name: str) -> None:
        """Remove a provider."""
        if name in self._providers:
            del self._providers[name]
            del self._provider_configs[name]
            del self._rate_limiters[name]

    def _get_health_key(self, provider: str, model: str) -> str:
        """Get health tracking key."""
        return f"{provider}:{model}"

    def _get_or_create_health(
        self,
        provider: str,
        model: str,
    ) -> ProviderHealth:
        """Get or create health tracker."""
        key = self._get_health_key(provider, model)
        if key not in self._health:
            self._health[key] = ProviderHealth(provider=provider, model=model)
        return self._health[key]

    def _select_provider(
        self,
        config: LLMConfig,
    ) -> Tuple[str, BaseLLMProvider]:
        """Select a provider based on routing strategy."""
        # Determine provider from model
        model_info = get_model_info(config.model)
        if model_info:
            provider_name = model_info.provider.value
        else:
            provider_name = self.config.default_provider

        # Check if provider exists
        if provider_name not in self._providers:
            # Try to create it
            self.add_provider(provider_name)

        # Apply routing strategy
        if self.config.routing_strategy == RoutingStrategy.SINGLE:
            pass  # Use determined provider

        elif self.config.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            available = list(self._providers.keys())
            self._round_robin_index = (self._round_robin_index + 1) % len(available)
            provider_name = available[self._round_robin_index]

        elif self.config.routing_strategy == RoutingStrategy.RANDOM:
            provider_name = random.choice(list(self._providers.keys()))

        elif self.config.routing_strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            best_latency = float('inf')
            best_provider = provider_name

            for name in self._providers:
                health = self._get_or_create_health(name, config.model)
                if health.is_available() and health.avg_latency_ms < best_latency:
                    best_latency = health.avg_latency_ms
                    best_provider = name

            provider_name = best_provider

        elif self.config.routing_strategy == RoutingStrategy.COST_OPTIMIZED:
            best_cost = float('inf')
            best_provider = provider_name

            for name in self._providers:
                info = get_model_info(config.model)
                if info and info.output_price_per_1k < best_cost:
                    best_cost = info.output_price_per_1k
                    best_provider = name

            provider_name = best_provider

        elif self.config.routing_strategy == RoutingStrategy.WEIGHTED:
            weights = self.config.provider_weights
            if weights:
                providers = list(weights.keys())
                weight_values = [weights.get(p, 1.0) for p in providers]
                provider_name = random.choices(providers, weights=weight_values)[0]

        provider = self._providers.get(provider_name)
        if not provider:
            raise ValueError(f"Provider not available: {provider_name}")

        return provider_name, provider

    def _get_fallback_chain(
        self,
        primary_provider: str,
        config: LLMConfig,
    ) -> List[Tuple[str, str]]:
        """Get fallback chain for a request."""
        if self.config.fallback_chain:
            return self.config.fallback_chain

        # Build automatic fallback chain
        chain = []

        # Add same-provider fallback models
        model_info = get_model_info(config.model)
        if model_info:
            provider_type = model_info.provider

            # Add cheaper/faster models from same provider
            if provider_type == LLMProvider.OPENAI:
                if config.model != "gpt-4o-mini":
                    chain.append(("openai", "gpt-4o-mini"))
                if config.model != "gpt-3.5-turbo":
                    chain.append(("openai", "gpt-3.5-turbo"))

            elif provider_type == LLMProvider.ANTHROPIC:
                if config.model != "claude-3-5-haiku-20241022":
                    chain.append(("anthropic", "claude-3-5-haiku-20241022"))

        # Add cross-provider fallbacks
        for name in self._providers:
            if name != primary_provider:
                default_model = self._provider_configs[name].default_model
                if default_model:
                    chain.append((name, default_model))

        return chain

    async def complete(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Generate a completion with automatic fallback.
        """
        config = config or LLMConfig(model=self.config.default_model)

        # Generate request ID
        self._request_count += 1
        request_id = f"req_{self._request_count}_{int(time.time())}"

        # Create request context
        ctx = RequestContext(
            request_id=request_id,
            messages=messages,
            config=config,
        )

        # Select primary provider
        primary_name, primary_provider = self._select_provider(config)

        # Try primary provider
        response = await self._try_provider(
            ctx,
            primary_name,
            primary_provider,
            config,
        )

        if response:
            return response

        # Try fallback chain
        if self.config.fallback.enabled:
            fallback_chain = self._get_fallback_chain(primary_name, config)

            for fallback_name, fallback_model in fallback_chain:
                if len(ctx.attempts) >= self.config.fallback.max_attempts:
                    break

                # Get or create provider
                if fallback_name not in self._providers:
                    try:
                        self.add_provider(fallback_name)
                    except Exception:
                        continue

                fallback_provider = self._providers.get(fallback_name)
                if not fallback_provider:
                    continue

                # Update config with fallback model
                fallback_config = LLMConfig(
                    model=fallback_model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    tools=config.tools,
                    tool_choice=config.tool_choice,
                )

                # Exponential backoff
                delay = self.config.fallback.retry_delay_ms * (
                    self.config.fallback.backoff_multiplier ** (len(ctx.attempts) - 1)
                )
                delay = min(delay, self.config.fallback.max_retry_delay_ms)
                await asyncio.sleep(delay / 1000)

                response = await self._try_provider(
                    ctx,
                    fallback_name,
                    fallback_provider,
                    fallback_config,
                )

                if response:
                    return response

        # All attempts failed
        raise RuntimeError(
            f"All LLM providers failed. Attempts: {len(ctx.attempts)}. "
            f"Last error: {ctx.error}"
        )

    async def _try_provider(
        self,
        ctx: RequestContext,
        provider_name: str,
        provider: BaseLLMProvider,
        config: LLMConfig,
    ) -> Optional[LLMResponse]:
        """Try to complete request with a specific provider."""
        health = self._get_or_create_health(provider_name, config.model)

        # Check circuit breaker
        if not health.is_available():
            logger.warning(f"Provider {provider_name} circuit open, skipping")
            return None

        # Check rate limit
        rate_limiter = self._rate_limiters.get(provider_name)
        if rate_limiter and not await rate_limiter.acquire():
            logger.warning(f"Provider {provider_name} rate limited")
            return None

        # Check cost limit
        self._check_cost_reset()
        if self._daily_cost >= self.config.max_cost_per_day:
            logger.warning("Daily cost limit reached")
            return None

        start_time = time.time()

        try:
            response = await asyncio.wait_for(
                provider.complete(ctx.messages, config),
                timeout=self.config.request_timeout_ms / 1000,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Record success
            health.record_success(latency_ms)

            # Track usage
            self._total_usage = self._total_usage + response.usage
            self._daily_cost += response.usage.cost_usd

            # Record attempt
            ctx.attempts.append({
                "provider": provider_name,
                "model": config.model,
                "success": True,
                "latency_ms": latency_ms,
            })

            ctx.response = response
            ctx.successful_provider = provider_name

            return response

        except asyncio.TimeoutError:
            error = "timeout"
            health.record_failure(error)
            ctx.error = error
            ctx.attempts.append({
                "provider": provider_name,
                "model": config.model,
                "success": False,
                "error": error,
            })
            logger.warning(f"Provider {provider_name} timed out")

        except Exception as e:
            error = str(e)
            health.record_failure(error)
            ctx.error = error
            ctx.attempts.append({
                "provider": provider_name,
                "model": config.model,
                "success": False,
                "error": error,
            })
            logger.warning(f"Provider {provider_name} error: {error}")

        return None

    async def stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Generate a streaming completion with automatic fallback.
        """
        config = config or LLMConfig(model=self.config.default_model)
        config.stream = True

        # Select provider
        provider_name, provider = self._select_provider(config)

        # Check availability
        health = self._get_or_create_health(provider_name, config.model)
        if not health.is_available():
            raise RuntimeError(f"Provider {provider_name} not available")

        start_time = time.time()

        try:
            async for chunk in provider.stream(messages, config):
                yield chunk

            # Record success
            latency_ms = (time.time() - start_time) * 1000
            health.record_success(latency_ms)

        except Exception as e:
            health.record_failure(str(e))
            raise

    async def complete_with_tools(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
        max_iterations: Optional[int] = None,
    ) -> Tuple[LLMResponse, List[ToolResult]]:
        """
        Generate completion with automatic tool execution.

        Continues until no more tool calls or max iterations reached.
        """
        if not self.tool_registry:
            raise ValueError("Tool registry not configured")

        config = config or LLMConfig(model=self.config.default_model)
        max_iterations = max_iterations or self.config.max_tool_iterations

        # Add tools to config
        if not config.tools:
            config.tools = self.tool_registry.to_openai_format()

        # Conversation history for this execution
        conversation = list(messages)
        all_tool_results: List[ToolResult] = []

        for iteration in range(max_iterations):
            # Get completion
            response = await self.complete(conversation, config)

            # Check for tool calls
            if not response.tool_calls:
                return response, all_tool_results

            # Execute tool calls
            tool_calls = [ToolCall.from_dict(tc) for tc in response.tool_calls]

            results = await self.tool_registry.execute_all(
                tool_calls,
                parallel=self.config.parallel_tool_execution,
            )

            all_tool_results.extend(results)

            # Add assistant message with tool calls
            conversation.append(response.to_message())

            # Add tool results
            for result in results:
                conversation.append(LLMMessage.tool(
                    content=result.to_message_content(),
                    tool_call_id=result.tool_call_id,
                    name=result.tool_name,
                ))

        # Max iterations reached
        logger.warning(f"Max tool iterations ({max_iterations}) reached")
        return response, all_tool_results

    def _check_cost_reset(self) -> None:
        """Reset daily cost if needed."""
        now = time.time()
        if now - self._cost_reset_time > 86400:  # 24 hours
            self._daily_cost = 0.0
            self._cost_reset_time = now

    def get_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all providers."""
        return {
            key: {
                "provider": health.provider,
                "model": health.model,
                "total_requests": health.total_requests,
                "success_rate": health.success_rate,
                "avg_latency_ms": health.avg_latency_ms,
                "circuit_open": health.circuit_open,
                "last_error": health.last_error,
            }
            for key, health in self._health.items()
        }

    def get_usage(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self._request_count,
            "total_tokens": self._total_usage.total_tokens,
            "total_cost_usd": self._total_usage.cost_usd,
            "daily_cost_usd": self._daily_cost,
            "providers": {
                key: {
                    "requests": health.total_requests,
                    "success_rate": health.success_rate,
                }
                for key, health in self._health.items()
            },
        }

    async def close(self) -> None:
        """Close all providers."""
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()


# Convenience function for creating orchestrator
def create_orchestrator(
    default_provider: str = "openai",
    default_model: str = "gpt-4o-mini",
    fallback_enabled: bool = True,
    tool_registry: Optional[ToolRegistry] = None,
    **provider_kwargs,
) -> LLMOrchestrator:
    """
    Create an LLM orchestrator with common defaults.
    """
    config = OrchestratorConfig(
        default_provider=default_provider,
        default_model=default_model,
        fallback=FallbackConfig(enabled=fallback_enabled),
    )

    orchestrator = LLMOrchestrator(
        config=config,
        tool_registry=tool_registry,
    )

    # Add default provider
    orchestrator.add_provider(default_provider, **provider_kwargs)

    return orchestrator


__all__ = [
    "RoutingStrategy",
    "FallbackConfig",
    "ProviderHealth",
    "RequestContext",
    "OrchestratorConfig",
    "RateLimiter",
    "LLMOrchestrator",
    "create_orchestrator",
]
