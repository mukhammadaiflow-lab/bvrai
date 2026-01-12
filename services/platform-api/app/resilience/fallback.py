"""
Fallback Strategies

Advanced fallback patterns for graceful degradation:
- Cache-based fallbacks
- Default value fallbacks
- Circuit breaker fallbacks
- Graceful degradation chains
"""

from typing import Optional, Callable, Dict, Any, TypeVar, Generic, Union, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from functools import wraps
import asyncio
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FallbackStrategy(ABC, Generic[T]):
    """Abstract base for fallback strategies."""

    @abstractmethod
    async def get_fallback(
        self,
        operation: str,
        exception: Exception,
        *args,
        **kwargs,
    ) -> T:
        """Get fallback value for failed operation."""
        pass

    @abstractmethod
    def supports(self, exception: Exception) -> bool:
        """Check if this strategy can handle the exception."""
        pass


class DefaultFallback(FallbackStrategy[T]):
    """
    Returns a default value on failure.

    Usage:
        fallback = DefaultFallback(default_value=[])

        @fallback
        async def get_items():
            ...
    """

    def __init__(
        self,
        default_value: T,
        exceptions: tuple = (Exception,),
    ):
        self.default_value = default_value
        self.exceptions = exceptions

    async def get_fallback(
        self,
        operation: str,
        exception: Exception,
        *args,
        **kwargs,
    ) -> T:
        """Return default value."""
        logger.warning(
            f"Returning default value for '{operation}' due to: {exception}"
        )
        return self.default_value

    def supports(self, exception: Exception) -> bool:
        """Check if exception is supported."""
        return isinstance(exception, self.exceptions)

    def __call__(self, func: Callable) -> Callable:
        """Decorator."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except self.exceptions as e:
                return await self.get_fallback(func.__name__, e, *args, **kwargs)
        return wrapper


class CacheFallback(FallbackStrategy[T]):
    """
    Returns cached value on failure.

    Uses a cached result when the operation fails.
    """

    def __init__(
        self,
        cache: Optional[Dict[str, Any]] = None,
        ttl_seconds: int = 300,
        exceptions: tuple = (Exception,),
        key_func: Optional[Callable[..., str]] = None,
    ):
        self._cache: Dict[str, Dict[str, Any]] = cache or {}
        self.ttl_seconds = ttl_seconds
        self.exceptions = exceptions
        self.key_func = key_func

    def _make_key(self, operation: str, *args, **kwargs) -> str:
        """Generate cache key."""
        if self.key_func:
            return self.key_func(*args, **kwargs)
        return f"{operation}:{hash((args, frozenset(kwargs.items())))}"

    async def cache_result(
        self,
        operation: str,
        result: T,
        *args,
        **kwargs,
    ) -> None:
        """Cache a successful result."""
        key = self._make_key(operation, *args, **kwargs)
        self._cache[key] = {
            "value": result,
            "cached_at": datetime.utcnow(),
        }

    async def get_fallback(
        self,
        operation: str,
        exception: Exception,
        *args,
        **kwargs,
    ) -> T:
        """Get cached value."""
        key = self._make_key(operation, *args, **kwargs)

        if key not in self._cache:
            logger.warning(f"No cached value for '{operation}', re-raising")
            raise exception

        cached = self._cache[key]
        age = (datetime.utcnow() - cached["cached_at"]).total_seconds()

        if age > self.ttl_seconds:
            logger.warning(f"Cached value for '{operation}' expired")
            # Still return stale value on failure
            logger.info(f"Returning stale cached value (age: {age:.0f}s)")

        logger.warning(
            f"Returning cached value for '{operation}' due to: {exception}"
        )
        return cached["value"]

    def supports(self, exception: Exception) -> bool:
        """Check if exception is supported."""
        return isinstance(exception, self.exceptions)

    def __call__(self, func: Callable) -> Callable:
        """Decorator with automatic caching."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                await self.cache_result(func.__name__, result, *args, **kwargs)
                return result
            except self.exceptions as e:
                return await self.get_fallback(func.__name__, e, *args, **kwargs)
        return wrapper


class CircuitFallback(FallbackStrategy[T]):
    """
    Fallback that integrates with circuit breaker.

    Uses different fallbacks based on circuit state.
    """

    def __init__(
        self,
        circuit_breaker: Any,  # CircuitBreaker type
        closed_fallback: Optional[FallbackStrategy[T]] = None,
        open_fallback: Optional[FallbackStrategy[T]] = None,
        half_open_fallback: Optional[FallbackStrategy[T]] = None,
    ):
        self.circuit_breaker = circuit_breaker
        self.closed_fallback = closed_fallback
        self.open_fallback = open_fallback
        self.half_open_fallback = half_open_fallback

    async def get_fallback(
        self,
        operation: str,
        exception: Exception,
        *args,
        **kwargs,
    ) -> T:
        """Get fallback based on circuit state."""
        from app.resilience.circuit_breaker import CircuitState

        state = self.circuit_breaker.state

        if state == CircuitState.CLOSED and self.closed_fallback:
            return await self.closed_fallback.get_fallback(
                operation, exception, *args, **kwargs
            )
        elif state == CircuitState.OPEN and self.open_fallback:
            return await self.open_fallback.get_fallback(
                operation, exception, *args, **kwargs
            )
        elif state == CircuitState.HALF_OPEN and self.half_open_fallback:
            return await self.half_open_fallback.get_fallback(
                operation, exception, *args, **kwargs
            )

        # No fallback configured, re-raise
        raise exception

    def supports(self, exception: Exception) -> bool:
        """Check if any fallback supports exception."""
        return (
            (self.closed_fallback and self.closed_fallback.supports(exception)) or
            (self.open_fallback and self.open_fallback.supports(exception)) or
            (self.half_open_fallback and self.half_open_fallback.supports(exception))
        )


class FallbackChain(FallbackStrategy[T]):
    """
    Chain of fallback strategies.

    Tries each fallback in order until one succeeds.
    """

    def __init__(self, strategies: List[FallbackStrategy[T]]):
        self.strategies = strategies

    async def get_fallback(
        self,
        operation: str,
        exception: Exception,
        *args,
        **kwargs,
    ) -> T:
        """Try each strategy in order."""
        last_exception = exception

        for strategy in self.strategies:
            if strategy.supports(last_exception):
                try:
                    return await strategy.get_fallback(
                        operation, last_exception, *args, **kwargs
                    )
                except Exception as e:
                    logger.debug(f"Fallback strategy failed: {e}")
                    last_exception = e

        # All fallbacks failed
        raise last_exception

    def supports(self, exception: Exception) -> bool:
        """Check if any strategy supports exception."""
        return any(s.supports(exception) for s in self.strategies)

    def __call__(self, func: Callable) -> Callable:
        """Decorator."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                return await self.get_fallback(func.__name__, e, *args, **kwargs)
        return wrapper


class CallbackFallback(FallbackStrategy[T]):
    """
    Uses a callback function as fallback.

    The callback receives the exception and original arguments.
    """

    def __init__(
        self,
        callback: Callable[..., T],
        exceptions: tuple = (Exception,),
    ):
        self.callback = callback
        self.exceptions = exceptions

    async def get_fallback(
        self,
        operation: str,
        exception: Exception,
        *args,
        **kwargs,
    ) -> T:
        """Call the fallback callback."""
        if asyncio.iscoroutinefunction(self.callback):
            return await self.callback(exception, *args, **kwargs)
        return self.callback(exception, *args, **kwargs)

    def supports(self, exception: Exception) -> bool:
        """Check if exception is supported."""
        return isinstance(exception, self.exceptions)


class GracefulDegradation:
    """
    Graceful degradation pattern.

    Provides different levels of service based on system health.
    """

    def __init__(
        self,
        levels: Dict[int, Callable[..., T]],
        health_check: Optional[Callable[[], int]] = None,
    ):
        """
        Args:
            levels: Dict of level -> handler function (0 = full, higher = degraded)
            health_check: Function returning current degradation level
        """
        self.levels = levels
        self.health_check = health_check or (lambda: 0)
        self._current_level = 0
        self._lock = asyncio.Lock()

    @property
    def current_level(self) -> int:
        """Get current degradation level."""
        return self._current_level

    async def set_level(self, level: int) -> None:
        """Set degradation level."""
        async with self._lock:
            if level != self._current_level:
                logger.warning(f"Degradation level changed: {self._current_level} -> {level}")
                self._current_level = level

    async def execute(self, *args, **kwargs) -> T:
        """Execute at current degradation level."""
        level = self._current_level

        if level not in self.levels:
            # Find closest available level
            available = sorted(self.levels.keys())
            level = min(available, key=lambda x: abs(x - level))

        handler = self.levels[level]

        if asyncio.iscoroutinefunction(handler):
            return await handler(*args, **kwargs)
        return handler(*args, **kwargs)

    def __call__(self, level: int) -> Callable:
        """Decorator to register a level handler."""
        def decorator(func: Callable) -> Callable:
            self.levels[level] = func
            return func
        return decorator


class FeatureFallback:
    """
    Feature-based fallback that can disable features.

    Useful for graceful degradation of non-critical features.
    """

    def __init__(self):
        self._disabled_features: set = set()
        self._fallbacks: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()

    async def disable_feature(self, feature: str) -> None:
        """Disable a feature."""
        async with self._lock:
            self._disabled_features.add(feature)
            logger.warning(f"Feature disabled: {feature}")

    async def enable_feature(self, feature: str) -> None:
        """Enable a feature."""
        async with self._lock:
            self._disabled_features.discard(feature)
            logger.info(f"Feature enabled: {feature}")

    def is_disabled(self, feature: str) -> bool:
        """Check if feature is disabled."""
        return feature in self._disabled_features

    def register_fallback(self, feature: str, fallback: Callable) -> None:
        """Register a fallback for a feature."""
        self._fallbacks[feature] = fallback

    def feature(self, name: str, fallback: Optional[Callable] = None) -> Callable:
        """Decorator to mark a feature with fallback."""
        def decorator(func: Callable) -> Callable:
            if fallback:
                self._fallbacks[name] = fallback

            @wraps(func)
            async def wrapper(*args, **kwargs):
                if self.is_disabled(name):
                    if name in self._fallbacks:
                        fb = self._fallbacks[name]
                        if asyncio.iscoroutinefunction(fb):
                            return await fb(*args, **kwargs)
                        return fb(*args, **kwargs)
                    return None

                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Feature '{name}' failed: {e}")
                    # Auto-disable on repeated failures (could add threshold)
                    if name in self._fallbacks:
                        fb = self._fallbacks[name]
                        if asyncio.iscoroutinefunction(fb):
                            return await fb(*args, **kwargs)
                        return fb(*args, **kwargs)
                    raise

            return wrapper
        return decorator


class RetryWithFallback(FallbackStrategy[T]):
    """
    Combines retry with fallback.

    Retries the operation, then falls back if all retries fail.
    """

    def __init__(
        self,
        max_retries: int = 3,
        fallback_value: Optional[T] = None,
        fallback_func: Optional[Callable[..., T]] = None,
        exceptions: tuple = (Exception,),
    ):
        self.max_retries = max_retries
        self.fallback_value = fallback_value
        self.fallback_func = fallback_func
        self.exceptions = exceptions

    async def get_fallback(
        self,
        operation: str,
        exception: Exception,
        *args,
        **kwargs,
    ) -> T:
        """Get fallback after retries exhausted."""
        if self.fallback_func:
            if asyncio.iscoroutinefunction(self.fallback_func):
                return await self.fallback_func(exception, *args, **kwargs)
            return self.fallback_func(exception, *args, **kwargs)
        return self.fallback_value

    def supports(self, exception: Exception) -> bool:
        """Check if exception is supported."""
        return isinstance(exception, self.exceptions)

    def __call__(self, func: Callable) -> Callable:
        """Decorator with retry and fallback."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(self.max_retries):
                try:
                    return await func(*args, **kwargs)
                except self.exceptions as e:
                    last_exception = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed: {e}"
                    )
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff

            return await self.get_fallback(func.__name__, last_exception, *args, **kwargs)

        return wrapper


# Decorator factory
def fallback(
    default: Optional[T] = None,
    fallback_func: Optional[Callable[..., T]] = None,
    cache: bool = False,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Flexible fallback decorator.

    Usage:
        @fallback(default=[])
        async def get_items():
            ...

        @fallback(fallback_func=get_cached_items)
        async def get_items():
            ...
    """
    def decorator(func: Callable) -> Callable:
        if cache:
            strategy = CacheFallback(exceptions=exceptions)
        elif fallback_func:
            strategy = CallbackFallback(fallback_func, exceptions=exceptions)
        else:
            strategy = DefaultFallback(default, exceptions=exceptions)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                if cache and hasattr(strategy, 'cache_result'):
                    await strategy.cache_result(func.__name__, result, *args, **kwargs)
                return result
            except exceptions as e:
                return await strategy.get_fallback(func.__name__, e, *args, **kwargs)

        return wrapper
    return decorator


class ShedLoadFallback:
    """
    Load shedding fallback pattern.

    When under load, starts rejecting lower-priority requests.
    """

    def __init__(
        self,
        load_threshold: float = 0.8,
        load_check: Optional[Callable[[], float]] = None,
        priority_check: Optional[Callable[..., int]] = None,
    ):
        """
        Args:
            load_threshold: Load level at which to start shedding (0-1)
            load_check: Function returning current load (0-1)
            priority_check: Function returning request priority (higher = more important)
        """
        self.load_threshold = load_threshold
        self.load_check = load_check or (lambda: 0.0)
        self.priority_check = priority_check or (lambda *a, **k: 0)

    def should_shed(self, *args, **kwargs) -> bool:
        """Check if request should be shed."""
        current_load = self.load_check()
        if current_load < self.load_threshold:
            return False

        # Higher priority requests are less likely to be shed
        priority = self.priority_check(*args, **kwargs)
        shed_probability = (current_load - self.load_threshold) / (1 - self.load_threshold)
        priority_factor = 1.0 / (1 + priority)

        import random
        return random.random() < (shed_probability * priority_factor)

    def __call__(
        self,
        fallback_value: Optional[T] = None,
        fallback_func: Optional[Callable[..., T]] = None,
    ) -> Callable:
        """Decorator with load shedding."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if self.should_shed(*args, **kwargs):
                    logger.warning(f"Shedding load for {func.__name__}")
                    if fallback_func:
                        if asyncio.iscoroutinefunction(fallback_func):
                            return await fallback_func(*args, **kwargs)
                        return fallback_func(*args, **kwargs)
                    return fallback_value

                return await func(*args, **kwargs)
            return wrapper
        return decorator
