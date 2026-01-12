"""Circuit breaker pattern for fault tolerance."""

from typing import Optional, Callable, Any, Dict, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import functools

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation, requests flow through
    OPEN = "open"  # Failing, requests are rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: float = 30.0  # Seconds before trying again (open -> half-open)
    half_open_max_calls: int = 3  # Max calls allowed in half-open state
    excluded_exceptions: tuple = ()  # Exceptions that don't count as failures
    fallback: Optional[Callable] = None  # Fallback function when open


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    current_state: CircuitState = CircuitState.CLOSED


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    Usage:
        breaker = CircuitBreaker("my-service")

        @breaker
        async def call_external_service():
            ...

        # Or manually
        async with breaker:
            await call_external_service()
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        self._stats = CircuitBreakerStats()

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        self._stats.current_state = self._state
        return self._stats

    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self._state == CircuitState.OPEN

    def is_closed(self) -> bool:
        """Check if circuit is closed."""
        return self._state == CircuitState.CLOSED

    def is_half_open(self) -> bool:
        """Check if circuit is half-open."""
        return self._state == CircuitState.HALF_OPEN

    async def can_execute(self) -> bool:
        """Check if execution is allowed."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if timeout has passed
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited calls in half-open state
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.last_success_time = datetime.utcnow()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    async def record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        async with self._lock:
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.last_failure_time = datetime.utcnow()
            self._last_failure_time = datetime.utcnow()

            # Check if exception should be excluded
            if isinstance(exception, self.config.excluded_exceptions):
                return

            self._failure_count += 1

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    async def record_rejection(self) -> None:
        """Record a rejected call."""
        async with self._lock:
            self._stats.rejected_calls += 1

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset from open state."""
        if self._last_failure_time is None:
            return True
        elapsed = (datetime.utcnow() - self._last_failure_time).total_seconds()
        return elapsed >= self.config.timeout

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._stats.state_changes += 1

        logger.info(
            f"Circuit breaker '{self.name}' state change: {old_state.value} -> {new_state.value}"
        )

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.OPEN:
            self._success_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
            self._half_open_calls = 0

    async def __aenter__(self):
        """Enter context manager."""
        if not await self.can_execute():
            await self.record_rejection()
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is {self._state.value}"
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if exc_type is None:
            await self.record_success()
        elif exc_val is not None:
            await self.record_failure(exc_val)
        return False

    def __call__(self, func: Callable) -> Callable:
        """Decorator for protecting functions."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with self:
                return await func(*args, **kwargs)
        return wrapper

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time = None
        logger.info(f"Circuit breaker '{self.name}' manually reset")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Usage:
        registry = CircuitBreakerRegistry()
        breaker = registry.get_or_create("my-service")
    """

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
        self._default_config = CircuitBreakerConfig()

    def set_default_config(self, config: CircuitBreakerConfig) -> None:
        """Set default configuration for new breakers."""
        self._default_config = config

    async def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name,
                    config or self._default_config,
                )
            return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)

    def get_all_stats(self) -> Dict[str, CircuitBreakerStats]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.stats for name, breaker in self._breakers.items()}

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        async with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()


# Global registry
_registry = CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker from the global registry."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_registry.get_or_create(name, config))


async def get_circuit_breaker_async(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker from the global registry (async)."""
    return await _registry.get_or_create(name, config)


def circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> Callable:
    """
    Decorator for adding circuit breaker to functions.

    Usage:
        @circuit_breaker("external-api")
        async def call_api():
            ...
    """
    def decorator(func: Callable) -> Callable:
        breaker: Optional[CircuitBreaker] = None

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal breaker
            if breaker is None:
                breaker = await _registry.get_or_create(name, config)

            async with breaker:
                return await func(*args, **kwargs)

        return wrapper
    return decorator
