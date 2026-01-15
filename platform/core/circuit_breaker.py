"""
Circuit Breaker and Resilience Patterns
=======================================

Enterprise-grade resilience patterns for building fault-tolerant
distributed systems.

Features:
- Circuit breaker pattern
- Retry with exponential backoff
- Bulkhead pattern
- Timeout management
- Fallback handlers
- Rate limiting
- Health-aware routing

Author: Builder Engine Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

T = TypeVar("T")
FallbackHandler = Callable[..., Awaitable[T]]


# =============================================================================
# ENUMS
# =============================================================================


class CircuitState(str, Enum):
    """Circuit breaker states"""

    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class RetryStrategy(str, Enum):
    """Retry strategies"""

    NONE = "none"
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    DECORRELATED_JITTER = "decorrelated_jitter"


# =============================================================================
# CONFIGURATION
# =============================================================================


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration"""

    # Thresholds
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 30.0

    # Timing
    reset_timeout_seconds: float = 60.0
    half_open_max_calls: int = 3

    # Metrics window
    rolling_window_seconds: float = 60.0
    rolling_window_buckets: int = 10

    # Error handling
    record_exceptions: List[str] = Field(default_factory=list)
    ignore_exceptions: List[str] = Field(default_factory=list)

    # Rate limiting
    max_requests_per_second: Optional[float] = None


class RetryPolicy(BaseModel):
    """Retry policy configuration"""

    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.1

    # Retryable conditions
    retryable_exceptions: List[str] = Field(default_factory=list)
    retryable_status_codes: List[int] = Field(default_factory=lambda: [500, 502, 503, 504])

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if self.strategy == RetryStrategy.NONE:
            return 0

        elif self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay_seconds

        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay_seconds * attempt

        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay_seconds * (self.multiplier ** (attempt - 1))

        elif self.strategy == RetryStrategy.DECORRELATED_JITTER:
            # AWS-style decorrelated jitter
            delay = min(
                self.max_delay_seconds,
                random.uniform(self.base_delay_seconds, delay * 3 if attempt > 1 else self.base_delay_seconds * 3)
            )
            return delay

        else:
            delay = self.base_delay_seconds

        # Apply jitter
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)

        return min(delay, self.max_delay_seconds)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics"""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    timeout_calls: int = 0
    state_transitions: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    avg_response_time_ms: float = 0.0
    total_response_time_ms: float = 0.0

    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls

    def record_success(self, response_time_ms: float) -> None:
        self.total_calls += 1
        self.successful_calls += 1
        self.last_success_time = datetime.utcnow()
        self.total_response_time_ms += response_time_ms
        self.avg_response_time_ms = self.total_response_time_ms / self.total_calls

    def record_failure(self, response_time_ms: float = 0) -> None:
        self.total_calls += 1
        self.failed_calls += 1
        self.last_failure_time = datetime.utcnow()
        if response_time_ms > 0:
            self.total_response_time_ms += response_time_ms
            self.avg_response_time_ms = self.total_response_time_ms / self.total_calls

    def record_rejection(self) -> None:
        self.rejected_calls += 1

    def record_timeout(self) -> None:
        self.timeout_calls += 1


@dataclass
class RollingWindowEntry:
    """Entry in a rolling window"""

    success: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    response_time_ms: float = 0.0


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation.

    Prevents cascading failures by detecting failures and
    temporarily blocking calls to a failing service.

    Usage:
        breaker = CircuitBreaker("external-api")

        @breaker
        async def call_api():
            return await http_client.get("/api")

        # Or manually
        async with breaker:
            result = await call_api()
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[FallbackHandler] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback = fallback

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_state_change: datetime = datetime.utcnow()
        self._half_open_calls = 0
        self._metrics = CircuitBreakerMetrics()
        self._rolling_window: Deque[RollingWindowEntry] = deque(
            maxlen=100
        )
        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger(f"circuit_breaker.{name}")

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        return self._state == CircuitState.HALF_OPEN

    @property
    def metrics(self) -> CircuitBreakerMetrics:
        return self._metrics

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    async def _check_state(self) -> bool:
        """Check if calls should be allowed"""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if reset timeout has passed
                if self._should_attempt_reset():
                    await self._transition_to(CircuitState.HALF_OPEN)
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited calls in half-open state
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset from open state"""
        if not self._last_failure_time:
            return True

        elapsed = (datetime.utcnow() - self._last_failure_time).total_seconds()
        return elapsed >= self.config.reset_timeout_seconds

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state"""
        old_state = self._state
        self._state = new_state
        self._last_state_change = datetime.utcnow()
        self._metrics.state_transitions += 1

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0

        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0

        self._logger.info(
            "circuit_state_changed",
            old_state=old_state.value,
            new_state=new_state.value
        )

    async def _record_success(self, response_time_ms: float) -> None:
        """Record a successful call"""
        async with self._lock:
            self._success_count += 1
            self._metrics.record_success(response_time_ms)
            self._rolling_window.append(
                RollingWindowEntry(success=True, response_time_ms=response_time_ms)
            )

            if self._state == CircuitState.HALF_OPEN:
                if self._success_count >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)

    async def _record_failure(self, response_time_ms: float = 0) -> None:
        """Record a failed call"""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.utcnow()
            self._metrics.record_failure(response_time_ms)
            self._rolling_window.append(
                RollingWindowEntry(success=False, response_time_ms=response_time_ms)
            )

            if self._state == CircuitState.HALF_OPEN:
                await self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs
    ) -> T:
        """Execute a function with circuit breaker protection"""
        # Check if call is allowed
        if not await self._check_state():
            self._metrics.record_rejection()

            if self.fallback:
                return await self.fallback(*args, **kwargs)

            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is open"
            )

        start_time = time.time()

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout_seconds
            )

            response_time_ms = (time.time() - start_time) * 1000
            await self._record_success(response_time_ms)

            return result

        except asyncio.TimeoutError:
            response_time_ms = (time.time() - start_time) * 1000
            self._metrics.record_timeout()
            await self._record_failure(response_time_ms)

            if self.fallback:
                return await self.fallback(*args, **kwargs)

            raise

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000

            # Check if exception should be recorded
            exception_name = type(e).__name__

            if self.config.ignore_exceptions and exception_name in self.config.ignore_exceptions:
                raise

            if self.config.record_exceptions:
                if exception_name not in self.config.record_exceptions:
                    raise

            await self._record_failure(response_time_ms)

            if self.fallback:
                return await self.fallback(*args, **kwargs)

            raise

    def __call__(
        self,
        func: Callable[..., Awaitable[T]]
    ) -> Callable[..., Awaitable[T]]:
        """Decorator to wrap a function with circuit breaker"""
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await self.execute(func, *args, **kwargs)
        return wrapper

    async def __aenter__(self) -> "CircuitBreaker":
        """Async context manager entry"""
        if not await self._check_state():
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is open"
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit"""
        if exc_type is None:
            await self._record_success(0)
        else:
            await self._record_failure(0)

    # -------------------------------------------------------------------------
    # Manual Control
    # -------------------------------------------------------------------------

    async def force_open(self) -> None:
        """Force the circuit to open state"""
        async with self._lock:
            await self._transition_to(CircuitState.OPEN)

    async def force_closed(self) -> None:
        """Force the circuit to closed state"""
        async with self._lock:
            await self._transition_to(CircuitState.CLOSED)

    async def reset(self) -> None:
        """Reset the circuit breaker"""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None
            self._rolling_window.clear()

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "metrics": {
                "total_calls": self._metrics.total_calls,
                "successful_calls": self._metrics.successful_calls,
                "failed_calls": self._metrics.failed_calls,
                "rejected_calls": self._metrics.rejected_calls,
                "timeout_calls": self._metrics.timeout_calls,
                "failure_rate": self._metrics.failure_rate,
                "success_rate": self._metrics.success_rate,
                "avg_response_time_ms": self._metrics.avg_response_time_ms
            },
            "last_failure_time": self._last_failure_time.isoformat() if self._last_failure_time else None,
            "last_state_change": self._last_state_change.isoformat()
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


# =============================================================================
# RETRY DECORATOR
# =============================================================================


def retry(
    policy: Optional[RetryPolicy] = None,
    on_retry: Optional[Callable[[int, Exception], Awaitable[None]]] = None
) -> Callable:
    """
    Retry decorator with configurable policy.

    Usage:
        @retry(RetryPolicy(max_retries=3))
        async def call_api():
            return await http_client.get("/api")
    """
    policy = policy or RetryPolicy()

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(1, policy.max_retries + 2):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if retryable
                    if policy.retryable_exceptions:
                        if type(e).__name__ not in policy.retryable_exceptions:
                            raise

                    if attempt <= policy.max_retries:
                        delay = policy.get_delay(attempt)

                        if on_retry:
                            await on_retry(attempt, e)

                        logger.warning(
                            "retry_attempt",
                            attempt=attempt,
                            delay=delay,
                            error=str(e)
                        )

                        await asyncio.sleep(delay)
                    else:
                        raise

            if last_exception:
                raise last_exception

        return wrapper
    return decorator


# =============================================================================
# BULKHEAD
# =============================================================================


class Bulkhead:
    """
    Bulkhead pattern for isolating failures.

    Limits concurrent executions to prevent resource exhaustion.

    Usage:
        bulkhead = Bulkhead("database", max_concurrent=10)

        async with bulkhead:
            result = await db.query(...)
    """

    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        max_queued: int = 100,
        timeout_seconds: float = 30.0
    ):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_queued = max_queued
        self.timeout_seconds = timeout_seconds

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue_size = 0
        self._active_count = 0
        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger(f"bulkhead.{name}")

        # Metrics
        self._total_calls = 0
        self._rejected_calls = 0
        self._timeout_calls = 0

    async def acquire(self) -> bool:
        """Acquire a slot in the bulkhead"""
        async with self._lock:
            if self._queue_size >= self.max_queued:
                self._rejected_calls += 1
                return False

            self._queue_size += 1

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.timeout_seconds
            )

            async with self._lock:
                self._queue_size -= 1
                self._active_count += 1
                self._total_calls += 1

            return True

        except asyncio.TimeoutError:
            async with self._lock:
                self._queue_size -= 1
                self._timeout_calls += 1
            return False

    async def release(self) -> None:
        """Release a slot in the bulkhead"""
        self._semaphore.release()

        async with self._lock:
            self._active_count -= 1

    async def __aenter__(self) -> "Bulkhead":
        if not await self.acquire():
            raise BulkheadFullError(f"Bulkhead '{self.name}' is full")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.release()

    def get_status(self) -> Dict[str, Any]:
        """Get bulkhead status"""
        return {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "max_queued": self.max_queued,
            "active_count": self._active_count,
            "queue_size": self._queue_size,
            "total_calls": self._total_calls,
            "rejected_calls": self._rejected_calls,
            "timeout_calls": self._timeout_calls
        }


class BulkheadFullError(Exception):
    """Raised when bulkhead is full"""
    pass


# =============================================================================
# FALLBACK HANDLER
# =============================================================================


class FallbackHandler(Generic[T]):
    """
    Fallback handler for providing alternative responses.

    Usage:
        fallback = FallbackHandler(
            primary=call_primary_service,
            fallback=call_backup_service,
            default=return_cached_response
        )

        result = await fallback.execute()
    """

    def __init__(
        self,
        primary: Callable[..., Awaitable[T]],
        fallback: Optional[Callable[..., Awaitable[T]]] = None,
        default: Optional[Callable[..., Awaitable[T]]] = None,
        cache: Optional[Dict[str, T]] = None
    ):
        self.primary = primary
        self.fallback = fallback
        self.default = default
        self.cache = cache or {}
        self._logger = structlog.get_logger("fallback_handler")

    async def execute(self, *args, cache_key: Optional[str] = None, **kwargs) -> T:
        """Execute with fallback chain"""
        # Try primary
        try:
            result = await self.primary(*args, **kwargs)

            # Cache successful result
            if cache_key:
                self.cache[cache_key] = result

            return result

        except Exception as primary_error:
            self._logger.warning("primary_failed", error=str(primary_error))

            # Try fallback
            if self.fallback:
                try:
                    return await self.fallback(*args, **kwargs)
                except Exception as fallback_error:
                    self._logger.warning("fallback_failed", error=str(fallback_error))

            # Try cache
            if cache_key and cache_key in self.cache:
                self._logger.info("returning_cached_value")
                return self.cache[cache_key]

            # Try default
            if self.default:
                try:
                    return await self.default(*args, **kwargs)
                except Exception as default_error:
                    self._logger.warning("default_failed", error=str(default_error))

            # Re-raise original error
            raise primary_error


# =============================================================================
# RATE LIMITER
# =============================================================================


class RateLimiter:
    """
    Token bucket rate limiter.

    Usage:
        limiter = RateLimiter(rate=100, burst=10)

        if await limiter.acquire():
            await make_request()
    """

    def __init__(
        self,
        rate: float,  # tokens per second
        burst: int = 1
    ):
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens"""
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_update

            # Refill tokens
            self._tokens = min(
                self.burst,
                self._tokens + elapsed * self.rate
            )
            self._last_update = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            return False

    async def wait(self, tokens: int = 1) -> None:
        """Wait until tokens are available"""
        while not await self.acquire(tokens):
            await asyncio.sleep(1 / self.rate)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    reset_timeout: float = 60.0,
    fallback: Optional[FallbackHandler] = None
) -> CircuitBreaker:
    """Create a circuit breaker with common defaults"""
    return CircuitBreaker(
        name=name,
        config=CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            reset_timeout_seconds=reset_timeout
        ),
        fallback=fallback
    )


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
) -> Callable:
    """Create a retry decorator with common defaults"""
    return retry(RetryPolicy(
        max_retries=max_retries,
        base_delay_seconds=base_delay,
        strategy=strategy
    ))
