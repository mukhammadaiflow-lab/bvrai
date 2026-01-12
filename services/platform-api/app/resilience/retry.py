"""Retry patterns with exponential backoff."""

from typing import Optional, Callable, Tuple, Type, Any, TypeVar, Union, List
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import random
import logging
import functools

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(
        self,
        message: str,
        attempts: int,
        last_exception: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Exponential backoff base
    jitter: float = 0.1  # Random jitter factor (0-1)
    jitter_mode: str = "full"  # "full", "equal", "decorrelated"
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    non_retryable_exceptions: Tuple[Type[Exception], ...] = ()
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
    retry_condition: Optional[Callable[[Exception], bool]] = None


@dataclass
class RetryStats:
    """Statistics for retry operations."""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_delay: float = 0.0
    retries_exhausted: int = 0
    exceptions_by_type: dict = field(default_factory=dict)


class RetryPolicy:
    """
    Retry policy with exponential backoff and jitter.

    Supports multiple jitter modes:
    - full: delay = random(0, base_delay * exponential_base^attempt)
    - equal: delay = (base_delay * exponential_base^attempt) / 2 + random(0, base_delay * exponential_base^attempt / 2)
    - decorrelated: delay = random(base_delay, previous_delay * 3)

    Usage:
        policy = RetryPolicy(RetryConfig(max_attempts=5))

        @policy
        async def flaky_operation():
            ...

        # Or manually
        result = await policy.execute(flaky_operation)
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self._stats = RetryStats()
        self._previous_delay = self.config.base_delay

    @property
    def stats(self) -> RetryStats:
        """Get retry statistics."""
        return self._stats

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        if self.config.jitter_mode == "decorrelated":
            # Decorrelated jitter: delay = random(base, prev * 3)
            delay = random.uniform(
                self.config.base_delay,
                self._previous_delay * 3,
            )
            self._previous_delay = delay
        elif self.config.jitter_mode == "equal":
            # Equal jitter: delay = half + random(0, half)
            base = self.config.base_delay * (self.config.exponential_base ** attempt)
            half = base / 2
            delay = half + random.uniform(0, half)
        else:
            # Full jitter: delay = random(0, base * 2^attempt)
            base = self.config.base_delay * (self.config.exponential_base ** attempt)
            delay = random.uniform(0, base)

        # Apply jitter factor
        jitter = random.uniform(-self.config.jitter, self.config.jitter)
        delay = delay * (1 + jitter)

        # Clamp to max delay
        return min(delay, self.config.max_delay)

    def should_retry(self, exception: Exception) -> bool:
        """Check if exception should trigger a retry."""
        # Check non-retryable exceptions first
        if isinstance(exception, self.config.non_retryable_exceptions):
            return False

        # Check custom condition
        if self.config.retry_condition:
            return self.config.retry_condition(exception)

        # Check retryable exceptions
        return isinstance(exception, self.config.retryable_exceptions)

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute function with retry policy."""
        last_exception: Optional[Exception] = None

        for attempt in range(self.config.max_attempts):
            self._stats.total_attempts += 1

            try:
                result = await func(*args, **kwargs)
                self._stats.successful_attempts += 1
                return result

            except Exception as e:
                last_exception = e
                self._stats.failed_attempts += 1

                # Track exception type
                exc_name = type(e).__name__
                self._stats.exceptions_by_type[exc_name] = \
                    self._stats.exceptions_by_type.get(exc_name, 0) + 1

                # Check if we should retry
                if not self.should_retry(e):
                    logger.warning(
                        f"Non-retryable exception on attempt {attempt + 1}: {e}"
                    )
                    raise

                # Check if we have more attempts
                if attempt >= self.config.max_attempts - 1:
                    logger.error(
                        f"Retry exhausted after {attempt + 1} attempts: {e}"
                    )
                    self._stats.retries_exhausted += 1
                    raise RetryExhausted(
                        f"Failed after {attempt + 1} attempts",
                        attempts=attempt + 1,
                        last_exception=e,
                    )

                # Calculate delay
                delay = self.calculate_delay(attempt)
                self._stats.total_delay += delay

                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                # Call on_retry callback
                if self.config.on_retry:
                    self.config.on_retry(attempt + 1, e, delay)

                # Wait before retrying
                await asyncio.sleep(delay)

        # Should never reach here, but just in case
        raise RetryExhausted(
            "Retry exhausted",
            attempts=self.config.max_attempts,
            last_exception=last_exception,
        )

    def __call__(self, func: Callable) -> Callable:
        """Decorator for adding retry to functions."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)
        return wrapper


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: float = 0.1,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    non_retryable_exceptions: Tuple[Type[Exception], ...] = (),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Callable:
    """
    Decorator for adding retry behavior to async functions.

    Usage:
        @retry(max_attempts=5, base_delay=2.0)
        async def flaky_operation():
            ...
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions,
        non_retryable_exceptions=non_retryable_exceptions,
        on_retry=on_retry,
    )
    policy = RetryPolicy(config)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await policy.execute(func, *args, **kwargs)
        return wrapper

    return decorator


class RetryWithCircuitBreaker:
    """
    Combined retry and circuit breaker pattern.

    Usage:
        from app.resilience.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker("my-service")
        retry_cb = RetryWithCircuitBreaker(
            retry_config=RetryConfig(max_attempts=3),
            circuit_breaker=cb,
        )

        @retry_cb
        async def call_service():
            ...
    """

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker: Optional[Any] = None,  # CircuitBreaker type
    ):
        from app.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError

        self.retry_policy = RetryPolicy(retry_config)
        self.circuit_breaker = circuit_breaker
        self._circuit_breaker_open_error = CircuitBreakerOpenError

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute with retry and circuit breaker."""
        if self.circuit_breaker:
            async with self.circuit_breaker:
                return await self.retry_policy.execute(func, *args, **kwargs)
        return await self.retry_policy.execute(func, *args, **kwargs)

    def __call__(self, func: Callable) -> Callable:
        """Decorator."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)
        return wrapper


class AdaptiveRetry:
    """
    Adaptive retry that adjusts based on success rate.

    When success rate is high, uses fewer retries.
    When success rate is low, uses more retries with longer delays.
    """

    def __init__(
        self,
        min_attempts: int = 1,
        max_attempts: int = 5,
        window_size: int = 100,
        success_threshold: float = 0.9,
        failure_threshold: float = 0.5,
    ):
        self.min_attempts = min_attempts
        self.max_attempts = max_attempts
        self.window_size = window_size
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
        self._results: List[bool] = []
        self._current_attempts = min_attempts
        self._lock = asyncio.Lock()

    @property
    def current_attempts(self) -> int:
        """Get current number of attempts."""
        return self._current_attempts

    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        if not self._results:
            return 1.0
        return sum(self._results) / len(self._results)

    async def record_result(self, success: bool) -> None:
        """Record result and adjust attempts."""
        async with self._lock:
            self._results.append(success)

            # Keep only window_size results
            if len(self._results) > self.window_size:
                self._results = self._results[-self.window_size:]

            # Adjust attempts based on success rate
            rate = self.success_rate
            if rate >= self.success_threshold and self._current_attempts > self.min_attempts:
                self._current_attempts -= 1
            elif rate <= self.failure_threshold and self._current_attempts < self.max_attempts:
                self._current_attempts += 1

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute with adaptive retry."""
        config = RetryConfig(max_attempts=self._current_attempts)
        policy = RetryPolicy(config)

        try:
            result = await policy.execute(func, *args, **kwargs)
            await self.record_result(True)
            return result
        except Exception as e:
            await self.record_result(False)
            raise

    def __call__(self, func: Callable) -> Callable:
        """Decorator."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)
        return wrapper


# Predefined retry policies
QUICK_RETRY = RetryConfig(
    max_attempts=3,
    base_delay=0.5,
    max_delay=5.0,
)

STANDARD_RETRY = RetryConfig(
    max_attempts=5,
    base_delay=1.0,
    max_delay=30.0,
)

AGGRESSIVE_RETRY = RetryConfig(
    max_attempts=10,
    base_delay=2.0,
    max_delay=120.0,
)

# Network-specific retry
NETWORK_RETRY = RetryConfig(
    max_attempts=5,
    base_delay=1.0,
    max_delay=30.0,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,
    ),
)
