"""
Circuit Breaker Pattern Implementation.

This module provides a production-grade circuit breaker for protecting
external service calls with automatic fallback and recovery.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """State of the circuit breaker."""

    CLOSED = "closed"       # Normal operation, requests pass through
    OPEN = "open"           # Failing, requests are rejected
    HALF_OPEN = "half_open" # Testing recovery, limited requests allowed


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""

    state: CircuitState
    failure_count: int = 0
    success_count: int = 0
    total_requests: int = 0
    rejected_requests: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    last_state_change: float = field(default_factory=time.time)
    consecutive_successes: int = 0
    consecutive_failures: int = 0

    # Timing
    total_response_time_ms: float = 0.0
    avg_response_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "rejected_requests": self.rejected_requests,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "last_state_change": self.last_state_change,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "avg_response_time_ms": round(self.avg_response_time_ms, 2),
            "failure_rate": round(
                self.failure_count / self.total_requests * 100
                if self.total_requests > 0 else 0,
                2
            ),
        }


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    # Thresholds
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes to close from half-open
    timeout_threshold_ms: float = 5000  # Timeout to count as failure

    # Timing
    recovery_timeout_s: float = 30.0    # Time in open state before half-open
    half_open_max_calls: int = 3        # Max calls allowed in half-open

    # Sliding window (for more sophisticated failure counting)
    window_size_s: float = 60.0         # Sliding window for failure rate
    failure_rate_threshold: float = 0.5  # Failure rate to trip circuit

    # Exceptions
    excluded_exceptions: List[type] = field(default_factory=list)


class CircuitBreakerError(Exception):
    """Exception raised when circuit is open."""

    def __init__(self, circuit_name: str, state: CircuitState, retry_after: float):
        self.circuit_name = circuit_name
        self.state = state
        self.retry_after = retry_after
        super().__init__(
            f"Circuit '{circuit_name}' is {state.value}. Retry after {retry_after:.1f}s"
        )


class CircuitBreaker:
    """
    Production-grade circuit breaker implementation.

    Features:
    - Three states: CLOSED, OPEN, HALF_OPEN
    - Configurable failure threshold
    - Automatic recovery with half-open testing
    - Sliding window failure rate calculation
    - Excluded exceptions support
    - Async-first design
    - Statistics and monitoring
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[Callable[..., Any]] = None,
        on_state_change: Optional[Callable[[str, CircuitState, CircuitState], None]] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Unique identifier for this circuit
            config: Configuration options
            fallback: Optional fallback function to call when circuit is open
            on_state_change: Optional callback when state changes
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._fallback = fallback
        self._on_state_change = on_state_change

        # State
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats(state=CircuitState.CLOSED)

        # Half-open tracking
        self._half_open_calls = 0

        # Sliding window for failure rate
        self._call_history: List[tuple[float, bool]] = []  # (timestamp, success)

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info(f"Circuit breaker '{name}' initialized")

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get current statistics."""
        return self._stats

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing)."""
        return self._state == CircuitState.OPEN

    async def _check_state(self) -> None:
        """Check and potentially update state based on timeouts."""
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            time_in_open = time.time() - self._stats.last_state_change
            if time_in_open >= self.config.recovery_timeout_s:
                await self._transition_to(CircuitState.HALF_OPEN)

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        if new_state == self._state:
            return

        old_state = self._state
        self._state = new_state
        self._stats.state = new_state
        self._stats.last_state_change = time.time()

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0

        logger.info(
            f"Circuit '{self.name}' state change: {old_state.value} -> {new_state.value}"
        )

        if self._on_state_change:
            try:
                self._on_state_change(self.name, old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

    async def _record_success(self, response_time_ms: float) -> None:
        """Record a successful call."""
        async with self._lock:
            self._stats.success_count += 1
            self._stats.total_requests += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = time.time()

            # Update response time
            self._stats.total_response_time_ms += response_time_ms
            self._stats.avg_response_time_ms = (
                self._stats.total_response_time_ms / self._stats.success_count
            )

            # Add to history
            self._call_history.append((time.time(), True))
            self._cleanup_history()

            # State transitions on success
            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)

    async def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        async with self._lock:
            self._stats.failure_count += 1
            self._stats.total_requests += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = time.time()

            # Add to history
            self._call_history.append((time.time(), False))
            self._cleanup_history()

            # State transitions on failure
            if self._state == CircuitState.CLOSED:
                # Check consecutive failures
                if self._stats.consecutive_failures >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)
                # Or check failure rate
                elif self._calculate_failure_rate() >= self.config.failure_rate_threshold:
                    await self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                await self._transition_to(CircuitState.OPEN)

    def _cleanup_history(self) -> None:
        """Remove old entries from call history."""
        cutoff = time.time() - self.config.window_size_s
        self._call_history = [
            (ts, success) for ts, success in self._call_history
            if ts > cutoff
        ]

    def _calculate_failure_rate(self) -> float:
        """Calculate failure rate from sliding window."""
        if not self._call_history:
            return 0.0

        failures = sum(1 for _, success in self._call_history if not success)
        return failures / len(self._call_history)

    def _is_excluded_exception(self, error: Exception) -> bool:
        """Check if exception should be excluded from failure counting."""
        return any(
            isinstance(error, exc_type)
            for exc_type in self.config.excluded_exceptions
        )

    async def call(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            func: The function to execute (can be async)
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function or fallback

        Raises:
            CircuitBreakerError: If circuit is open and no fallback
        """
        # Check state transitions
        await self._check_state()

        # Check if call is allowed
        if self._state == CircuitState.OPEN:
            self._stats.rejected_requests += 1
            retry_after = (
                self.config.recovery_timeout_s -
                (time.time() - self._stats.last_state_change)
            )

            if self._fallback:
                logger.debug(f"Circuit '{self.name}' open, using fallback")
                return await self._execute_fallback(*args, **kwargs)

            raise CircuitBreakerError(self.name, self._state, max(0, retry_after))

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.config.half_open_max_calls:
                self._stats.rejected_requests += 1
                if self._fallback:
                    return await self._execute_fallback(*args, **kwargs)
                raise CircuitBreakerError(
                    self.name,
                    self._state,
                    self.config.recovery_timeout_s,
                )
            self._half_open_calls += 1

        # Execute the call
        start_time = time.time()

        try:
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout_threshold_ms / 1000,
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: func(*args, **kwargs)
                )

            response_time_ms = (time.time() - start_time) * 1000
            await self._record_success(response_time_ms)

            return result

        except asyncio.TimeoutError:
            logger.warning(f"Circuit '{self.name}' call timed out")
            await self._record_failure(TimeoutError("Call timed out"))

            if self._fallback:
                return await self._execute_fallback(*args, **kwargs)
            raise

        except Exception as e:
            if self._is_excluded_exception(e):
                # Don't count excluded exceptions as failures
                logger.debug(f"Circuit '{self.name}' excluded exception: {type(e).__name__}")
                raise

            logger.warning(f"Circuit '{self.name}' call failed: {type(e).__name__}: {e}")
            await self._record_failure(e)

            if self._fallback:
                return await self._execute_fallback(*args, **kwargs)
            raise

    async def _execute_fallback(self, *args, **kwargs) -> Any:
        """Execute the fallback function."""
        if not self._fallback:
            return None

        try:
            if asyncio.iscoroutinefunction(self._fallback):
                return await self._fallback(*args, **kwargs)
            else:
                return self._fallback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Circuit '{self.name}' fallback failed: {e}")
            raise

    async def force_open(self) -> None:
        """Force the circuit to open state."""
        await self._transition_to(CircuitState.OPEN)

    async def force_close(self) -> None:
        """Force the circuit to closed state."""
        await self._transition_to(CircuitState.CLOSED)
        self._stats.consecutive_failures = 0

    async def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        async with self._lock:
            await self._transition_to(CircuitState.CLOSED)
            self._stats = CircuitBreakerStats(state=CircuitState.CLOSED)
            self._half_open_calls = 0
            self._call_history.clear()


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized management, monitoring, and configuration
    of circuit breakers across the application.
    """

    def __init__(self):
        self._circuits: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[Callable] = None,
        on_state_change: Optional[Callable] = None,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        async with self._lock:
            if name not in self._circuits:
                self._circuits[name] = CircuitBreaker(
                    name=name,
                    config=config,
                    fallback=fallback,
                    on_state_change=on_state_change,
                )
            return self._circuits[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._circuits.get(name)

    async def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all circuit breakers."""
        return {
            name: circuit.stats.to_dict()
            for name, circuit in self._circuits.items()
        }

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for circuit in self._circuits.values():
            await circuit.reset()

    def get_prometheus_metrics(self) -> str:
        """Export all circuit breakers as Prometheus metrics."""
        lines = []

        lines.append("# HELP circuit_breaker_state Current state of circuit breaker (0=closed, 1=open, 2=half_open)")
        lines.append("# TYPE circuit_breaker_state gauge")

        for name, circuit in self._circuits.items():
            state_value = {
                CircuitState.CLOSED: 0,
                CircuitState.OPEN: 1,
                CircuitState.HALF_OPEN: 2,
            }[circuit.state]
            lines.append(f'circuit_breaker_state{{name="{name}"}} {state_value}')

        lines.append("")
        lines.append("# HELP circuit_breaker_failures_total Total failures")
        lines.append("# TYPE circuit_breaker_failures_total counter")

        for name, circuit in self._circuits.items():
            lines.append(
                f'circuit_breaker_failures_total{{name="{name}"}} {circuit.stats.failure_count}'
            )

        lines.append("")
        lines.append("# HELP circuit_breaker_successes_total Total successes")
        lines.append("# TYPE circuit_breaker_successes_total counter")

        for name, circuit in self._circuits.items():
            lines.append(
                f'circuit_breaker_successes_total{{name="{name}"}} {circuit.stats.success_count}'
            )

        return "\n".join(lines)


# Decorator for easy circuit breaker usage
def circuit_protected(
    circuit_name: str,
    config: Optional[CircuitBreakerConfig] = None,
    fallback: Optional[Callable] = None,
    registry: Optional[CircuitBreakerRegistry] = None,
):
    """
    Decorator to protect a function with a circuit breaker.

    Usage:
        @circuit_protected("my_service")
        async def call_external_service():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Create or get circuit breaker
        _registry = registry or CircuitBreakerRegistry()
        _circuit: Optional[CircuitBreaker] = None

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            nonlocal _circuit

            if _circuit is None:
                _circuit = await _registry.get_or_create(
                    name=circuit_name,
                    config=config,
                    fallback=fallback,
                )

            return await _circuit.call(func, *args, **kwargs)

        return wrapper

    return decorator


# Global registry instance
default_registry = CircuitBreakerRegistry()
