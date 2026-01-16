"""
Circuit Breaker
===============

Circuit breaker pattern implementation for fault tolerance and
graceful degradation.

Author: Platform Engineering Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitOpenError(Exception):
    """Raised when circuit is open"""

    def __init__(
        self,
        message: str,
        circuit_name: str,
        retry_after: float,
    ):
        super().__init__(message)
        self.circuit_name = circuit_name
        self.retry_after = retry_after


@dataclass
class CircuitConfig:
    """Circuit breaker configuration"""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes to close from half-open
    timeout_seconds: float = 30.0  # Time in open state before half-open
    half_open_max_calls: int = 3  # Max calls in half-open state
    failure_rate_threshold: float = 0.5  # Failure rate to trigger open
    slow_call_threshold_seconds: float = 10.0  # Threshold for slow calls
    slow_call_rate_threshold: float = 0.5  # Slow call rate to trigger open
    window_size: int = 10  # Size of sliding window


@dataclass
class CircuitMetrics:
    """Metrics for a circuit breaker"""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    slow_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None
    current_state: CircuitState = CircuitState.CLOSED

    @property
    def failure_rate(self) -> float:
        """Get current failure rate"""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls

    @property
    def slow_call_rate(self) -> float:
        """Get slow call rate"""
        if self.total_calls == 0:
            return 0.0
        return self.slow_calls / self.total_calls

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "slow_calls": self.slow_calls,
            "rejected_calls": self.rejected_calls,
            "failure_rate": self.failure_rate,
            "slow_call_rate": self.slow_call_rate,
            "state_changes": self.state_changes,
            "current_state": self.current_state.value,
            "last_failure_time": (
                self.last_failure_time.isoformat()
                if self.last_failure_time
                else None
            ),
            "last_success_time": (
                self.last_success_time.isoformat()
                if self.last_success_time
                else None
            ),
        }


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service failing, requests rejected immediately
    - HALF_OPEN: Testing if service recovered

    Usage:
        breaker = CircuitBreaker("user_service")

        try:
            result = await breaker.call(async_function, arg1, arg2)
        except CircuitOpenError:
            # Handle circuit open
            return fallback_value
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitConfig()
        self._state = CircuitState.CLOSED
        self._metrics = CircuitMetrics()
        self._lock = threading.RLock()
        self._logger = structlog.get_logger(f"circuit.{name}")

        # Sliding window for tracking recent calls
        self._window: List[Dict[str, Any]] = []

        # Half-open state tracking
        self._half_open_calls = 0
        self._half_open_successes = 0

        # Timing
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> CircuitState:
        """Get current state"""
        with self._lock:
            self._check_state_transition()
            return self._state

    @property
    def metrics(self) -> CircuitMetrics:
        """Get metrics"""
        with self._lock:
            self._metrics.current_state = self._state
            return self._metrics

    async def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of the function

        Raises:
            CircuitOpenError: If circuit is open
        """
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.OPEN:
                self._metrics.rejected_calls += 1
                retry_after = self._get_retry_after()
                raise CircuitOpenError(
                    message=f"Circuit {self.name} is open",
                    circuit_name=self.name,
                    retry_after=retry_after,
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._metrics.rejected_calls += 1
                    raise CircuitOpenError(
                        message=f"Circuit {self.name} is half-open at capacity",
                        circuit_name=self.name,
                        retry_after=1.0,
                    )
                self._half_open_calls += 1

        # Execute the call
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            duration = time.time() - start_time
            self._record_success(duration)
            return result

        except Exception as e:
            duration = time.time() - start_time
            self._record_failure(e, duration)
            raise

    def _record_success(self, duration: float) -> None:
        """Record a successful call"""
        with self._lock:
            self._metrics.total_calls += 1
            self._metrics.successful_calls += 1
            self._metrics.last_success_time = datetime.utcnow()

            is_slow = duration > self.config.slow_call_threshold_seconds
            if is_slow:
                self._metrics.slow_calls += 1

            # Add to sliding window
            self._add_to_window(success=True, slow=is_slow)

            # Handle half-open state
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def _record_failure(self, error: Exception, duration: float) -> None:
        """Record a failed call"""
        with self._lock:
            self._metrics.total_calls += 1
            self._metrics.failed_calls += 1
            self._metrics.last_failure_time = datetime.utcnow()

            is_slow = duration > self.config.slow_call_threshold_seconds
            if is_slow:
                self._metrics.slow_calls += 1

            # Add to sliding window
            self._add_to_window(success=False, slow=is_slow)

            # Check if should open circuit
            if self._state == CircuitState.CLOSED:
                if self._should_open():
                    self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open returns to open
                self._transition_to(CircuitState.OPEN)

            self._logger.warning(
                "circuit_call_failed",
                circuit=self.name,
                error=str(error),
                state=self._state.value,
            )

    def _add_to_window(self, success: bool, slow: bool) -> None:
        """Add call to sliding window"""
        self._window.append({
            "success": success,
            "slow": slow,
            "timestamp": time.time(),
        })

        # Trim window
        while len(self._window) > self.config.window_size:
            self._window.pop(0)

    def _should_open(self) -> bool:
        """Check if circuit should open"""
        if len(self._window) < self.config.window_size:
            return False

        # Calculate failure rate in window
        failures = sum(1 for call in self._window if not call["success"])
        failure_rate = failures / len(self._window)

        if failure_rate >= self.config.failure_rate_threshold:
            return True

        # Check slow call rate
        slow_calls = sum(1 for call in self._window if call["slow"])
        slow_rate = slow_calls / len(self._window)

        if slow_rate >= self.config.slow_call_rate_threshold:
            return True

        return False

    def _check_state_transition(self) -> None:
        """Check for automatic state transitions"""
        if self._state == CircuitState.OPEN:
            if self._opened_at is not None:
                elapsed = time.time() - self._opened_at
                if elapsed >= self.config.timeout_seconds:
                    self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state"""
        old_state = self._state
        self._state = new_state
        self._metrics.state_changes += 1
        self._metrics.last_state_change = datetime.utcnow()

        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._half_open_successes = 0
        elif new_state == CircuitState.CLOSED:
            self._window.clear()
            self._opened_at = None

        self._logger.info(
            "circuit_state_change",
            circuit=self.name,
            old_state=old_state.value,
            new_state=new_state.value,
        )

    def _get_retry_after(self) -> float:
        """Get seconds until circuit might close"""
        if self._opened_at is None:
            return 0.0
        elapsed = time.time() - self._opened_at
        remaining = self.config.timeout_seconds - elapsed
        return max(0, remaining)

    def reset(self) -> None:
        """Force reset the circuit to closed state"""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._logger.info("circuit_reset", circuit=self.name)

    def force_open(self) -> None:
        """Force the circuit to open state"""
        with self._lock:
            self._transition_to(CircuitState.OPEN)
            self._logger.info("circuit_forced_open", circuit=self.name)


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Usage:
        registry = CircuitBreakerRegistry()
        breaker = registry.get("user_service")
        result = await breaker.call(fetch_user, user_id)
    """

    def __init__(self, default_config: Optional[CircuitConfig] = None):
        self._default_config = default_config or CircuitConfig()
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._configs: Dict[str, CircuitConfig] = {}
        self._lock = threading.Lock()
        self._logger = structlog.get_logger("circuit_registry")

    def get(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        with self._lock:
            if name not in self._breakers:
                config = self._configs.get(name, self._default_config)
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]

    def configure(self, name: str, config: CircuitConfig) -> None:
        """Configure a circuit breaker"""
        with self._lock:
            self._configs[name] = config
            # Update existing breaker if present
            if name in self._breakers:
                self._breakers[name].config = config

    def get_all_metrics(self) -> Dict[str, CircuitMetrics]:
        """Get metrics for all breakers"""
        with self._lock:
            return {
                name: breaker.metrics
                for name, breaker in self._breakers.items()
            }

    def reset_all(self) -> None:
        """Reset all circuit breakers"""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def get_open_circuits(self) -> List[str]:
        """Get names of open circuits"""
        with self._lock:
            return [
                name
                for name, breaker in self._breakers.items()
                if breaker.state == CircuitState.OPEN
            ]


# =============================================================================
# FALLBACK DECORATOR
# =============================================================================


def circuit_breaker(
    name: str,
    registry: Optional[CircuitBreakerRegistry] = None,
    fallback: Optional[Callable[..., Any]] = None,
) -> Callable:
    """
    Decorator for circuit breaker pattern.

    Usage:
        @circuit_breaker("user_service", fallback=get_cached_user)
        async def get_user(user_id: str):
            return await fetch_from_db(user_id)
    """
    _registry = registry or CircuitBreakerRegistry()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        breaker = _registry.get(name)

        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await breaker.call(func, *args, **kwargs)
            except CircuitOpenError:
                if fallback:
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args, **kwargs)
                    return fallback(*args, **kwargs)
                raise

        return wrapper

    return decorator


# =============================================================================
# BULKHEAD PATTERN
# =============================================================================


class Bulkhead:
    """
    Bulkhead pattern for limiting concurrent requests.

    Prevents a single service from consuming all resources.
    """

    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        max_waiting: int = 100,
        timeout_seconds: float = 30.0,
    ):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_waiting = max_waiting
        self.timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._waiting = 0
        self._lock = threading.Lock()
        self._logger = structlog.get_logger(f"bulkhead.{name}")

    async def acquire(self) -> bool:
        """Acquire a slot"""
        with self._lock:
            if self._waiting >= self.max_waiting:
                return False
            self._waiting += 1

        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.timeout_seconds,
            )
            return acquired
        except asyncio.TimeoutError:
            return False
        finally:
            with self._lock:
                self._waiting -= 1

    def release(self) -> None:
        """Release a slot"""
        self._semaphore.release()

    async def __aenter__(self) -> "Bulkhead":
        if not await self.acquire():
            raise RuntimeError(f"Bulkhead {self.name} at capacity")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()
