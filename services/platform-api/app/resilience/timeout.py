"""
Timeout Management

Advanced timeout handling with:
- Configurable timeouts
- Adaptive timeout adjustment
- Cascading timeouts
- Deadline propagation
"""

from typing import Optional, Callable, Dict, Any, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from collections import deque
import asyncio
import time
import logging
import statistics

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TimeoutExceeded(Exception):
    """Raised when operation times out."""

    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        operation: Optional[str] = None,
        elapsed: Optional[float] = None,
    ):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        self.elapsed = elapsed


@dataclass
class TimeoutConfig:
    """Configuration for timeout handling."""
    default_timeout: float = 30.0  # Default timeout in seconds
    connect_timeout: float = 10.0  # Connection timeout
    read_timeout: float = 30.0  # Read timeout
    write_timeout: float = 30.0  # Write timeout
    total_timeout: float = 60.0  # Total operation timeout

    # Adaptive timeout settings
    enable_adaptive: bool = False
    min_timeout: float = 1.0
    max_timeout: float = 120.0
    percentile: float = 95.0  # Target percentile
    multiplier: float = 1.5  # Multiply percentile by this
    window_size: int = 100  # Number of samples


@dataclass
class TimeoutStats:
    """Statistics for timeout operations."""
    total_operations: int = 0
    successful_operations: int = 0
    timed_out_operations: int = 0
    total_elapsed_time: float = 0.0
    avg_elapsed_time: float = 0.0
    p50_elapsed_time: float = 0.0
    p95_elapsed_time: float = 0.0
    p99_elapsed_time: float = 0.0


class TimeoutManager:
    """
    Centralized timeout management.

    Provides consistent timeout handling across the application
    with support for different timeout types and adaptive adjustment.
    """

    def __init__(self, config: Optional[TimeoutConfig] = None):
        self.config = config or TimeoutConfig()
        self._operation_times: Dict[str, deque] = {}
        self._adaptive_timeouts: Dict[str, float] = {}
        self._stats: Dict[str, TimeoutStats] = {}
        self._lock = asyncio.Lock()

    def _get_timeout(self, operation: str, override: Optional[float] = None) -> float:
        """Get timeout for operation."""
        if override is not None:
            return override

        if self.config.enable_adaptive and operation in self._adaptive_timeouts:
            return self._adaptive_timeouts[operation]

        return self.config.default_timeout

    async def _record_completion(
        self,
        operation: str,
        elapsed: float,
        success: bool,
    ) -> None:
        """Record operation completion for adaptive timeout."""
        async with self._lock:
            # Initialize if needed
            if operation not in self._operation_times:
                self._operation_times[operation] = deque(maxlen=self.config.window_size)
                self._stats[operation] = TimeoutStats()

            # Record time
            self._operation_times[operation].append(elapsed)

            # Update stats
            stats = self._stats[operation]
            stats.total_operations += 1
            stats.total_elapsed_time += elapsed

            if success:
                stats.successful_operations += 1
            else:
                stats.timed_out_operations += 1

            # Calculate stats
            times = list(self._operation_times[operation])
            if times:
                stats.avg_elapsed_time = statistics.mean(times)
                stats.p50_elapsed_time = statistics.median(times)
                if len(times) >= 20:
                    sorted_times = sorted(times)
                    p95_idx = int(len(sorted_times) * 0.95)
                    p99_idx = int(len(sorted_times) * 0.99)
                    stats.p95_elapsed_time = sorted_times[p95_idx]
                    stats.p99_elapsed_time = sorted_times[p99_idx]

            # Update adaptive timeout
            if self.config.enable_adaptive and len(times) >= 10:
                sorted_times = sorted(times)
                percentile_idx = int(len(sorted_times) * (self.config.percentile / 100))
                percentile_value = sorted_times[min(percentile_idx, len(sorted_times) - 1)]
                new_timeout = percentile_value * self.config.multiplier

                # Clamp to min/max
                new_timeout = max(self.config.min_timeout, min(self.config.max_timeout, new_timeout))
                self._adaptive_timeouts[operation] = new_timeout

    async def execute(
        self,
        operation: str,
        func: Callable[..., T],
        *args,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> T:
        """Execute operation with timeout."""
        timeout_value = self._get_timeout(operation, timeout)
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout_value,
            )
            elapsed = time.time() - start_time
            await self._record_completion(operation, elapsed, success=True)
            return result

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            await self._record_completion(operation, elapsed, success=False)
            raise TimeoutExceeded(
                f"Operation '{operation}' timed out after {timeout_value}s",
                timeout_seconds=timeout_value,
                operation=operation,
                elapsed=elapsed,
            )

    def timeout(
        self,
        operation: Optional[str] = None,
        seconds: Optional[float] = None,
    ) -> Callable:
        """Decorator for adding timeout to functions."""
        def decorator(func: Callable) -> Callable:
            op_name = operation or func.__name__

            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await self.execute(
                    op_name,
                    func,
                    *args,
                    timeout=seconds,
                    **kwargs,
                )

            return wrapper
        return decorator

    def get_stats(self, operation: Optional[str] = None) -> Dict[str, TimeoutStats]:
        """Get timeout statistics."""
        if operation:
            return {operation: self._stats.get(operation, TimeoutStats())}
        return dict(self._stats)

    def get_adaptive_timeout(self, operation: str) -> Optional[float]:
        """Get current adaptive timeout for operation."""
        return self._adaptive_timeouts.get(operation)


class AdaptiveTimeout:
    """
    Adaptive timeout that adjusts based on observed latencies.

    Uses percentile-based adjustment to set appropriate timeouts
    based on historical data.
    """

    def __init__(
        self,
        initial_timeout: float = 30.0,
        min_timeout: float = 1.0,
        max_timeout: float = 120.0,
        percentile: float = 95.0,
        multiplier: float = 1.5,
        window_size: int = 100,
        adjustment_interval: float = 10.0,
    ):
        """
        Args:
            initial_timeout: Starting timeout value
            min_timeout: Minimum allowed timeout
            max_timeout: Maximum allowed timeout
            percentile: Target percentile for timeout
            multiplier: Multiply percentile by this
            window_size: Number of samples to keep
            adjustment_interval: Seconds between adjustments
        """
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.percentile = percentile
        self.multiplier = multiplier
        self.window_size = window_size
        self.adjustment_interval = adjustment_interval

        self._current_timeout = initial_timeout
        self._latencies: deque = deque(maxlen=window_size)
        self._last_adjustment = time.time()
        self._lock = asyncio.Lock()

        # Stats
        self._total_operations = 0
        self._timeouts = 0

    @property
    def current_timeout(self) -> float:
        """Get current timeout value."""
        return self._current_timeout

    async def record_latency(self, latency: float) -> None:
        """Record a latency measurement."""
        async with self._lock:
            self._latencies.append(latency)
            self._total_operations += 1

            # Check if we should adjust
            if time.time() - self._last_adjustment >= self.adjustment_interval:
                self._adjust_timeout()

    async def record_timeout(self) -> None:
        """Record a timeout occurrence."""
        async with self._lock:
            self._timeouts += 1
            self._total_operations += 1

            # Immediate increase on timeout
            new_timeout = min(self.max_timeout, self._current_timeout * 1.5)
            if new_timeout != self._current_timeout:
                logger.info(f"Increasing timeout after timeout: {self._current_timeout:.2f}s -> {new_timeout:.2f}s")
                self._current_timeout = new_timeout

    def _adjust_timeout(self) -> None:
        """Adjust timeout based on latencies."""
        if len(self._latencies) < 10:
            return

        self._last_adjustment = time.time()

        # Calculate percentile
        sorted_latencies = sorted(self._latencies)
        percentile_idx = int(len(sorted_latencies) * (self.percentile / 100))
        percentile_value = sorted_latencies[min(percentile_idx, len(sorted_latencies) - 1)]

        # Calculate new timeout
        new_timeout = percentile_value * self.multiplier
        new_timeout = max(self.min_timeout, min(self.max_timeout, new_timeout))

        if abs(new_timeout - self._current_timeout) / self._current_timeout > 0.1:
            logger.info(
                f"Adjusting adaptive timeout: {self._current_timeout:.2f}s -> {new_timeout:.2f}s "
                f"(p{self.percentile}: {percentile_value:.2f}s)"
            )
            self._current_timeout = new_timeout

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute function with adaptive timeout."""
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self._current_timeout,
            )
            elapsed = time.time() - start_time
            await self.record_latency(elapsed)
            return result

        except asyncio.TimeoutError:
            await self.record_timeout()
            raise TimeoutExceeded(
                f"Operation timed out after {self._current_timeout}s",
                timeout_seconds=self._current_timeout,
                elapsed=time.time() - start_time,
            )

    def __call__(self, func: Callable) -> Callable:
        """Decorator."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)
        return wrapper

    def get_stats(self) -> Dict[str, Any]:
        """Get timeout statistics."""
        latencies = list(self._latencies)
        return {
            "current_timeout": self._current_timeout,
            "total_operations": self._total_operations,
            "timeouts": self._timeouts,
            "timeout_rate": self._timeouts / self._total_operations if self._total_operations > 0 else 0,
            "avg_latency": statistics.mean(latencies) if latencies else 0,
            "p50_latency": statistics.median(latencies) if latencies else 0,
            "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else 0,
            "sample_count": len(latencies),
        }


class DeadlineContext:
    """
    Deadline-based timeout that propagates through the call stack.

    Useful for ensuring operations complete within a total time budget,
    even when making multiple nested calls.
    """

    def __init__(self, deadline: datetime):
        self.deadline = deadline
        self._start_time = datetime.utcnow()

    @classmethod
    def from_timeout(cls, timeout_seconds: float) -> "DeadlineContext":
        """Create deadline from timeout."""
        return cls(datetime.utcnow() + timedelta(seconds=timeout_seconds))

    @property
    def remaining(self) -> float:
        """Get remaining time until deadline."""
        remaining = (self.deadline - datetime.utcnow()).total_seconds()
        return max(0, remaining)

    @property
    def is_exceeded(self) -> bool:
        """Check if deadline has passed."""
        return datetime.utcnow() >= self.deadline

    @property
    def elapsed(self) -> float:
        """Get elapsed time since start."""
        return (datetime.utcnow() - self._start_time).total_seconds()

    def check(self) -> None:
        """Check if deadline is exceeded and raise if so."""
        if self.is_exceeded:
            raise TimeoutExceeded(
                "Deadline exceeded",
                timeout_seconds=(self.deadline - self._start_time).total_seconds(),
                elapsed=self.elapsed,
            )

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute with deadline timeout."""
        remaining = self.remaining
        if remaining <= 0:
            raise TimeoutExceeded(
                "Deadline already exceeded",
                timeout_seconds=0,
                elapsed=self.elapsed,
            )

        try:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=remaining,
            )
        except asyncio.TimeoutError:
            raise TimeoutExceeded(
                "Deadline exceeded during operation",
                timeout_seconds=(self.deadline - self._start_time).total_seconds(),
                elapsed=self.elapsed,
            )


class CascadingTimeout:
    """
    Timeout that cascades through dependent operations.

    Allocates timeout budget to different phases of an operation.
    """

    def __init__(
        self,
        total_timeout: float,
        phases: Dict[str, float],
    ):
        """
        Args:
            total_timeout: Total timeout budget
            phases: Dict of phase_name -> percentage of total (should sum to 1.0)
        """
        self.total_timeout = total_timeout
        self.phases = phases
        self._start_time: Optional[float] = None
        self._phase_times: Dict[str, float] = {}

    def start(self) -> None:
        """Start the timeout tracking."""
        self._start_time = time.time()

    def get_phase_timeout(self, phase: str) -> float:
        """Get timeout for a specific phase."""
        if phase not in self.phases:
            raise ValueError(f"Unknown phase: {phase}")

        # Calculate remaining time
        if self._start_time is None:
            remaining = self.total_timeout
        else:
            elapsed = time.time() - self._start_time
            remaining = self.total_timeout - elapsed

        # Get phase allocation
        phase_allocation = self.phases[phase] * self.total_timeout

        # Return minimum of allocation and remaining
        return max(0.1, min(phase_allocation, remaining))

    async def execute_phase(
        self,
        phase: str,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute a phase with its allocated timeout."""
        if self._start_time is None:
            self.start()

        timeout = self.get_phase_timeout(phase)
        phase_start = time.time()

        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout,
            )
            self._phase_times[phase] = time.time() - phase_start
            return result

        except asyncio.TimeoutError:
            self._phase_times[phase] = time.time() - phase_start
            raise TimeoutExceeded(
                f"Phase '{phase}' timed out",
                timeout_seconds=timeout,
                operation=phase,
                elapsed=self._phase_times[phase],
            )


# Decorator factory
def timeout(seconds: float, operation: Optional[str] = None) -> Callable:
    """
    Simple timeout decorator.

    Usage:
        @timeout(30.0)
        async def slow_operation():
            ...
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__

        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds,
                )
            except asyncio.TimeoutError:
                raise TimeoutExceeded(
                    f"Operation '{op_name}' timed out after {seconds}s",
                    timeout_seconds=seconds,
                    operation=op_name,
                )

        return wrapper
    return decorator


async def with_timeout(
    func: Callable[..., T],
    timeout_seconds: float,
    *args,
    operation: Optional[str] = None,
    **kwargs,
) -> T:
    """
    Execute async function with timeout.

    Usage:
        result = await with_timeout(slow_operation, 30.0, arg1, arg2)
    """
    try:
        return await asyncio.wait_for(
            func(*args, **kwargs),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise TimeoutExceeded(
            f"Operation timed out after {timeout_seconds}s",
            timeout_seconds=timeout_seconds,
            operation=operation or (func.__name__ if hasattr(func, "__name__") else "unknown"),
        )


# Context variable for deadline propagation
import contextvars

_deadline_context: contextvars.ContextVar[Optional[DeadlineContext]] = contextvars.ContextVar(
    "_deadline_context",
    default=None,
)


def get_current_deadline() -> Optional[DeadlineContext]:
    """Get current deadline from context."""
    return _deadline_context.get()


def set_deadline(deadline: DeadlineContext) -> contextvars.Token:
    """Set deadline in context."""
    return _deadline_context.set(deadline)


def clear_deadline(token: contextvars.Token) -> None:
    """Clear deadline from context."""
    _deadline_context.reset(token)
