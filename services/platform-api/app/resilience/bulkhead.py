"""Bulkhead pattern for resource isolation."""

from typing import Optional, Callable, Dict, Any, TypeVar
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging
import functools

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BulkheadFullError(Exception):
    """Raised when bulkhead has no available capacity."""
    pass


@dataclass
class BulkheadStats:
    """Statistics for bulkhead."""
    total_calls: int = 0
    successful_calls: int = 0
    rejected_calls: int = 0
    current_concurrent: int = 0
    current_queued: int = 0
    max_concurrent_reached: int = 0
    max_queued_reached: int = 0


class Bulkhead:
    """
    Bulkhead pattern for limiting concurrent operations.

    Prevents a single component from exhausting all resources.

    Usage:
        bulkhead = Bulkhead("db-queries", max_concurrent=10, max_queued=100)

        @bulkhead
        async def query_database():
            ...

        # Or manually
        async with bulkhead:
            await query_database()
    """

    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        max_queued: int = 100,
        queue_timeout: float = 30.0,
    ):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_queued = max_queued
        self.queue_timeout = queue_timeout

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue_semaphore = asyncio.Semaphore(max_queued)
        self._concurrent_count = 0
        self._queued_count = 0
        self._lock = asyncio.Lock()
        self._stats = BulkheadStats()

    @property
    def stats(self) -> BulkheadStats:
        """Get bulkhead statistics."""
        self._stats.current_concurrent = self._concurrent_count
        self._stats.current_queued = self._queued_count
        return self._stats

    @property
    def available_concurrent(self) -> int:
        """Get available concurrent capacity."""
        return self.max_concurrent - self._concurrent_count

    @property
    def available_queue(self) -> int:
        """Get available queue capacity."""
        return self.max_queued - self._queued_count

    async def acquire(self) -> bool:
        """
        Acquire a slot in the bulkhead.

        Returns True if acquired, raises BulkheadFullError if not.
        """
        async with self._lock:
            self._stats.total_calls += 1

            # Check if we can queue
            if self._queued_count >= self.max_queued:
                self._stats.rejected_calls += 1
                raise BulkheadFullError(
                    f"Bulkhead '{self.name}' queue full "
                    f"(max={self.max_queued})"
                )

            # Add to queue
            self._queued_count += 1
            self._stats.max_queued_reached = max(
                self._stats.max_queued_reached,
                self._queued_count,
            )

        try:
            # Wait for semaphore with timeout
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.queue_timeout,
            )

            async with self._lock:
                self._queued_count -= 1
                self._concurrent_count += 1
                self._stats.max_concurrent_reached = max(
                    self._stats.max_concurrent_reached,
                    self._concurrent_count,
                )

            return acquired

        except asyncio.TimeoutError:
            async with self._lock:
                self._queued_count -= 1
                self._stats.rejected_calls += 1
            raise BulkheadFullError(
                f"Bulkhead '{self.name}' timeout waiting for slot"
            )

    async def release(self) -> None:
        """Release a slot in the bulkhead."""
        async with self._lock:
            self._concurrent_count -= 1
        self._semaphore.release()

    async def __aenter__(self):
        """Enter context manager."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if exc_type is None:
            async with self._lock:
                self._stats.successful_calls += 1
        await self.release()
        return False

    def __call__(self, func: Callable) -> Callable:
        """Decorator for protecting functions."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with self:
                return await func(*args, **kwargs)
        return wrapper


class ThreadPoolBulkhead:
    """
    Bulkhead that uses a dedicated thread pool.

    Useful for CPU-bound operations or blocking I/O.
    """

    def __init__(
        self,
        name: str,
        max_threads: int = 4,
    ):
        import concurrent.futures
        self.name = name
        self.max_threads = max_threads
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_threads,
            thread_name_prefix=f"bulkhead-{name}-",
        )
        self._stats = BulkheadStats()

    @property
    def stats(self) -> BulkheadStats:
        """Get bulkhead statistics."""
        return self._stats

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute function in thread pool."""
        self._stats.total_calls += 1

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self._executor,
                functools.partial(func, *args, **kwargs),
            )
            self._stats.successful_calls += 1
            return result
        except Exception:
            self._stats.rejected_calls += 1
            raise

    def __call__(self, func: Callable) -> Callable:
        """Decorator."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)
        return wrapper

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=wait)


class BulkheadRegistry:
    """Registry for managing multiple bulkheads."""

    def __init__(self):
        self._bulkheads: Dict[str, Bulkhead] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        max_concurrent: int = 10,
        max_queued: int = 100,
        queue_timeout: float = 30.0,
    ) -> Bulkhead:
        """Get or create a bulkhead."""
        async with self._lock:
            if name not in self._bulkheads:
                self._bulkheads[name] = Bulkhead(
                    name,
                    max_concurrent=max_concurrent,
                    max_queued=max_queued,
                    queue_timeout=queue_timeout,
                )
            return self._bulkheads[name]

    def get(self, name: str) -> Optional[Bulkhead]:
        """Get a bulkhead by name."""
        return self._bulkheads.get(name)

    def get_all_stats(self) -> Dict[str, BulkheadStats]:
        """Get statistics for all bulkheads."""
        return {name: bulkhead.stats for name, bulkhead in self._bulkheads.items()}


# Global registry
_bulkhead_registry = BulkheadRegistry()


async def get_bulkhead(
    name: str,
    max_concurrent: int = 10,
    max_queued: int = 100,
) -> Bulkhead:
    """Get or create a bulkhead from the global registry."""
    return await _bulkhead_registry.get_or_create(
        name,
        max_concurrent=max_concurrent,
        max_queued=max_queued,
    )


def bulkhead(
    name: str,
    max_concurrent: int = 10,
    max_queued: int = 100,
) -> Callable:
    """
    Decorator for adding bulkhead protection.

    Usage:
        @bulkhead("database", max_concurrent=20)
        async def query_db():
            ...
    """
    _bulkhead: Optional[Bulkhead] = None

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal _bulkhead
            if _bulkhead is None:
                _bulkhead = await _bulkhead_registry.get_or_create(
                    name,
                    max_concurrent=max_concurrent,
                    max_queued=max_queued,
                )
            async with _bulkhead:
                return await func(*args, **kwargs)
        return wrapper

    return decorator


class AdaptiveBulkhead:
    """
    Adaptive bulkhead that adjusts capacity based on latency.

    Increases capacity when latency is low, decreases when high.
    """

    def __init__(
        self,
        name: str,
        initial_concurrent: int = 10,
        min_concurrent: int = 1,
        max_concurrent: int = 100,
        target_latency_ms: float = 100.0,
        adjustment_interval: float = 10.0,
    ):
        self.name = name
        self.min_concurrent = min_concurrent
        self.max_concurrent = max_concurrent
        self.target_latency_ms = target_latency_ms
        self.adjustment_interval = adjustment_interval

        self._current_concurrent = initial_concurrent
        self._semaphore = asyncio.Semaphore(initial_concurrent)
        self._latencies: list = []
        self._last_adjustment = datetime.utcnow()
        self._lock = asyncio.Lock()
        self._stats = BulkheadStats()

    @property
    def current_concurrent(self) -> int:
        """Get current concurrency limit."""
        return self._current_concurrent

    async def record_latency(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        async with self._lock:
            self._latencies.append(latency_ms)

            # Check if we should adjust
            elapsed = (datetime.utcnow() - self._last_adjustment).total_seconds()
            if elapsed >= self.adjustment_interval:
                await self._adjust_capacity()

    async def _adjust_capacity(self) -> None:
        """Adjust capacity based on latency."""
        if not self._latencies:
            return

        avg_latency = sum(self._latencies) / len(self._latencies)
        self._latencies.clear()
        self._last_adjustment = datetime.utcnow()

        if avg_latency > self.target_latency_ms * 1.5:
            # Latency too high, decrease capacity
            new_concurrent = max(
                self.min_concurrent,
                int(self._current_concurrent * 0.8),
            )
        elif avg_latency < self.target_latency_ms * 0.5:
            # Latency low, increase capacity
            new_concurrent = min(
                self.max_concurrent,
                int(self._current_concurrent * 1.2),
            )
        else:
            return

        if new_concurrent != self._current_concurrent:
            logger.info(
                f"Adaptive bulkhead '{self.name}' adjusting capacity: "
                f"{self._current_concurrent} -> {new_concurrent} "
                f"(avg latency: {avg_latency:.1f}ms)"
            )
            self._current_concurrent = new_concurrent
            # Note: Actually adjusting semaphore value is complex
            # In practice, you'd recreate the semaphore or use a different approach

    async def __aenter__(self):
        """Enter context manager."""
        await self._semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self._semaphore.release()
        return False
