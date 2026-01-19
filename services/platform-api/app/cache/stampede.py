"""
Cache Stampede Protection

Prevents cache stampede (thundering herd) with:
- Probabilistic early expiration
- Locking strategies
- Request coalescing
- Semaphore-based limiting
"""

from typing import Optional, Any, Callable, Awaitable, Dict, TypeVar
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import random
import logging
import hashlib
import time

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CachedValue:
    """Cached value with metadata for stampede protection."""
    value: Any
    created_at: datetime
    expires_at: datetime
    compute_time_ms: float = 0.0
    access_count: int = 0

    @property
    def ttl_remaining(self) -> float:
        """Get remaining TTL in seconds."""
        return max(0, (self.expires_at - datetime.utcnow()).total_seconds())

    @property
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()


class ProbabilisticEarlyExpiration:
    """
    XFetch algorithm for probabilistic early expiration.

    Recomputes value before expiration with probability that
    increases as TTL approaches zero.

    This prevents multiple clients from recomputing simultaneously
    when a cached value expires.
    """

    def __init__(
        self,
        beta: float = 1.0,
        min_remaining_ttl: float = 5.0,
    ):
        """
        Args:
            beta: Controls early expiration aggressiveness (higher = earlier)
            min_remaining_ttl: Minimum TTL before considering early refresh
        """
        self.beta = beta
        self.min_remaining_ttl = min_remaining_ttl

    def should_recompute(
        self,
        cached: CachedValue,
    ) -> bool:
        """
        Determine if value should be recomputed.

        Uses XFetch algorithm:
        recompute if: current_time - (compute_time * beta * log(random)) >= expiry
        """
        ttl_remaining = cached.ttl_remaining

        # Don't early-refresh if still has plenty of time
        if ttl_remaining > self.min_remaining_ttl:
            return False

        # Calculate recompute probability
        # delta = compute_time * beta * -log(random)
        # Recompute if: now + delta >= expiry
        # Which is: delta >= ttl_remaining

        compute_time_sec = cached.compute_time_ms / 1000.0
        random_factor = -1 * self.beta * random.expovariate(1)
        delta = compute_time_sec * random_factor

        return delta >= ttl_remaining

    async def get_or_compute(
        self,
        cache: Any,
        key: str,
        compute: Callable[[], Awaitable[T]],
        ttl: int,
        use_pickle: bool = False,
    ) -> T:
        """
        Get value with probabilistic early expiration.

        If the cached value is approaching expiration, may trigger
        early recomputation.
        """
        # Get current cached value with metadata
        cached_data = await cache.get(f"{key}:meta", use_pickle=True)

        if cached_data is not None:
            cached = CachedValue(**cached_data)

            # Check if we should early-refresh
            if not self.should_recompute(cached):
                cached.access_count += 1
                # Update access count asynchronously
                asyncio.create_task(
                    cache.set(f"{key}:meta", cached.__dict__, ttl=int(cached.ttl_remaining) + 10, use_pickle=True)
                )
                return cached.value

            logger.debug(f"Triggering early refresh for key {key}")

        # Compute new value
        start_time = time.time()
        value = await compute()
        compute_time = (time.time() - start_time) * 1000

        # Store with metadata
        cached = CachedValue(
            value=value,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=ttl),
            compute_time_ms=compute_time,
        )
        await cache.set(f"{key}:meta", cached.__dict__, ttl=ttl + 10, use_pickle=True)

        return value


class LockingStrategy:
    """
    Lock-based stampede protection.

    Only one client recomputes while others wait or get stale value.
    """

    def __init__(
        self,
        lock_timeout: float = 10.0,
        wait_timeout: float = 5.0,
        serve_stale: bool = True,
    ):
        """
        Args:
            lock_timeout: How long to hold the lock
            wait_timeout: How long other clients wait for lock
            serve_stale: If True, serve stale value while recomputing
        """
        self.lock_timeout = lock_timeout
        self.wait_timeout = wait_timeout
        self.serve_stale = serve_stale
        self._locks: Dict[str, asyncio.Lock] = {}
        self._computing: Dict[str, asyncio.Event] = {}

    async def get_or_compute(
        self,
        cache: Any,
        key: str,
        compute: Callable[[], Awaitable[T]],
        ttl: int,
        use_pickle: bool = False,
    ) -> T:
        """Get value with lock-based stampede protection."""
        # Try to get from cache
        value = await cache.get(key, use_pickle=use_pickle)
        if value is not None:
            return value

        # Get or create lock for this key
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()

        lock = self._locks[key]

        # Try to acquire lock
        acquired = False
        try:
            acquired = await asyncio.wait_for(
                lock.acquire(),
                timeout=self.wait_timeout,
            )
        except asyncio.TimeoutError:
            # Couldn't get lock, return stale or raise
            if self.serve_stale:
                stale_value = await cache.get(f"{key}:stale", use_pickle=use_pickle)
                if stale_value is not None:
                    return stale_value
            raise CacheStampedeTimeout(f"Timeout waiting for cache key: {key}")

        try:
            # Double-check cache (another process may have filled it)
            value = await cache.get(key, use_pickle=use_pickle)
            if value is not None:
                return value

            # Compute value
            value = await compute()

            # Store in cache
            await cache.set(key, value, ttl=ttl, use_pickle=use_pickle)

            # Store stale copy for fallback
            if self.serve_stale:
                await cache.set(f"{key}:stale", value, ttl=ttl * 2, use_pickle=use_pickle)

            return value

        finally:
            if acquired:
                lock.release()

    async def cleanup_locks(self) -> None:
        """Clean up unused locks."""
        # Remove locks that aren't held
        for key, lock in list(self._locks.items()):
            if not lock.locked():
                del self._locks[key]


class SemaphoreStrategy:
    """
    Semaphore-based stampede protection.

    Limits concurrent recomputations for the same key.
    """

    def __init__(
        self,
        max_concurrent: int = 1,
        wait_timeout: float = 5.0,
    ):
        """
        Args:
            max_concurrent: Maximum concurrent recomputations per key
            wait_timeout: How long to wait for semaphore
        """
        self.max_concurrent = max_concurrent
        self.wait_timeout = wait_timeout
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._pending: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    async def get_or_compute(
        self,
        cache: Any,
        key: str,
        compute: Callable[[], Awaitable[T]],
        ttl: int,
        use_pickle: bool = False,
    ) -> T:
        """Get value with semaphore-based coalescing."""
        # Try to get from cache
        value = await cache.get(key, use_pickle=use_pickle)
        if value is not None:
            return value

        async with self._lock:
            # Check if computation is pending
            if key in self._pending:
                # Wait for pending computation
                future = self._pending[key]
                async with self._lock:
                    pass  # Release lock while waiting
                return await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=self.wait_timeout,
                )

            # Get or create semaphore
            if key not in self._semaphores:
                self._semaphores[key] = asyncio.Semaphore(self.max_concurrent)

            # Create pending future
            future = asyncio.get_event_loop().create_future()
            self._pending[key] = future

        try:
            semaphore = self._semaphores[key]

            async with semaphore:
                # Double-check cache
                value = await cache.get(key, use_pickle=use_pickle)
                if value is not None:
                    future.set_result(value)
                    return value

                # Compute value
                value = await compute()

                # Store in cache
                await cache.set(key, value, ttl=ttl, use_pickle=use_pickle)

                # Resolve future for waiting clients
                future.set_result(value)

                return value

        except Exception as e:
            future.set_exception(e)
            raise

        finally:
            async with self._lock:
                if key in self._pending:
                    del self._pending[key]


class CacheStampedeTimeout(Exception):
    """Raised when cache stampede protection times out."""
    pass


class StampedeProtection:
    """
    Combined stampede protection with multiple strategies.

    Usage:
        protection = StampedeProtection(cache)

        # With probabilistic early expiration
        value = await protection.get_or_compute(
            key="expensive:123",
            compute=lambda: fetch_expensive_data(123),
            ttl=3600,
            strategy="probabilistic",
        )

        # With locking
        value = await protection.get_or_compute(
            key="expensive:456",
            compute=lambda: fetch_expensive_data(456),
            ttl=3600,
            strategy="locking",
        )
    """

    def __init__(
        self,
        cache: Any,
        default_strategy: str = "probabilistic",
    ):
        self.cache = cache
        self.default_strategy = default_strategy

        self._probabilistic = ProbabilisticEarlyExpiration()
        self._locking = LockingStrategy()
        self._semaphore = SemaphoreStrategy()

    async def get_or_compute(
        self,
        key: str,
        compute: Callable[[], Awaitable[T]],
        ttl: int,
        strategy: Optional[str] = None,
        use_pickle: bool = False,
    ) -> T:
        """
        Get value with stampede protection.

        Args:
            key: Cache key
            compute: Function to compute value if not cached
            ttl: Time to live in seconds
            strategy: Protection strategy ("probabilistic", "locking", "semaphore")
            use_pickle: Use pickle for serialization
        """
        strategy = strategy or self.default_strategy

        if strategy == "probabilistic":
            return await self._probabilistic.get_or_compute(
                self.cache, key, compute, ttl, use_pickle
            )
        elif strategy == "locking":
            return await self._locking.get_or_compute(
                self.cache, key, compute, ttl, use_pickle
            )
        elif strategy == "semaphore":
            return await self._semaphore.get_or_compute(
                self.cache, key, compute, ttl, use_pickle
            )
        else:
            raise ValueError(f"Unknown stampede protection strategy: {strategy}")

    def cached(
        self,
        ttl: int,
        key_builder: Optional[Callable[..., str]] = None,
        strategy: Optional[str] = None,
        use_pickle: bool = False,
    ) -> Callable:
        """
        Decorator for caching with stampede protection.

        Usage:
            @protection.cached(ttl=3600)
            async def expensive_operation(user_id: str):
                ...
        """
        def decorator(func: Callable) -> Callable:
            async def wrapper(*args, **kwargs):
                # Build cache key
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    key_data = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
                    cache_key = hashlib.md5(key_data.encode()).hexdigest()

                return await self.get_or_compute(
                    key=cache_key,
                    compute=lambda: func(*args, **kwargs),
                    ttl=ttl,
                    strategy=strategy,
                    use_pickle=use_pickle,
                )

            return wrapper
        return decorator


class RequestCoalescing:
    """
    Request coalescing for identical concurrent requests.

    Multiple identical requests that arrive within a window
    share the same computation.
    """

    def __init__(self, window_ms: float = 100):
        """
        Args:
            window_ms: Time window for coalescing requests
        """
        self.window_ms = window_ms
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def execute(
        self,
        key: str,
        compute: Callable[[], Awaitable[T]],
    ) -> T:
        """Execute with request coalescing."""
        async with self._lock:
            # Check if there's a pending request
            if key in self._pending:
                pending = self._pending[key]
                elapsed_ms = (time.time() - pending["started"]) * 1000

                # If within window, wait for existing request
                if elapsed_ms < self.window_ms:
                    future = pending["future"]
                    async with self._lock:
                        pass  # Release lock while waiting
                    return await asyncio.shield(future)

            # Start new request
            future = asyncio.get_event_loop().create_future()
            self._pending[key] = {
                "future": future,
                "started": time.time(),
            }

        try:
            # Compute value
            value = await compute()
            future.set_result(value)
            return value

        except Exception as e:
            future.set_exception(e)
            raise

        finally:
            async with self._lock:
                if key in self._pending:
                    del self._pending[key]
