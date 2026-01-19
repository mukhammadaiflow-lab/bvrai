"""In-memory cache implementation."""

from typing import Optional, Any, Dict, List, TypeVar, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import asyncio
import hashlib
import functools
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry:
    """A single cache entry."""
    value: Any
    expires_at: Optional[datetime]
    created_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


class LRUCache:
    """
    LRU (Least Recently Used) cache.

    Thread-safe with async support.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[int] = None,
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = MemoryCacheStats()

    async def get(self, key: str, default: T = None) -> Optional[T]:
        """Get a value from cache."""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return default

            if entry.is_expired:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return default

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()

            self._stats.hits += 1
            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set a value in cache."""
        ttl = ttl or self.default_ttl
        expires_at = datetime.utcnow() + timedelta(seconds=ttl) if ttl else None

        async with self._lock:
            # Remove if exists
            if key in self._cache:
                del self._cache[key]

            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # Remove oldest
                self._stats.evictions += 1

            # Add new entry
            self._cache[key] = CacheEntry(
                value=value,
                expires_at=expires_at,
            )
            self._stats.sets += 1

    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.deletes += 1
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                del self._cache[key]
                return False
            return True

    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._cache.clear()

    async def size(self) -> int:
        """Get number of entries."""
        return len(self._cache)

    async def cleanup_expired(self) -> int:
        """Remove expired entries."""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
            self._stats.evictions += len(expired_keys)
            return len(expired_keys)

    @property
    def stats(self) -> "MemoryCacheStats":
        """Get cache statistics."""
        return self._stats


class TTLCache:
    """
    Time-based cache that automatically expires entries.

    Includes background cleanup task.
    """

    def __init__(
        self,
        default_ttl: int = 3600,
        cleanup_interval: int = 60,
        max_size: Optional[int] = None,
    ):
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        self._stats = MemoryCacheStats()

    async def start(self) -> None:
        """Start the background cleanup task."""
        if self._running:
            return
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("TTL cache started")

    async def stop(self) -> None:
        """Stop the background cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        logger.info("TTL cache stopped")

    async def _cleanup_loop(self) -> None:
        """Background loop for cleaning up expired entries."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

    async def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        async with self._lock:
            now = datetime.utcnow()
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.expires_at and entry.expires_at < now
            ]
            for key in expired_keys:
                del self._cache[key]
            if expired_keys:
                self._stats.evictions += len(expired_keys)
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def get(self, key: str, default: T = None) -> Optional[T]:
        """Get a value from cache."""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return default

            if entry.is_expired:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return default

            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
            self._stats.hits += 1
            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set a value in cache."""
        ttl = ttl or self.default_ttl
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)

        async with self._lock:
            # Check size limit
            if self.max_size and len(self._cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].last_accessed,
                )
                del self._cache[oldest_key]
                self._stats.evictions += 1

            self._cache[key] = CacheEntry(
                value=value,
                expires_at=expires_at,
            )
            self._stats.sets += 1

    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.deletes += 1
                return True
            return False

    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._cache.clear()

    @property
    def stats(self) -> "MemoryCacheStats":
        """Get cache statistics."""
        return self._stats


class TwoLevelCache:
    """
    Two-level cache with L1 memory and L2 Redis.

    Provides fast local access with distributed backing store.
    """

    def __init__(
        self,
        l1_cache: LRUCache,
        l2_cache: Any,  # RedisCache
        l1_ttl: Optional[int] = 60,
    ):
        self.l1 = l1_cache
        self.l2 = l2_cache
        self.l1_ttl = l1_ttl

    async def get(self, key: str, default: T = None) -> Optional[T]:
        """Get value, checking L1 first, then L2."""
        # Try L1 first
        value = await self.l1.get(key)
        if value is not None:
            return value

        # Try L2
        value = await self.l2.get(key)
        if value is not None:
            # Populate L1
            await self.l1.set(key, value, ttl=self.l1_ttl)
            return value

        return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set value in both L1 and L2."""
        await self.l1.set(key, value, ttl=self.l1_ttl)
        await self.l2.set(key, value, ttl=ttl)

    async def delete(self, key: str) -> bool:
        """Delete from both L1 and L2."""
        l1_deleted = await self.l1.delete(key)
        l2_deleted = await self.l2.delete(key)
        return l1_deleted or l2_deleted

    async def invalidate_l1(self, key: str) -> bool:
        """Invalidate only L1 cache (for distributed invalidation)."""
        return await self.l1.delete(key)


@dataclass
class MemoryCacheStats:
    """Statistics for memory cache."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
        }


def cached(
    ttl: Optional[int] = 300,
    key_builder: Optional[Callable[..., str]] = None,
    cache: Optional[LRUCache] = None,
) -> Callable:
    """
    Decorator for caching function results in memory.

    Usage:
        @cached(ttl=60)
        async def get_user(user_id: str):
            ...
    """
    _cache = cache or LRUCache(max_size=1000, default_ttl=ttl)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                key_data = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()

            # Try to get from cache
            value = await _cache.get(cache_key)
            if value is not None:
                return value

            # Compute and cache
            value = await func(*args, **kwargs)
            await _cache.set(cache_key, value, ttl=ttl)
            return value

        wrapper.cache = _cache
        return wrapper

    return decorator


class ComputeCache:
    """
    Cache specifically for expensive computations.

    Includes deduplication of concurrent requests.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self._cache = LRUCache(max_size=max_size, default_ttl=default_ttl)
        self._pending: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    async def get_or_compute(
        self,
        key: str,
        compute: Callable[[], Awaitable[T]],
        ttl: Optional[int] = None,
    ) -> T:
        """
        Get value from cache or compute it.

        Deduplicates concurrent requests for the same key.
        """
        # Check cache first
        value = await self._cache.get(key)
        if value is not None:
            return value

        async with self._lock:
            # Check if there's a pending computation
            if key in self._pending:
                future = self._pending[key]
            else:
                # Start new computation
                future = asyncio.get_event_loop().create_future()
                self._pending[key] = future

                # Release lock during computation
                asyncio.create_task(self._compute_and_cache(
                    key, compute, future, ttl
                ))

        # Wait for result
        return await future

    async def _compute_and_cache(
        self,
        key: str,
        compute: Callable[[], Awaitable[T]],
        future: asyncio.Future,
        ttl: Optional[int],
    ) -> None:
        """Compute value and update cache."""
        try:
            value = await compute()
            await self._cache.set(key, value, ttl=ttl)
            future.set_result(value)
        except Exception as e:
            future.set_exception(e)
        finally:
            async with self._lock:
                if key in self._pending:
                    del self._pending[key]
