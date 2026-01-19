"""
Multi-tier Caching System
=========================

Enterprise-grade caching system with support for multiple cache layers,
intelligent eviction, cache warming, and distributed caching.

Features:
- Multi-tier caching (L1 memory, L2 Redis, L3 persistent)
- Multiple eviction strategies (LRU, LFU, TTL, FIFO)
- Cache warming and preloading
- Write-through and write-behind
- Cache invalidation patterns
- Statistics and monitoring
- Distributed cache coherence

Author: Builder Engine Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
import time
import zlib
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from heapq import heappush, heappop
from typing import (
    Any,
    Awaitable,
    Callable,
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
KeyType = Union[str, Tuple[str, ...]]


# =============================================================================
# ENUMS
# =============================================================================


class EvictionStrategy(str, Enum):
    """Cache eviction strategies"""

    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used
    FIFO = "fifo"        # First In First Out
    TTL = "ttl"          # Time To Live based
    RANDOM = "random"    # Random eviction
    NONE = "none"        # No eviction (fixed size)


class CachePolicy(str, Enum):
    """Cache write policies"""

    WRITE_THROUGH = "write_through"   # Write to cache and backend
    WRITE_BEHIND = "write_behind"     # Write to cache, async to backend
    WRITE_AROUND = "write_around"     # Write only to backend
    READ_THROUGH = "read_through"     # Read from cache, fallback to backend
    CACHE_ASIDE = "cache_aside"       # Application manages cache


class CacheLayer(str, Enum):
    """Cache layer identifiers"""

    L1 = "l1"        # In-memory cache (fastest)
    L2 = "l2"        # Distributed cache (Redis)
    L3 = "l3"        # Persistent cache (disk/DB)
    ALL = "all"      # All layers


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class CacheEntry:
    """
    Represents a cached item with metadata.
    """

    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    ttl_seconds: Optional[float] = None
    access_count: int = 0
    size_bytes: int = 0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    compressed: bool = False
    version: int = 1

    @property
    def is_expired(self) -> bool:
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False

    @property
    def age_seconds(self) -> float:
        return (datetime.utcnow() - self.created_at).total_seconds()

    def touch(self) -> None:
        """Update access time and count"""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics"""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0
    memory_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def record_hit(self) -> None:
        self.hits += 1

    def record_miss(self) -> None:
        self.misses += 1

    def record_set(self) -> None:
        self.sets += 1

    def record_delete(self) -> None:
        self.deletes += 1

    def record_eviction(self) -> None:
        self.evictions += 1


# =============================================================================
# CACHE BACKENDS
# =============================================================================


class CacheBackend(ABC):
    """Abstract base class for cache backends"""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
        tags: Optional[Set[str]] = None
    ) -> None:
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        pass

    @abstractmethod
    async def clear(self) -> int:
        pass

    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[str]:
        pass

    @abstractmethod
    async def size(self) -> int:
        pass


class MemoryCache(CacheBackend):
    """
    In-memory cache with configurable eviction.
    """

    def __init__(
        self,
        max_size: int = 10000,
        max_memory_bytes: Optional[int] = None,
        eviction_strategy: EvictionStrategy = EvictionStrategy.LRU,
        default_ttl_seconds: Optional[float] = None
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_bytes
        self.eviction_strategy = eviction_strategy
        self.default_ttl_seconds = default_ttl_seconds

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._frequency: Dict[str, int] = defaultdict(int)
        self._memory_used: int = 0
        self._lock = asyncio.Lock()
        self._stats = CacheStats()
        self._logger = structlog.get_logger("cache.memory")

    async def get(self, key: str) -> Optional[CacheEntry]:
        async with self._lock:
            if key not in self._cache:
                self._stats.record_miss()
                return None

            entry = self._cache[key]

            if entry.is_expired:
                await self._remove_entry(key)
                self._stats.expirations += 1
                self._stats.record_miss()
                return None

            # Update access
            entry.touch()
            self._frequency[key] += 1

            # LRU: Move to end
            if self.eviction_strategy == EvictionStrategy.LRU:
                self._cache.move_to_end(key)

            self._stats.record_hit()
            return entry

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
        tags: Optional[Set[str]] = None
    ) -> None:
        async with self._lock:
            ttl = ttl_seconds or self.default_ttl_seconds

            # Calculate size
            size_bytes = len(pickle.dumps(value))

            entry = CacheEntry(
                key=key,
                value=value,
                ttl_seconds=ttl,
                expires_at=datetime.utcnow() + timedelta(seconds=ttl) if ttl else None,
                tags=tags or set(),
                size_bytes=size_bytes
            )

            # Remove old entry if exists
            if key in self._cache:
                await self._remove_entry(key)

            # Evict if needed
            while self._should_evict(entry):
                await self._evict_one()

            self._cache[key] = entry
            self._memory_used += size_bytes
            self._frequency[key] = 1
            self._stats.size = len(self._cache)
            self._stats.memory_bytes = self._memory_used
            self._stats.record_set()

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                await self._remove_entry(key)
                self._stats.record_delete()
                return True
            return False

    async def exists(self, key: str) -> bool:
        return key in self._cache and not self._cache[key].is_expired

    async def clear(self) -> int:
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._frequency.clear()
            self._memory_used = 0
            self._stats.size = 0
            return count

    async def keys(self, pattern: str = "*") -> List[str]:
        import fnmatch
        return [k for k in self._cache.keys() if fnmatch.fnmatch(k, pattern)]

    async def size(self) -> int:
        return len(self._cache)

    def _should_evict(self, new_entry: CacheEntry) -> bool:
        """Check if eviction is needed"""
        if len(self._cache) >= self.max_size:
            return True

        if self.max_memory_bytes:
            if self._memory_used + new_entry.size_bytes > self.max_memory_bytes:
                return True

        return False

    async def _evict_one(self) -> None:
        """Evict one entry based on strategy"""
        if not self._cache:
            return

        if self.eviction_strategy == EvictionStrategy.LRU:
            # First item is least recently used
            key = next(iter(self._cache))

        elif self.eviction_strategy == EvictionStrategy.LFU:
            # Find least frequently used
            key = min(self._cache.keys(), key=lambda k: self._frequency[k])

        elif self.eviction_strategy == EvictionStrategy.FIFO:
            key = next(iter(self._cache))

        elif self.eviction_strategy == EvictionStrategy.TTL:
            # Find entry closest to expiration
            key = min(
                self._cache.keys(),
                key=lambda k: (self._cache[k].expires_at or datetime.max)
            )

        elif self.eviction_strategy == EvictionStrategy.RANDOM:
            import random
            key = random.choice(list(self._cache.keys()))

        else:
            return

        await self._remove_entry(key)
        self._stats.record_eviction()

    async def _remove_entry(self, key: str) -> None:
        """Remove an entry"""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._memory_used -= entry.size_bytes
            self._frequency.pop(key, None)
            self._stats.size = len(self._cache)

    @property
    def stats(self) -> CacheStats:
        return self._stats


class RedisCache(CacheBackend):
    """
    Redis-backed distributed cache.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "cache",
        default_ttl_seconds: Optional[float] = 3600.0,
        serializer: str = "json"
    ):
        self._url = url
        self._prefix = prefix
        self._default_ttl = default_ttl_seconds
        self._serializer = serializer
        self._redis: Optional[Any] = None
        self._stats = CacheStats()
        self._logger = structlog.get_logger("cache.redis")

    async def connect(self) -> None:
        """Connect to Redis"""
        try:
            import redis.asyncio as aioredis
            self._redis = await aioredis.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=False
            )
            self._logger.info("redis_cache_connected")
        except Exception as e:
            self._logger.error("redis_connection_failed", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        if self._redis:
            await self._redis.close()

    def _key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    def _serialize(self, entry: CacheEntry) -> bytes:
        data = {
            "key": entry.key,
            "value": entry.value,
            "created_at": entry.created_at.isoformat(),
            "ttl_seconds": entry.ttl_seconds,
            "tags": list(entry.tags),
            "metadata": entry.metadata,
            "version": entry.version
        }
        return json.dumps(data).encode("utf-8")

    def _deserialize(self, data: bytes) -> CacheEntry:
        parsed = json.loads(data.decode("utf-8"))
        return CacheEntry(
            key=parsed["key"],
            value=parsed["value"],
            created_at=datetime.fromisoformat(parsed["created_at"]),
            ttl_seconds=parsed.get("ttl_seconds"),
            tags=set(parsed.get("tags", [])),
            metadata=parsed.get("metadata", {}),
            version=parsed.get("version", 1)
        )

    async def get(self, key: str) -> Optional[CacheEntry]:
        if not self._redis:
            return None

        data = await self._redis.get(self._key(key))
        if data:
            self._stats.record_hit()
            return self._deserialize(data)

        self._stats.record_miss()
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
        tags: Optional[Set[str]] = None
    ) -> None:
        if not self._redis:
            return

        ttl = ttl_seconds or self._default_ttl

        entry = CacheEntry(
            key=key,
            value=value,
            ttl_seconds=ttl,
            tags=tags or set()
        )

        data = self._serialize(entry)

        if ttl:
            await self._redis.setex(self._key(key), int(ttl), data)
        else:
            await self._redis.set(self._key(key), data)

        # Store tags for invalidation
        if tags:
            for tag in tags:
                await self._redis.sadd(f"{self._prefix}:tag:{tag}", key)

        self._stats.record_set()

    async def delete(self, key: str) -> bool:
        if not self._redis:
            return False

        result = await self._redis.delete(self._key(key))
        if result:
            self._stats.record_delete()
        return result > 0

    async def exists(self, key: str) -> bool:
        if not self._redis:
            return False
        return await self._redis.exists(self._key(key)) > 0

    async def clear(self) -> int:
        if not self._redis:
            return 0

        keys = await self._redis.keys(f"{self._prefix}:*")
        if keys:
            return await self._redis.delete(*keys)
        return 0

    async def keys(self, pattern: str = "*") -> List[str]:
        if not self._redis:
            return []

        full_pattern = f"{self._prefix}:{pattern}"
        keys = await self._redis.keys(full_pattern)
        prefix_len = len(self._prefix) + 1
        return [k.decode()[prefix_len:] if isinstance(k, bytes) else k[prefix_len:] for k in keys]

    async def size(self) -> int:
        return len(await self.keys())

    async def delete_by_tag(self, tag: str) -> int:
        """Delete all entries with a specific tag"""
        if not self._redis:
            return 0

        tag_key = f"{self._prefix}:tag:{tag}"
        keys = await self._redis.smembers(tag_key)

        count = 0
        for key in keys:
            if await self.delete(key.decode() if isinstance(key, bytes) else key):
                count += 1

        await self._redis.delete(tag_key)
        return count

    @property
    def stats(self) -> CacheStats:
        return self._stats


# =============================================================================
# CACHE MANAGER
# =============================================================================


class CacheManager:
    """
    Multi-tier cache manager.

    Manages multiple cache layers and provides a unified interface
    for caching operations.

    Usage:
        cache = CacheManager()
        await cache.start()

        # Set value
        await cache.set("user:123", user_data, ttl=3600)

        # Get value
        user = await cache.get("user:123")

        # With decorator
        @cache.cached(ttl=300)
        async def get_user(user_id: str) -> User:
            return await db.get_user(user_id)

        await cache.stop()
    """

    def __init__(
        self,
        l1_config: Optional[Dict[str, Any]] = None,
        l2_config: Optional[Dict[str, Any]] = None,
        default_ttl_seconds: float = 3600.0,
        policy: CachePolicy = CachePolicy.CACHE_ASIDE
    ):
        self._l1: Optional[MemoryCache] = None
        self._l2: Optional[RedisCache] = None
        self._default_ttl = default_ttl_seconds
        self._policy = policy
        self._running = False
        self._logger = structlog.get_logger("cache_manager")

        # Configure L1 (memory)
        l1_config = l1_config or {}
        self._l1 = MemoryCache(
            max_size=l1_config.get("max_size", 10000),
            max_memory_bytes=l1_config.get("max_memory_bytes"),
            eviction_strategy=l1_config.get("eviction_strategy", EvictionStrategy.LRU),
            default_ttl_seconds=l1_config.get("default_ttl_seconds", 300.0)
        )

        # Configure L2 (Redis) if provided
        if l2_config:
            self._l2 = RedisCache(
                url=l2_config.get("url", "redis://localhost:6379"),
                prefix=l2_config.get("prefix", "cache"),
                default_ttl_seconds=l2_config.get("default_ttl_seconds", 3600.0)
            )

    async def start(self) -> None:
        """Start the cache manager"""
        if self._l2:
            await self._l2.connect()
        self._running = True
        self._logger.info("cache_manager_started")

    async def stop(self) -> None:
        """Stop the cache manager"""
        if self._l2:
            await self._l2.disconnect()
        self._running = False
        self._logger.info("cache_manager_stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # -------------------------------------------------------------------------
    # Basic Operations
    # -------------------------------------------------------------------------

    async def get(
        self,
        key: str,
        default: Any = None,
        layer: CacheLayer = CacheLayer.ALL
    ) -> Any:
        """
        Get a value from cache.

        Args:
            key: Cache key
            default: Default value if not found
            layer: Which layer(s) to check

        Returns:
            Cached value or default
        """
        # Try L1 first
        if layer in (CacheLayer.L1, CacheLayer.ALL) and self._l1:
            entry = await self._l1.get(key)
            if entry:
                return entry.value

        # Try L2
        if layer in (CacheLayer.L2, CacheLayer.ALL) and self._l2:
            entry = await self._l2.get(key)
            if entry:
                # Populate L1
                if self._l1:
                    await self._l1.set(
                        key, entry.value,
                        ttl_seconds=entry.ttl_seconds,
                        tags=entry.tags
                    )
                return entry.value

        return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
        tags: Optional[Set[str]] = None,
        layer: CacheLayer = CacheLayer.ALL
    ) -> None:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live
            tags: Tags for invalidation
            layer: Which layer(s) to set
        """
        ttl = ttl_seconds or self._default_ttl

        # Set in L1
        if layer in (CacheLayer.L1, CacheLayer.ALL) and self._l1:
            await self._l1.set(key, value, ttl_seconds=ttl, tags=tags)

        # Set in L2
        if layer in (CacheLayer.L2, CacheLayer.ALL) and self._l2:
            await self._l2.set(key, value, ttl_seconds=ttl, tags=tags)

    async def delete(
        self,
        key: str,
        layer: CacheLayer = CacheLayer.ALL
    ) -> bool:
        """Delete a value from cache"""
        deleted = False

        if layer in (CacheLayer.L1, CacheLayer.ALL) and self._l1:
            if await self._l1.delete(key):
                deleted = True

        if layer in (CacheLayer.L2, CacheLayer.ALL) and self._l2:
            if await self._l2.delete(key):
                deleted = True

        return deleted

    async def exists(
        self,
        key: str,
        layer: CacheLayer = CacheLayer.ALL
    ) -> bool:
        """Check if a key exists"""
        if layer in (CacheLayer.L1, CacheLayer.ALL) and self._l1:
            if await self._l1.exists(key):
                return True

        if layer in (CacheLayer.L2, CacheLayer.ALL) and self._l2:
            if await self._l2.exists(key):
                return True

        return False

    async def clear(self, layer: CacheLayer = CacheLayer.ALL) -> int:
        """Clear cache"""
        count = 0

        if layer in (CacheLayer.L1, CacheLayer.ALL) and self._l1:
            count += await self._l1.clear()

        if layer in (CacheLayer.L2, CacheLayer.ALL) and self._l2:
            count += await self._l2.clear()

        return count

    # -------------------------------------------------------------------------
    # Advanced Operations
    # -------------------------------------------------------------------------

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Awaitable[Any]],
        ttl_seconds: Optional[float] = None,
        tags: Optional[Set[str]] = None
    ) -> Any:
        """Get value or compute and cache it"""
        value = await self.get(key)

        if value is None:
            value = await factory()
            await self.set(key, value, ttl_seconds=ttl_seconds, tags=tags)

        return value

    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with a specific tag"""
        count = 0

        if self._l2:
            count += await self._l2.delete_by_tag(tag)

        return count

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values"""
        results = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                results[key] = value
        return results

    async def set_many(
        self,
        items: Dict[str, Any],
        ttl_seconds: Optional[float] = None
    ) -> None:
        """Set multiple values"""
        for key, value in items.items():
            await self.set(key, value, ttl_seconds=ttl_seconds)

    async def delete_many(self, keys: List[str]) -> int:
        """Delete multiple values"""
        count = 0
        for key in keys:
            if await self.delete(key):
                count += 1
        return count

    # -------------------------------------------------------------------------
    # Decorator
    # -------------------------------------------------------------------------

    def cached(
        self,
        ttl_seconds: Optional[float] = None,
        key_builder: Optional[Callable[..., str]] = None,
        tags: Optional[Set[str]] = None
    ) -> Callable:
        """
        Decorator for caching function results.

        Usage:
            @cache.cached(ttl=300)
            async def get_user(user_id: str) -> User:
                return await db.get_user(user_id)
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                # Build cache key
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    key_parts = [func.__name__]
                    key_parts.extend(str(arg) for arg in args)
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = ":".join(key_parts)

                # Try to get from cache
                cached_value = await self.get(cache_key)
                if cached_value is not None:
                    return cached_value

                # Call function
                result = await func(*args, **kwargs)

                # Cache result
                await self.set(
                    cache_key, result,
                    ttl_seconds=ttl_seconds or self._default_ttl,
                    tags=tags
                )

                return result

            return wrapper
        return decorator

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {}

        if self._l1:
            stats["l1"] = {
                "hits": self._l1.stats.hits,
                "misses": self._l1.stats.misses,
                "hit_rate": self._l1.stats.hit_rate,
                "size": self._l1.stats.size,
                "memory_bytes": self._l1.stats.memory_bytes,
                "evictions": self._l1.stats.evictions
            }

        if self._l2:
            stats["l2"] = {
                "hits": self._l2.stats.hits,
                "misses": self._l2.stats.misses,
                "hit_rate": self._l2.stats.hit_rate,
                "sets": self._l2.stats.sets,
                "deletes": self._l2.stats.deletes
            }

        return stats


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def cache_key(*parts: Any) -> str:
    """Build a cache key from parts"""
    return ":".join(str(p) for p in parts)


def hash_key(data: Any) -> str:
    """Create a hash-based cache key"""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()
