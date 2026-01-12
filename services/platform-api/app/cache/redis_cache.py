"""Redis-based caching layer."""

from typing import Optional, Any, Dict, List, Union, TypeVar, Callable, Awaitable
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import json
import pickle
import hashlib
import logging
import functools

try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    Redis = Any

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheSerializationError(Exception):
    """Raised when serialization/deserialization fails."""
    pass


@dataclass
class CacheConfig:
    """Configuration for Redis cache."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    default_ttl: int = 3600  # 1 hour
    key_prefix: str = "bvrai:"
    max_connections: int = 50
    socket_timeout: float = 5.0
    retry_on_timeout: bool = True
    encoding: str = "utf-8"


class CacheSerializer:
    """
    Serializer for cache values.

    Supports JSON and pickle serialization.
    """

    @staticmethod
    def serialize(value: Any, use_pickle: bool = False) -> bytes:
        """Serialize a value for storage."""
        try:
            if use_pickle:
                return pickle.dumps(value)
            else:
                return json.dumps(value).encode('utf-8')
        except Exception as e:
            raise CacheSerializationError(f"Failed to serialize: {e}")

    @staticmethod
    def deserialize(data: bytes, use_pickle: bool = False) -> Any:
        """Deserialize a value from storage."""
        try:
            if use_pickle:
                return pickle.loads(data)
            else:
                return json.loads(data.decode('utf-8'))
        except Exception as e:
            raise CacheSerializationError(f"Failed to deserialize: {e}")


class RedisCache:
    """
    Redis cache implementation with advanced features.

    Features:
    - Multiple serialization methods
    - Automatic key prefixing
    - TTL support
    - Cache stampede prevention
    - Statistics tracking
    - Pub/sub for cache invalidation

    Usage:
        cache = RedisCache(CacheConfig())
        await cache.connect()

        # Basic operations
        await cache.set("key", {"data": "value"}, ttl=3600)
        value = await cache.get("key")

        # With decorator
        @cache.cached(ttl=300)
        async def expensive_operation(user_id: str):
            ...
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        if not HAS_REDIS:
            raise ImportError("redis package required. Install with: pip install redis")

        self.config = config or CacheConfig()
        self._client: Optional[Redis] = None
        self._pool = None
        self._stats = CacheStats()
        self._locks: Dict[str, asyncio.Lock] = {}

    @property
    def client(self) -> Redis:
        """Get Redis client."""
        if self._client is None:
            raise RuntimeError("Cache not connected. Call connect() first.")
        return self._client

    async def connect(self) -> None:
        """Connect to Redis."""
        self._pool = redis.ConnectionPool(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            max_connections=self.config.max_connections,
            socket_timeout=self.config.socket_timeout,
            retry_on_timeout=self.config.retry_on_timeout,
            decode_responses=False,
        )
        self._client = redis.Redis(connection_pool=self._pool)

        # Test connection
        await self._client.ping()
        logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
        logger.info("Disconnected from Redis")

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.config.key_prefix}{key}"

    # Basic Operations

    async def get(
        self,
        key: str,
        default: T = None,
        use_pickle: bool = False,
    ) -> Optional[T]:
        """Get a value from cache."""
        full_key = self._make_key(key)

        try:
            data = await self.client.get(full_key)
            if data is None:
                self._stats.misses += 1
                return default

            self._stats.hits += 1
            return CacheSerializer.deserialize(data, use_pickle)

        except Exception as e:
            logger.error(f"Cache get error for {key}: {e}")
            self._stats.errors += 1
            return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        use_pickle: bool = False,
    ) -> bool:
        """Set a value in cache."""
        full_key = self._make_key(key)
        ttl = ttl or self.config.default_ttl

        try:
            data = CacheSerializer.serialize(value, use_pickle)
            await self.client.setex(full_key, ttl, data)
            self._stats.sets += 1
            return True

        except Exception as e:
            logger.error(f"Cache set error for {key}: {e}")
            self._stats.errors += 1
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        full_key = self._make_key(key)

        try:
            result = await self.client.delete(full_key)
            self._stats.deletes += 1
            return result > 0

        except Exception as e:
            logger.error(f"Cache delete error for {key}: {e}")
            self._stats.errors += 1
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        full_key = self._make_key(key)
        return await self.client.exists(full_key) > 0

    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration on a key."""
        full_key = self._make_key(key)
        return await self.client.expire(full_key, ttl)

    async def ttl(self, key: str) -> int:
        """Get TTL of a key."""
        full_key = self._make_key(key)
        return await self.client.ttl(full_key)

    # Batch Operations

    async def mget(
        self,
        keys: List[str],
        use_pickle: bool = False,
    ) -> Dict[str, Any]:
        """Get multiple values."""
        full_keys = [self._make_key(k) for k in keys]

        try:
            values = await self.client.mget(full_keys)
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = CacheSerializer.deserialize(value, use_pickle)
                    self._stats.hits += 1
                else:
                    self._stats.misses += 1
            return result

        except Exception as e:
            logger.error(f"Cache mget error: {e}")
            self._stats.errors += 1
            return {}

    async def mset(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None,
        use_pickle: bool = False,
    ) -> bool:
        """Set multiple values."""
        ttl = ttl or self.config.default_ttl

        try:
            pipe = self.client.pipeline()
            for key, value in mapping.items():
                full_key = self._make_key(key)
                data = CacheSerializer.serialize(value, use_pickle)
                pipe.setex(full_key, ttl, data)
            await pipe.execute()
            self._stats.sets += len(mapping)
            return True

        except Exception as e:
            logger.error(f"Cache mset error: {e}")
            self._stats.errors += 1
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern."""
        full_pattern = self._make_key(pattern)

        try:
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = await self.client.scan(cursor, match=full_pattern)
                if keys:
                    deleted += await self.client.delete(*keys)
                if cursor == 0:
                    break
            self._stats.deletes += deleted
            return deleted

        except Exception as e:
            logger.error(f"Cache delete_pattern error: {e}")
            self._stats.errors += 1
            return 0

    # Counter Operations

    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment a counter."""
        full_key = self._make_key(key)
        return await self.client.incrby(full_key, amount)

    async def decr(self, key: str, amount: int = 1) -> int:
        """Decrement a counter."""
        full_key = self._make_key(key)
        return await self.client.decrby(full_key, amount)

    # Hash Operations

    async def hget(self, key: str, field: str) -> Optional[Any]:
        """Get a field from a hash."""
        full_key = self._make_key(key)
        data = await self.client.hget(full_key, field)
        if data:
            return json.loads(data)
        return None

    async def hset(self, key: str, field: str, value: Any) -> bool:
        """Set a field in a hash."""
        full_key = self._make_key(key)
        data = json.dumps(value)
        return await self.client.hset(full_key, field, data)

    async def hgetall(self, key: str) -> Dict[str, Any]:
        """Get all fields from a hash."""
        full_key = self._make_key(key)
        data = await self.client.hgetall(full_key)
        return {
            k.decode(): json.loads(v)
            for k, v in data.items()
        }

    async def hdel(self, key: str, *fields: str) -> int:
        """Delete fields from a hash."""
        full_key = self._make_key(key)
        return await self.client.hdel(full_key, *fields)

    # List Operations

    async def lpush(self, key: str, *values: Any) -> int:
        """Push values to the left of a list."""
        full_key = self._make_key(key)
        serialized = [json.dumps(v) for v in values]
        return await self.client.lpush(full_key, *serialized)

    async def rpush(self, key: str, *values: Any) -> int:
        """Push values to the right of a list."""
        full_key = self._make_key(key)
        serialized = [json.dumps(v) for v in values]
        return await self.client.rpush(full_key, *serialized)

    async def lrange(self, key: str, start: int, end: int) -> List[Any]:
        """Get a range of values from a list."""
        full_key = self._make_key(key)
        data = await self.client.lrange(full_key, start, end)
        return [json.loads(v) for v in data]

    async def llen(self, key: str) -> int:
        """Get length of a list."""
        full_key = self._make_key(key)
        return await self.client.llen(full_key)

    # Set Operations

    async def sadd(self, key: str, *members: str) -> int:
        """Add members to a set."""
        full_key = self._make_key(key)
        return await self.client.sadd(full_key, *members)

    async def srem(self, key: str, *members: str) -> int:
        """Remove members from a set."""
        full_key = self._make_key(key)
        return await self.client.srem(full_key, *members)

    async def smembers(self, key: str) -> set:
        """Get all members of a set."""
        full_key = self._make_key(key)
        members = await self.client.smembers(full_key)
        return {m.decode() for m in members}

    async def sismember(self, key: str, member: str) -> bool:
        """Check if member is in set."""
        full_key = self._make_key(key)
        return await self.client.sismember(full_key, member)

    # Sorted Set Operations

    async def zadd(
        self,
        key: str,
        mapping: Dict[str, float],
    ) -> int:
        """Add members to a sorted set."""
        full_key = self._make_key(key)
        return await self.client.zadd(full_key, mapping)

    async def zrange(
        self,
        key: str,
        start: int,
        end: int,
        withscores: bool = False,
    ) -> Union[List[str], List[tuple]]:
        """Get range from sorted set."""
        full_key = self._make_key(key)
        result = await self.client.zrange(full_key, start, end, withscores=withscores)
        if withscores:
            return [(m.decode(), s) for m, s in result]
        return [m.decode() for m in result]

    async def zscore(self, key: str, member: str) -> Optional[float]:
        """Get score of a member."""
        full_key = self._make_key(key)
        return await self.client.zscore(full_key, member)

    # Cache Stampede Prevention

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Awaitable[T]],
        ttl: Optional[int] = None,
        use_pickle: bool = False,
        lock_timeout: float = 10.0,
    ) -> T:
        """
        Get value or compute and cache it.

        Uses locking to prevent cache stampede.
        """
        # Try to get from cache first
        value = await self.get(key, use_pickle=use_pickle)
        if value is not None:
            return value

        # Get lock for this key
        lock_key = f"lock:{key}"
        if lock_key not in self._locks:
            self._locks[lock_key] = asyncio.Lock()

        async with self._locks[lock_key]:
            # Check again after acquiring lock
            value = await self.get(key, use_pickle=use_pickle)
            if value is not None:
                return value

            # Compute value
            value = await factory()

            # Store in cache
            await self.set(key, value, ttl=ttl, use_pickle=use_pickle)

            return value

    # Decorator

    def cached(
        self,
        ttl: Optional[int] = None,
        key_builder: Optional[Callable[..., str]] = None,
        use_pickle: bool = False,
    ) -> Callable:
        """
        Decorator for caching function results.

        Usage:
            @cache.cached(ttl=300)
            async def get_user(user_id: str):
                ...

            @cache.cached(key_builder=lambda x, y: f"sum:{x}:{y}")
            async def compute(x: int, y: int):
                ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Build cache key
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    # Default key: function_name:hash(args,kwargs)
                    key_data = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
                    key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
                    cache_key = f"{func.__name__}:{key_hash}"

                return await self.get_or_set(
                    cache_key,
                    lambda: func(*args, **kwargs),
                    ttl=ttl,
                    use_pickle=use_pickle,
                )

            # Add cache invalidation method
            async def invalidate(*args, **kwargs):
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    key_data = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
                    key_hash = hashlib.md5(key_data.encode()).hexdigest()[:16]
                    cache_key = f"{func.__name__}:{key_hash}"
                await self.delete(cache_key)

            wrapper.invalidate = invalidate
            return wrapper

        return decorator

    # Pub/Sub for Invalidation

    async def publish_invalidation(self, key: str) -> int:
        """Publish cache invalidation message."""
        channel = f"{self.config.key_prefix}invalidation"
        return await self.client.publish(channel, key)

    async def subscribe_invalidation(
        self,
        callback: Callable[[str], Awaitable[None]],
    ) -> None:
        """Subscribe to cache invalidation messages."""
        pubsub = self.client.pubsub()
        channel = f"{self.config.key_prefix}invalidation"

        await pubsub.subscribe(channel)

        async for message in pubsub.listen():
            if message["type"] == "message":
                key = message["data"].decode()
                await callback(key)

    # Statistics

    @property
    def stats(self) -> "CacheStats":
        """Get cache statistics."""
        return self._stats

    async def info(self) -> Dict[str, Any]:
        """Get Redis server info."""
        return await self.client.info()

    async def dbsize(self) -> int:
        """Get number of keys in database."""
        return await self.client.dbsize()

    async def flush(self) -> None:
        """Flush all keys with our prefix."""
        await self.delete_pattern("*")


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0

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
            "errors": self.errors,
            "hit_rate": self.hit_rate,
        }


class CacheManager:
    """
    Manager for multiple cache instances.

    Usage:
        manager = CacheManager()
        manager.add("default", RedisCache(default_config))
        manager.add("sessions", RedisCache(session_config))

        await manager.connect_all()

        default = manager.get("default")
        await default.set("key", "value")
    """

    def __init__(self):
        self._caches: Dict[str, RedisCache] = {}

    def add(self, name: str, cache: RedisCache) -> None:
        """Add a cache instance."""
        self._caches[name] = cache

    def get(self, name: str) -> RedisCache:
        """Get a cache instance."""
        if name not in self._caches:
            raise KeyError(f"Unknown cache: {name}")
        return self._caches[name]

    async def connect_all(self) -> None:
        """Connect all caches."""
        for name, cache in self._caches.items():
            try:
                await cache.connect()
                logger.info(f"Connected cache: {name}")
            except Exception as e:
                logger.error(f"Failed to connect cache {name}: {e}")
                raise

    async def disconnect_all(self) -> None:
        """Disconnect all caches."""
        for name, cache in self._caches.items():
            try:
                await cache.disconnect()
                logger.info(f"Disconnected cache: {name}")
            except Exception as e:
                logger.error(f"Failed to disconnect cache {name}: {e}")

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        return {name: cache.stats.to_dict() for name, cache in self._caches.items()}


# Global cache instance
_default_cache: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    """Get the default cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = RedisCache()
    return _default_cache


async def init_cache(config: Optional[CacheConfig] = None) -> RedisCache:
    """Initialize and connect the default cache."""
    global _default_cache
    _default_cache = RedisCache(config)
    await _default_cache.connect()
    return _default_cache
