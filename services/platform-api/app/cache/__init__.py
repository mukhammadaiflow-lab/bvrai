"""
Enterprise Caching Layer

Multi-tier caching with:
- In-memory LRU/LFU cache
- Distributed Redis cache
- Cache stampede prevention
- Cross-datacenter replication
- Intelligent invalidation
"""

from app.cache.redis_cache import (
    RedisCache,
    CacheConfig,
    CacheStats,
    CacheManager,
    CacheSerializer,
    CacheSerializationError,
    get_cache,
    init_cache,
)
from app.cache.memory_cache import (
    LRUCache,
    TTLCache,
    TwoLevelCache,
    MemoryCacheStats,
    CacheEntry,
    ComputeCache,
    cached,
)
from app.cache.distributed import (
    DistributedCache,
    ConsistentHashRing,
    CacheNode,
    NodeState,
    ReplicationStrategy,
    CrossDatacenterCache,
)
from app.cache.stampede import (
    StampedeProtection,
    ProbabilisticEarlyExpiration,
    LockingStrategy,
    SemaphoreStrategy,
    RequestCoalescing,
    CacheStampedeTimeout,
)

__all__ = [
    # Redis cache
    "RedisCache",
    "CacheConfig",
    "CacheStats",
    "CacheManager",
    "CacheSerializer",
    "CacheSerializationError",
    "get_cache",
    "init_cache",
    # Memory cache
    "LRUCache",
    "TTLCache",
    "TwoLevelCache",
    "MemoryCacheStats",
    "CacheEntry",
    "ComputeCache",
    "cached",
    # Distributed
    "DistributedCache",
    "ConsistentHashRing",
    "CacheNode",
    "NodeState",
    "ReplicationStrategy",
    "CrossDatacenterCache",
    # Stampede
    "StampedeProtection",
    "ProbabilisticEarlyExpiration",
    "LockingStrategy",
    "SemaphoreStrategy",
    "RequestCoalescing",
    "CacheStampedeTimeout",
]
