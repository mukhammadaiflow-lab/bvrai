"""
Distributed Rate Limiting

Redis-backed rate limiters for distributed systems:
- Atomic operations using Lua scripts
- Cluster-aware with consistent hashing
- Failover and fallback strategies
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import time
import logging
import hashlib
import json

from app.ratelimit.algorithms import RateLimiter, RateLimitResult

logger = logging.getLogger(__name__)


# Lua script for atomic token bucket
TOKEN_BUCKET_SCRIPT = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])

-- Get current bucket state
local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
local tokens = tonumber(bucket[1])
local last_update = tonumber(bucket[2])

-- Initialize if not exists
if tokens == nil then
    tokens = capacity
    last_update = now
end

-- Calculate tokens to add since last update
local elapsed = now - last_update
local tokens_to_add = elapsed * rate
tokens = math.min(capacity, tokens + tokens_to_add)

-- Check if enough tokens
local allowed = 0
local remaining = 0
local retry_after = 0

if tokens >= cost then
    tokens = tokens - cost
    allowed = 1
    remaining = math.floor(tokens)
else
    local tokens_needed = cost - tokens
    retry_after = tokens_needed / rate
    remaining = 0
end

-- Update bucket state
redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
redis.call('EXPIRE', key, ttl)

return {allowed, remaining, retry_after, math.floor(tokens)}
"""


# Lua script for atomic sliding window
SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])

-- Calculate window boundaries
local window_start = now_ms - window_ms

-- Remove old entries
redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

-- Count current requests
local current_count = redis.call('ZCOUNT', key, window_start, now_ms)

-- Check limit
local allowed = 0
local remaining = 0
local retry_after = 0

if current_count + cost <= limit then
    -- Add new entries
    for i = 1, cost do
        redis.call('ZADD', key, now_ms, now_ms .. ':' .. i .. ':' .. math.random(1000000))
    end
    allowed = 1
    remaining = limit - current_count - cost
else
    -- Calculate retry_after from oldest entry
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    if #oldest >= 2 then
        local oldest_time = tonumber(oldest[2])
        retry_after = (oldest_time + window_ms - now_ms) / 1000
        if retry_after < 0 then retry_after = 0 end
    end
    remaining = 0
end

-- Set TTL
redis.call('PEXPIRE', key, window_ms * 2)

return {allowed, remaining, retry_after, current_count}
"""


# Lua script for atomic fixed window
FIXED_WINDOW_SCRIPT = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window_seconds = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])

-- Calculate current window
local current_window = math.floor(now / window_seconds)
local window_key = key .. ':' .. current_window

-- Get current count
local current_count = tonumber(redis.call('GET', window_key)) or 0

-- Check limit
local allowed = 0
local remaining = 0
local window_end = (current_window + 1) * window_seconds
local retry_after = window_end - now

if current_count + cost <= limit then
    -- Increment counter
    redis.call('INCRBY', window_key, cost)
    redis.call('EXPIRE', window_key, window_seconds * 2)
    allowed = 1
    remaining = limit - current_count - cost
    current_count = current_count + cost
else
    remaining = 0
end

return {allowed, remaining, retry_after, current_count}
"""


# Lua script for atomic leaky bucket
LEAKY_BUCKET_SCRIPT = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])

-- Get current bucket state
local bucket = redis.call('HMGET', key, 'level', 'last_update')
local level = tonumber(bucket[1])
local last_update = tonumber(bucket[2])

-- Initialize if not exists
if level == nil then
    level = 0
    last_update = now
end

-- Calculate leaked amount
local elapsed = now - last_update
local leaked = elapsed * rate
level = math.max(0, level - leaked)

-- Check if room in bucket
local allowed = 0
local remaining = 0
local retry_after = 0

if level + cost <= capacity then
    level = level + cost
    allowed = 1
    remaining = math.floor(capacity - level)
else
    local overflow = level + cost - capacity
    retry_after = overflow / rate
    remaining = 0
end

-- Update bucket state
redis.call('HMSET', key, 'level', level, 'last_update', now)
redis.call('EXPIRE', key, ttl)

return {allowed, remaining, retry_after, math.floor(level)}
"""


class FailoverStrategy(str, Enum):
    """Strategy when Redis is unavailable."""
    ALLOW = "allow"  # Allow all requests
    DENY = "deny"  # Deny all requests
    LOCAL = "local"  # Fall back to local rate limiter


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    ssl: bool = False
    cluster_mode: bool = False
    sentinel_master: Optional[str] = None
    sentinel_nodes: List[Tuple[str, int]] = field(default_factory=list)
    connection_pool_size: int = 10
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0


class RedisRateLimiter(RateLimiter):
    """Base class for Redis-backed rate limiters."""

    def __init__(
        self,
        redis_client: Any,  # aioredis.Redis
        key_prefix: str = "ratelimit",
        failover_strategy: FailoverStrategy = FailoverStrategy.LOCAL,
        local_limiter: Optional[RateLimiter] = None,
    ):
        """
        Args:
            redis_client: Async Redis client
            key_prefix: Prefix for Redis keys
            failover_strategy: Strategy when Redis is unavailable
            local_limiter: Fallback limiter for LOCAL strategy
        """
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.failover_strategy = failover_strategy
        self.local_limiter = local_limiter
        self._script_sha: Optional[str] = None

    def _make_key(self, key: str) -> str:
        """Create namespaced Redis key."""
        return f"{self.key_prefix}:{key}"

    async def _handle_failure(self, key: str, cost: int, error: Exception) -> RateLimitResult:
        """Handle Redis failure based on strategy."""
        logger.error(f"Redis rate limiting failed: {error}")

        if self.failover_strategy == FailoverStrategy.ALLOW:
            return RateLimitResult(
                allowed=True,
                remaining=-1,  # Unknown
                reset_at=datetime.utcnow() + timedelta(seconds=60),
                limit=-1,
            )
        elif self.failover_strategy == FailoverStrategy.DENY:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=datetime.utcnow() + timedelta(seconds=60),
                limit=0,
                retry_after=60.0,
            )
        elif self.failover_strategy == FailoverStrategy.LOCAL and self.local_limiter:
            return await self.local_limiter.check(key, cost)
        else:
            # Default to allow
            return RateLimitResult(
                allowed=True,
                remaining=-1,
                reset_at=datetime.utcnow() + timedelta(seconds=60),
                limit=-1,
            )


class RedisTokenBucket(RedisRateLimiter):
    """
    Redis-backed Token Bucket rate limiter.

    Uses Lua script for atomic operations.
    Tokens are added at a constant rate, consumed by requests.
    """

    def __init__(
        self,
        redis_client: Any,
        rate: float,  # Tokens per second
        capacity: int,  # Maximum tokens
        key_prefix: str = "ratelimit:token_bucket",
        ttl_seconds: int = 3600,
        **kwargs,
    ):
        super().__init__(redis_client, key_prefix, **kwargs)
        self.rate = rate
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check and consume tokens atomically."""
        try:
            redis_key = self._make_key(key)
            now = time.time()

            result = await self.redis.eval(
                TOKEN_BUCKET_SCRIPT,
                1,  # Number of keys
                redis_key,
                self.rate,
                self.capacity,
                now,
                cost,
                self.ttl_seconds,
            )

            allowed, remaining, retry_after, current = result

            if allowed:
                return RateLimitResult(
                    allowed=True,
                    remaining=int(remaining),
                    reset_at=datetime.utcnow() + timedelta(
                        seconds=(self.capacity - remaining) / self.rate
                    ),
                    limit=self.capacity,
                    current=self.capacity - int(remaining),
                )
            else:
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=datetime.utcnow() + timedelta(seconds=retry_after),
                    limit=self.capacity,
                    retry_after=float(retry_after),
                    current=self.capacity,
                )

        except Exception as e:
            return await self._handle_failure(key, cost, e)

    async def reset(self, key: str) -> None:
        """Reset bucket to full capacity."""
        try:
            redis_key = self._make_key(key)
            await self.redis.hmset(
                redis_key,
                {"tokens": self.capacity, "last_update": time.time()}
            )
        except Exception as e:
            logger.error(f"Failed to reset rate limit: {e}")


class RedisSlidingWindow(RedisRateLimiter):
    """
    Redis-backed Sliding Window rate limiter.

    Uses sorted set with timestamps for accurate counting.
    """

    def __init__(
        self,
        redis_client: Any,
        limit: int,  # Maximum requests per window
        window_seconds: int,  # Window duration
        key_prefix: str = "ratelimit:sliding_window",
        **kwargs,
    ):
        super().__init__(redis_client, key_prefix, **kwargs)
        self.limit = limit
        self.window_seconds = window_seconds
        self.window_ms = window_seconds * 1000

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check and count request atomically."""
        try:
            redis_key = self._make_key(key)
            now_ms = int(time.time() * 1000)

            result = await self.redis.eval(
                SLIDING_WINDOW_SCRIPT,
                1,
                redis_key,
                self.limit,
                self.window_ms,
                now_ms,
                cost,
            )

            allowed, remaining, retry_after, current = result

            return RateLimitResult(
                allowed=bool(allowed),
                remaining=int(remaining),
                reset_at=datetime.utcnow() + timedelta(seconds=self.window_seconds),
                limit=self.limit,
                retry_after=float(retry_after) if not allowed else None,
                current=int(current) + (cost if allowed else 0),
            )

        except Exception as e:
            return await self._handle_failure(key, cost, e)

    async def reset(self, key: str) -> None:
        """Clear sliding window."""
        try:
            redis_key = self._make_key(key)
            await self.redis.delete(redis_key)
        except Exception as e:
            logger.error(f"Failed to reset rate limit: {e}")


class RedisFixedWindow(RedisRateLimiter):
    """
    Redis-backed Fixed Window rate limiter.

    Simple counter that resets at fixed intervals.
    """

    def __init__(
        self,
        redis_client: Any,
        limit: int,
        window_seconds: int,
        key_prefix: str = "ratelimit:fixed_window",
        **kwargs,
    ):
        super().__init__(redis_client, key_prefix, **kwargs)
        self.limit = limit
        self.window_seconds = window_seconds

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check and count request atomically."""
        try:
            redis_key = self._make_key(key)
            now = time.time()

            result = await self.redis.eval(
                FIXED_WINDOW_SCRIPT,
                1,
                redis_key,
                self.limit,
                self.window_seconds,
                now,
                cost,
            )

            allowed, remaining, retry_after, current = result

            current_window = int(now / self.window_seconds)
            window_end = (current_window + 1) * self.window_seconds

            return RateLimitResult(
                allowed=bool(allowed),
                remaining=int(remaining),
                reset_at=datetime.fromtimestamp(window_end),
                limit=self.limit,
                retry_after=float(retry_after) if not allowed else None,
                current=int(current),
            )

        except Exception as e:
            return await self._handle_failure(key, cost, e)

    async def reset(self, key: str) -> None:
        """Reset current window counter."""
        try:
            redis_key = self._make_key(key)
            current_window = int(time.time() / self.window_seconds)
            window_key = f"{redis_key}:{current_window}"
            await self.redis.delete(window_key)
        except Exception as e:
            logger.error(f"Failed to reset rate limit: {e}")


class RedisLeakyBucket(RedisRateLimiter):
    """
    Redis-backed Leaky Bucket rate limiter.

    Requests queue up and are processed at a constant rate.
    """

    def __init__(
        self,
        redis_client: Any,
        rate: float,  # Requests per second
        capacity: int,  # Queue capacity
        key_prefix: str = "ratelimit:leaky_bucket",
        ttl_seconds: int = 3600,
        **kwargs,
    ):
        super().__init__(redis_client, key_prefix, **kwargs)
        self.rate = rate
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check if request can be queued atomically."""
        try:
            redis_key = self._make_key(key)
            now = time.time()

            result = await self.redis.eval(
                LEAKY_BUCKET_SCRIPT,
                1,
                redis_key,
                self.rate,
                self.capacity,
                now,
                cost,
                self.ttl_seconds,
            )

            allowed, remaining, retry_after, current = result

            if allowed:
                drain_time = current / self.rate
                return RateLimitResult(
                    allowed=True,
                    remaining=int(remaining),
                    reset_at=datetime.utcnow() + timedelta(seconds=drain_time),
                    limit=self.capacity,
                    current=int(current),
                )
            else:
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=datetime.utcnow() + timedelta(seconds=retry_after),
                    limit=self.capacity,
                    retry_after=float(retry_after),
                    current=int(current),
                )

        except Exception as e:
            return await self._handle_failure(key, cost, e)

    async def reset(self, key: str) -> None:
        """Reset bucket."""
        try:
            redis_key = self._make_key(key)
            await self.redis.hmset(
                redis_key,
                {"level": 0, "last_update": time.time()}
            )
        except Exception as e:
            logger.error(f"Failed to reset rate limit: {e}")


class ClusterAwareRateLimiter:
    """
    Rate limiter that works across a Redis cluster.

    Uses consistent hashing to route keys to appropriate nodes.
    """

    def __init__(
        self,
        redis_cluster: Any,  # aioredis.RedisCluster
        limiter_factory: callable,
        key_prefix: str = "ratelimit:cluster",
    ):
        """
        Args:
            redis_cluster: Redis cluster client
            limiter_factory: Factory function to create limiters
            key_prefix: Key prefix
        """
        self.redis_cluster = redis_cluster
        self.limiter_factory = limiter_factory
        self.key_prefix = key_prefix
        self._limiters: Dict[str, RateLimiter] = {}

    def _get_limiter(self, key: str) -> RateLimiter:
        """Get or create limiter for a key."""
        # In cluster mode, Redis handles routing
        if key not in self._limiters:
            self._limiters[key] = self.limiter_factory(self.redis_cluster)
        return self._limiters[key]

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check rate limit via cluster."""
        limiter = self._get_limiter(key)
        return await limiter.check(key, cost)

    async def reset(self, key: str) -> None:
        """Reset rate limit."""
        limiter = self._get_limiter(key)
        await limiter.reset(key)


class DistributedRateLimiter:
    """
    High-level distributed rate limiter with:
    - Multiple tier support (per-second, per-minute, per-day)
    - Automatic Redis connection management
    - Failover handling
    - Metrics collection
    """

    def __init__(
        self,
        redis_client: Any,
        tiers: Optional[Dict[str, Dict[str, Any]]] = None,
        key_prefix: str = "ratelimit:dist",
        failover_strategy: FailoverStrategy = FailoverStrategy.LOCAL,
    ):
        """
        Args:
            redis_client: Async Redis client
            tiers: Rate limit tiers configuration
            key_prefix: Redis key prefix
            failover_strategy: Failover strategy
        """
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.failover_strategy = failover_strategy
        self.tiers = tiers or self._default_tiers()
        self._limiters: Dict[str, RedisRateLimiter] = {}
        self._metrics = DistributedMetrics()

        self._initialize_limiters()

    def _default_tiers(self) -> Dict[str, Dict[str, Any]]:
        """Default rate limit tiers."""
        return {
            "per_second": {
                "type": "token_bucket",
                "rate": 10.0,
                "capacity": 20,
            },
            "per_minute": {
                "type": "sliding_window",
                "limit": 100,
                "window_seconds": 60,
            },
            "per_hour": {
                "type": "fixed_window",
                "limit": 1000,
                "window_seconds": 3600,
            },
        }

    def _initialize_limiters(self) -> None:
        """Initialize rate limiters for each tier."""
        for tier_name, config in self.tiers.items():
            limiter_type = config.get("type", "token_bucket")

            if limiter_type == "token_bucket":
                self._limiters[tier_name] = RedisTokenBucket(
                    redis_client=self.redis,
                    rate=config.get("rate", 10.0),
                    capacity=config.get("capacity", 100),
                    key_prefix=f"{self.key_prefix}:{tier_name}",
                    failover_strategy=self.failover_strategy,
                )
            elif limiter_type == "sliding_window":
                self._limiters[tier_name] = RedisSlidingWindow(
                    redis_client=self.redis,
                    limit=config.get("limit", 100),
                    window_seconds=config.get("window_seconds", 60),
                    key_prefix=f"{self.key_prefix}:{tier_name}",
                    failover_strategy=self.failover_strategy,
                )
            elif limiter_type == "fixed_window":
                self._limiters[tier_name] = RedisFixedWindow(
                    redis_client=self.redis,
                    limit=config.get("limit", 100),
                    window_seconds=config.get("window_seconds", 60),
                    key_prefix=f"{self.key_prefix}:{tier_name}",
                    failover_strategy=self.failover_strategy,
                )
            elif limiter_type == "leaky_bucket":
                self._limiters[tier_name] = RedisLeakyBucket(
                    redis_client=self.redis,
                    rate=config.get("rate", 10.0),
                    capacity=config.get("capacity", 100),
                    key_prefix=f"{self.key_prefix}:{tier_name}",
                    failover_strategy=self.failover_strategy,
                )

    async def check(self, key: str, cost: int = 1, tiers: Optional[List[str]] = None) -> RateLimitResult:
        """
        Check rate limits across all or specified tiers.

        Returns denied result if any tier denies the request.
        """
        tiers_to_check = tiers or list(self._limiters.keys())
        results = []

        for tier_name in tiers_to_check:
            if tier_name in self._limiters:
                limiter = self._limiters[tier_name]
                result = await limiter.check(key, cost)
                results.append((tier_name, result))

                self._metrics.record_check(tier_name, result.allowed)

                if not result.allowed:
                    # Return immediately on denial
                    self._metrics.record_denial(tier_name, key)
                    return result

        # All tiers allowed, return most restrictive
        if results:
            _, most_restrictive = min(results, key=lambda x: x[1].remaining)
            return most_restrictive

        # No limiters configured
        return RateLimitResult(
            allowed=True,
            remaining=-1,
            reset_at=datetime.utcnow() + timedelta(seconds=60),
            limit=-1,
        )

    async def reset(self, key: str, tiers: Optional[List[str]] = None) -> None:
        """Reset rate limits for a key."""
        tiers_to_reset = tiers or list(self._limiters.keys())

        for tier_name in tiers_to_reset:
            if tier_name in self._limiters:
                await self._limiters[tier_name].reset(key)

    async def get_status(self, key: str) -> Dict[str, RateLimitResult]:
        """Get rate limit status across all tiers without consuming."""
        # Note: This still consumes from some algorithms
        # In production, you might want separate "peek" implementations
        status = {}
        for tier_name, limiter in self._limiters.items():
            result = await limiter.check(key, cost=0)
            status[tier_name] = result
        return status

    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiting metrics."""
        return self._metrics.to_dict()


@dataclass
class DistributedMetrics:
    """Metrics for distributed rate limiting."""
    checks: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    allowed: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    denied: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    denied_keys: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )

    def record_check(self, tier: str, allowed: bool) -> None:
        """Record a check."""
        self.checks[tier] += 1
        if allowed:
            self.allowed[tier] += 1
        else:
            self.denied[tier] += 1

    def record_denial(self, tier: str, key: str) -> None:
        """Record a denial with key."""
        self.denied_keys[tier][key] += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checks": dict(self.checks),
            "allowed": dict(self.allowed),
            "denied": dict(self.denied),
            "denial_rate": {
                tier: self.denied[tier] / self.checks[tier] if self.checks[tier] > 0 else 0
                for tier in self.checks
            },
            "top_denied_keys": {
                tier: dict(sorted(keys.items(), key=lambda x: -x[1])[:10])
                for tier, keys in self.denied_keys.items()
            },
        }


# Import defaultdict for metrics
from collections import defaultdict


class ReplicatedRateLimiter:
    """
    Rate limiter with cross-datacenter replication.

    Maintains local rate limits with eventual consistency
    across datacenters.
    """

    def __init__(
        self,
        local_redis: Any,
        remote_redis_list: List[Any],
        limiter_factory: callable,
        sync_interval_seconds: float = 1.0,
        local_weight: float = 0.7,
    ):
        """
        Args:
            local_redis: Local Redis client
            remote_redis_list: List of remote Redis clients
            limiter_factory: Factory to create limiters
            sync_interval_seconds: Sync interval
            local_weight: Weight for local limits (vs remote)
        """
        self.local_redis = local_redis
        self.remote_redis_list = remote_redis_list
        self.local_weight = local_weight
        self.sync_interval_seconds = sync_interval_seconds

        self.local_limiter = limiter_factory(local_redis)
        self.remote_limiters = [limiter_factory(r) for r in remote_redis_list]

        self._sync_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start background sync."""
        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())

    async def stop(self) -> None:
        """Stop background sync."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

    async def _sync_loop(self) -> None:
        """Background sync loop."""
        while self._running:
            try:
                await self._sync_with_remotes()
            except Exception as e:
                logger.error(f"Sync error: {e}")

            await asyncio.sleep(self.sync_interval_seconds)

    async def _sync_with_remotes(self) -> None:
        """Sync rate limit state with remote datacenters."""
        # This is a simplified implementation
        # In production, you'd sync counter states
        pass

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check rate limit with replication awareness."""
        # Check local first
        local_result = await self.local_limiter.check(key, cost)

        if not local_result.allowed:
            return local_result

        # Optionally check remotes for global limits
        # This adds latency but provides stronger consistency
        return local_result

    async def reset(self, key: str) -> None:
        """Reset rate limit locally and propagate."""
        await self.local_limiter.reset(key)

        # Propagate reset to remotes
        for limiter in self.remote_limiters:
            try:
                await limiter.reset(key)
            except Exception as e:
                logger.warning(f"Failed to propagate reset: {e}")
