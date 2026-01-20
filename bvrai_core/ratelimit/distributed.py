"""
Distributed Rate Limiting with Redis
====================================

Production-grade rate limiting implementation using Redis for distributed
state management. Supports multiple rate limiting algorithms and
graceful degradation.

Features:
    - Token bucket algorithm for smooth rate limiting
    - Sliding window for accurate per-minute/per-day limits
    - Fixed window for simple high-performance limiting
    - Distributed state via Redis
    - Graceful fallback when Redis unavailable
    - Per-key, per-endpoint, and global rate limits
    - Rate limit headers for client visibility

Usage:
    limiter = DistributedRateLimiter(redis_url="redis://localhost:6379/0")

    # Check rate limit
    result = await limiter.check_rate_limit(
        key="user:123",
        limit=100,
        window_seconds=60,
    )

    if not result.allowed:
        raise RateLimitExceeded(retry_after=result.retry_after)

Author: Platform Architecture Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Tuple

import structlog

logger = structlog.get_logger(__name__)

# Try to import Redis
try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    logger.warning(
        "redis package not installed. Rate limiting will use in-memory fallback. "
        "Install with: pip install redis"
    )


class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithms."""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int
    limit: int
    reset_at: float  # Unix timestamp
    retry_after: Optional[float] = None  # Seconds until retry allowed

    def to_headers(self) -> Dict[str, str]:
        """Convert to standard rate limit headers.

        Returns:
            Dict with X-RateLimit-* headers
        """
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(self.reset_at)),
        }
        if self.retry_after is not None:
            headers["Retry-After"] = str(int(self.retry_after))
        return headers


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitResult:
        """Check if request is within rate limit.

        Args:
            key: Unique identifier (e.g., "user:123" or "ip:1.2.3.4")
            limit: Maximum number of requests
            window_seconds: Time window in seconds

        Returns:
            RateLimitResult with allow/deny decision
        """
        pass

    @abstractmethod
    async def reset(self, key: str) -> None:
        """Reset rate limit for a key.

        Args:
            key: The key to reset
        """
        pass


class InMemoryRateLimiter(RateLimiter):
    """In-memory rate limiter for development and fallback.

    Warning: This does not work across multiple instances!
    Use DistributedRateLimiter in production.
    """

    def __init__(self):
        self._buckets: Dict[str, Tuple[int, float]] = {}  # key -> (count, window_start)
        self._lock = asyncio.Lock()

    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitResult:
        """Check rate limit using fixed window algorithm."""
        async with self._lock:
            now = time.time()
            window_start = now - (now % window_seconds)
            window_end = window_start + window_seconds

            # Get or create bucket
            bucket = self._buckets.get(key)
            if bucket is None or bucket[1] < window_start:
                # New window
                self._buckets[key] = (1, window_start)
                return RateLimitResult(
                    allowed=True,
                    remaining=limit - 1,
                    limit=limit,
                    reset_at=window_end,
                )

            count, _ = bucket
            if count >= limit:
                # Rate limited
                retry_after = window_end - now
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    limit=limit,
                    reset_at=window_end,
                    retry_after=retry_after,
                )

            # Allow and increment
            self._buckets[key] = (count + 1, window_start)
            return RateLimitResult(
                allowed=True,
                remaining=limit - count - 1,
                limit=limit,
                reset_at=window_end,
            )

    async def reset(self, key: str) -> None:
        """Reset rate limit for key."""
        async with self._lock:
            self._buckets.pop(key, None)


class DistributedRateLimiter(RateLimiter):
    """Redis-based distributed rate limiter.

    Implements sliding window rate limiting using Redis sorted sets.
    Falls back to in-memory limiting if Redis is unavailable.

    Example:
        limiter = DistributedRateLimiter(
            redis_url="redis://localhost:6379/0",
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
        )

        result = await limiter.check_rate_limit(
            key="api_key:sk-xxx",
            limit=1000,
            window_seconds=60,
        )
    """

    # Lua script for atomic sliding window rate limiting
    SLIDING_WINDOW_SCRIPT = """
    local key = KEYS[1]
    local now = tonumber(ARGV[1])
    local window = tonumber(ARGV[2])
    local limit = tonumber(ARGV[3])
    local member = ARGV[4]

    -- Remove old entries outside the window
    local window_start = now - window
    redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

    -- Count current entries
    local current = redis.call('ZCARD', key)

    if current >= limit then
        -- Get oldest entry to calculate retry time
        local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
        local retry_after = 0
        if oldest and #oldest >= 2 then
            retry_after = tonumber(oldest[2]) + window - now
        end
        return {0, current, retry_after}
    end

    -- Add new entry
    redis.call('ZADD', key, now, member)
    redis.call('EXPIRE', key, window + 1)

    return {1, current + 1, 0}
    """

    # Lua script for token bucket rate limiting
    TOKEN_BUCKET_SCRIPT = """
    local key = KEYS[1]
    local now = tonumber(ARGV[1])
    local rate = tonumber(ARGV[2])  -- tokens per second
    local burst = tonumber(ARGV[3])  -- max tokens (bucket size)
    local requested = tonumber(ARGV[4])

    -- Get current bucket state
    local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
    local tokens = tonumber(bucket[1]) or burst
    local last_update = tonumber(bucket[2]) or now

    -- Calculate tokens to add based on time elapsed
    local elapsed = now - last_update
    tokens = math.min(burst, tokens + elapsed * rate)

    -- Check if we have enough tokens
    local allowed = 0
    if tokens >= requested then
        tokens = tokens - requested
        allowed = 1
    end

    -- Update bucket
    redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
    redis.call('EXPIRE', key, math.ceil(burst / rate) + 1)

    -- Calculate retry time if denied
    local retry_after = 0
    if allowed == 0 then
        retry_after = (requested - tokens) / rate
    end

    return {allowed, math.floor(tokens), retry_after}
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW,
        key_prefix: str = "ratelimit:",
        fallback_enabled: bool = True,
    ):
        """Initialize distributed rate limiter.

        Args:
            redis_url: Redis connection URL
            algorithm: Rate limiting algorithm to use
            key_prefix: Prefix for Redis keys
            fallback_enabled: Whether to fall back to in-memory if Redis unavailable
        """
        self._redis_url = redis_url
        self._algorithm = algorithm
        self._key_prefix = key_prefix
        self._fallback_enabled = fallback_enabled
        self._redis: Optional["redis.Redis"] = None
        self._fallback = InMemoryRateLimiter() if fallback_enabled else None
        self._scripts_loaded = False
        self._sliding_window_sha: Optional[str] = None
        self._token_bucket_sha: Optional[str] = None

    async def _ensure_connected(self) -> bool:
        """Ensure Redis connection is established.

        Returns:
            True if connected, False otherwise
        """
        if not HAS_REDIS:
            return False

        if self._redis is None:
            try:
                self._redis = redis.from_url(
                    self._redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
                # Test connection
                await self._redis.ping()

                # Load Lua scripts
                self._sliding_window_sha = await self._redis.script_load(
                    self.SLIDING_WINDOW_SCRIPT
                )
                self._token_bucket_sha = await self._redis.script_load(
                    self.TOKEN_BUCKET_SCRIPT
                )
                self._scripts_loaded = True

                logger.info("Redis rate limiter connected", url=self._redis_url)
                return True

            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._redis = None
                return False

        try:
            await self._redis.ping()
            return True
        except Exception:
            self._redis = None
            return False

    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitResult:
        """Check rate limit using Redis.

        Args:
            key: Unique identifier for rate limiting
            limit: Maximum requests in window
            window_seconds: Window size in seconds

        Returns:
            RateLimitResult with decision and metadata
        """
        full_key = f"{self._key_prefix}{key}"
        now = time.time()

        # Try Redis
        if await self._ensure_connected():
            try:
                if self._algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                    return await self._sliding_window_check(
                        full_key, limit, window_seconds, now
                    )
                elif self._algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                    return await self._token_bucket_check(
                        full_key, limit, window_seconds, now
                    )
                else:
                    return await self._fixed_window_check(
                        full_key, limit, window_seconds, now
                    )

            except Exception as e:
                logger.error(f"Redis rate limit check failed: {e}")

        # Fall back to in-memory
        if self._fallback:
            logger.warning("Using in-memory rate limiter fallback")
            return await self._fallback.check_rate_limit(key, limit, window_seconds)

        # If no fallback, allow request (fail open)
        logger.warning("Rate limiting unavailable, allowing request")
        return RateLimitResult(
            allowed=True,
            remaining=limit,
            limit=limit,
            reset_at=now + window_seconds,
        )

    async def _sliding_window_check(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        now: float,
    ) -> RateLimitResult:
        """Sliding window rate limit check."""
        # Generate unique member ID for this request
        member = f"{now}:{id(asyncio.current_task())}"

        result = await self._redis.evalsha(
            self._sliding_window_sha,
            1,
            key,
            str(now),
            str(window_seconds),
            str(limit),
            member,
        )

        allowed = bool(result[0])
        current = int(result[1])
        retry_after = float(result[2]) if result[2] else None

        return RateLimitResult(
            allowed=allowed,
            remaining=max(0, limit - current),
            limit=limit,
            reset_at=now + window_seconds,
            retry_after=retry_after if not allowed else None,
        )

    async def _token_bucket_check(
        self,
        key: str,
        limit: int,  # Used as burst size
        window_seconds: int,  # Rate = limit / window_seconds
        now: float,
    ) -> RateLimitResult:
        """Token bucket rate limit check."""
        rate = limit / window_seconds  # Tokens per second
        burst = limit

        result = await self._redis.evalsha(
            self._token_bucket_sha,
            1,
            key,
            str(now),
            str(rate),
            str(burst),
            "1",  # Request 1 token
        )

        allowed = bool(result[0])
        tokens = int(result[1])
        retry_after = float(result[2]) if result[2] else None

        return RateLimitResult(
            allowed=allowed,
            remaining=tokens,
            limit=limit,
            reset_at=now + (limit / rate),
            retry_after=retry_after if not allowed else None,
        )

    async def _fixed_window_check(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        now: float,
    ) -> RateLimitResult:
        """Simple fixed window rate limit check."""
        window_key = f"{key}:{int(now // window_seconds)}"
        window_end = (int(now // window_seconds) + 1) * window_seconds

        # Increment counter
        count = await self._redis.incr(window_key)

        if count == 1:
            # New window, set expiry
            await self._redis.expire(window_key, window_seconds + 1)

        if count > limit:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=limit,
                reset_at=window_end,
                retry_after=window_end - now,
            )

        return RateLimitResult(
            allowed=True,
            remaining=limit - count,
            limit=limit,
            reset_at=window_end,
        )

    async def reset(self, key: str) -> None:
        """Reset rate limit for key."""
        if await self._ensure_connected():
            full_key = f"{self._key_prefix}{key}"
            try:
                await self._redis.delete(full_key)
            except Exception as e:
                logger.error(f"Failed to reset rate limit: {e}")

        if self._fallback:
            await self._fallback.reset(key)

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


# Factory function for creating rate limiters
def create_rate_limiter(
    redis_url: Optional[str] = None,
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW,
) -> RateLimiter:
    """Create appropriate rate limiter based on configuration.

    Args:
        redis_url: Redis URL (if None, uses in-memory)
        algorithm: Rate limiting algorithm

    Returns:
        RateLimiter instance
    """
    if redis_url and HAS_REDIS:
        return DistributedRateLimiter(
            redis_url=redis_url,
            algorithm=algorithm,
        )
    return InMemoryRateLimiter()


__all__ = [
    "RateLimiter",
    "InMemoryRateLimiter",
    "DistributedRateLimiter",
    "RateLimitResult",
    "RateLimitAlgorithm",
    "create_rate_limiter",
]
