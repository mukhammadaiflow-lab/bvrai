"""
Rate Limiter Implementations
============================

Multiple rate limiting algorithms including token bucket, sliding window,
fixed window, leaky bucket, and adaptive rate limiting.

Author: Platform Engineering Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded"""

    def __init__(
        self,
        message: str,
        limit: int,
        remaining: int,
        reset_at: datetime,
        retry_after: float,
    ):
        super().__init__(message)
        self.limit = limit
        self.remaining = remaining
        self.reset_at = reset_at
        self.retry_after = retry_after


@dataclass
class RateLimitConfig:
    """Configuration for rate limiters"""

    requests_per_second: float = 10.0
    burst_size: int = 20
    window_seconds: float = 60.0
    max_tokens: Optional[int] = None
    enable_adaptive: bool = False
    min_rate: float = 1.0
    max_rate: float = 1000.0


@dataclass
class RateLimitResult:
    """Result of a rate limit check"""

    allowed: bool
    limit: int
    remaining: int
    reset_at: datetime
    retry_after: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def headers(self) -> Dict[str, str]:
        """Get rate limit headers"""
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(self.reset_at.timestamp())),
            "Retry-After": str(int(self.retry_after)) if self.retry_after > 0 else "",
        }


class RateLimiter(ABC):
    """
    Base class for rate limiters.

    Usage:
        limiter = TokenBucketLimiter(config)
        result = await limiter.check("user_123")
        if not result.allowed:
            raise RateLimitExceeded(...)
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._logger = structlog.get_logger(self.__class__.__name__)

    @abstractmethod
    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check if request is allowed"""
        pass

    @abstractmethod
    async def reset(self, key: str) -> None:
        """Reset rate limit for a key"""
        pass

    async def acquire(self, key: str, cost: int = 1) -> RateLimitResult:
        """Acquire permission, raising exception if denied"""
        result = await self.check(key, cost)
        if not result.allowed:
            raise RateLimitExceeded(
                message=f"Rate limit exceeded for {key}",
                limit=result.limit,
                remaining=result.remaining,
                reset_at=result.reset_at,
                retry_after=result.retry_after,
            )
        return result


class TokenBucketLimiter(RateLimiter):
    """
    Token bucket rate limiter.

    Tokens are added at a fixed rate and consumed on each request.
    Allows bursting up to the bucket capacity.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        super().__init__(config or RateLimitConfig())
        self._buckets: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check if request is allowed using token bucket"""
        now = time.time()

        with self._lock:
            bucket = self._get_or_create_bucket(key, now)

            # Calculate tokens to add since last request
            time_passed = now - bucket["last_update"]
            tokens_to_add = time_passed * self.config.requests_per_second
            bucket["tokens"] = min(
                self.config.burst_size,
                bucket["tokens"] + tokens_to_add,
            )
            bucket["last_update"] = now

            # Check if we have enough tokens
            if bucket["tokens"] >= cost:
                bucket["tokens"] -= cost
                return RateLimitResult(
                    allowed=True,
                    limit=self.config.burst_size,
                    remaining=int(bucket["tokens"]),
                    reset_at=self._calculate_reset_time(bucket),
                )
            else:
                # Calculate retry after
                tokens_needed = cost - bucket["tokens"]
                retry_after = tokens_needed / self.config.requests_per_second

                return RateLimitResult(
                    allowed=False,
                    limit=self.config.burst_size,
                    remaining=0,
                    reset_at=self._calculate_reset_time(bucket),
                    retry_after=retry_after,
                )

    async def reset(self, key: str) -> None:
        """Reset bucket for a key"""
        with self._lock:
            if key in self._buckets:
                del self._buckets[key]

    def _get_or_create_bucket(self, key: str, now: float) -> Dict[str, float]:
        """Get or create a bucket"""
        if key not in self._buckets:
            self._buckets[key] = {
                "tokens": float(self.config.burst_size),
                "last_update": now,
            }
        return self._buckets[key]

    def _calculate_reset_time(self, bucket: Dict[str, float]) -> datetime:
        """Calculate when bucket will be full again"""
        tokens_needed = self.config.burst_size - bucket["tokens"]
        seconds_to_full = tokens_needed / self.config.requests_per_second
        return datetime.utcnow() + timedelta(seconds=seconds_to_full)


class SlidingWindowLimiter(RateLimiter):
    """
    Sliding window rate limiter.

    Tracks requests in a sliding time window for smooth rate limiting.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        super().__init__(config or RateLimitConfig())
        self._windows: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._max_requests = int(
            self.config.requests_per_second * self.config.window_seconds
        )

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check if request is allowed using sliding window"""
        now = time.time()
        window_start = now - self.config.window_seconds

        with self._lock:
            window = self._windows[key]

            # Remove expired entries
            while window and window[0] < window_start:
                window.popleft()

            current_count = len(window)

            if current_count + cost <= self._max_requests:
                # Add timestamps for each unit of cost
                for _ in range(cost):
                    window.append(now)

                return RateLimitResult(
                    allowed=True,
                    limit=self._max_requests,
                    remaining=self._max_requests - current_count - cost,
                    reset_at=self._calculate_reset_time(window),
                )
            else:
                # Calculate retry after
                if window:
                    oldest = window[0]
                    retry_after = (oldest + self.config.window_seconds) - now
                else:
                    retry_after = 0

                return RateLimitResult(
                    allowed=False,
                    limit=self._max_requests,
                    remaining=0,
                    reset_at=self._calculate_reset_time(window),
                    retry_after=max(0, retry_after),
                )

    async def reset(self, key: str) -> None:
        """Reset window for a key"""
        with self._lock:
            if key in self._windows:
                self._windows[key].clear()

    def _calculate_reset_time(self, window: Deque[float]) -> datetime:
        """Calculate when window will free up"""
        if not window:
            return datetime.utcnow()
        oldest = window[0]
        reset_time = oldest + self.config.window_seconds
        return datetime.fromtimestamp(reset_time)


class FixedWindowLimiter(RateLimiter):
    """
    Fixed window rate limiter.

    Simple counter reset at fixed intervals.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        super().__init__(config or RateLimitConfig())
        self._counters: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._max_requests = int(
            self.config.requests_per_second * self.config.window_seconds
        )

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check if request is allowed using fixed window"""
        now = time.time()
        window_start = int(now / self.config.window_seconds) * self.config.window_seconds

        with self._lock:
            counter = self._counters.get(key)

            # Check if window has expired
            if counter is None or counter["window_start"] < window_start:
                counter = {
                    "window_start": window_start,
                    "count": 0,
                }
                self._counters[key] = counter

            if counter["count"] + cost <= self._max_requests:
                counter["count"] += cost

                return RateLimitResult(
                    allowed=True,
                    limit=self._max_requests,
                    remaining=self._max_requests - counter["count"],
                    reset_at=datetime.fromtimestamp(
                        window_start + self.config.window_seconds
                    ),
                )
            else:
                retry_after = (window_start + self.config.window_seconds) - now

                return RateLimitResult(
                    allowed=False,
                    limit=self._max_requests,
                    remaining=0,
                    reset_at=datetime.fromtimestamp(
                        window_start + self.config.window_seconds
                    ),
                    retry_after=max(0, retry_after),
                )

    async def reset(self, key: str) -> None:
        """Reset counter for a key"""
        with self._lock:
            if key in self._counters:
                del self._counters[key]


class LeakyBucketLimiter(RateLimiter):
    """
    Leaky bucket rate limiter.

    Requests queue up and are processed at a fixed rate.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        super().__init__(config or RateLimitConfig())
        self._buckets: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check if request is allowed using leaky bucket"""
        now = time.time()

        with self._lock:
            bucket = self._get_or_create_bucket(key, now)

            # Leak water since last update
            time_passed = now - bucket["last_update"]
            leaked = time_passed * self.config.requests_per_second
            bucket["water"] = max(0, bucket["water"] - leaked)
            bucket["last_update"] = now

            # Check if we can add more water
            if bucket["water"] + cost <= self.config.burst_size:
                bucket["water"] += cost

                return RateLimitResult(
                    allowed=True,
                    limit=self.config.burst_size,
                    remaining=int(self.config.burst_size - bucket["water"]),
                    reset_at=self._calculate_reset_time(bucket),
                )
            else:
                # Calculate retry after
                overflow = bucket["water"] + cost - self.config.burst_size
                retry_after = overflow / self.config.requests_per_second

                return RateLimitResult(
                    allowed=False,
                    limit=self.config.burst_size,
                    remaining=0,
                    reset_at=self._calculate_reset_time(bucket),
                    retry_after=retry_after,
                )

    async def reset(self, key: str) -> None:
        """Reset bucket for a key"""
        with self._lock:
            if key in self._buckets:
                del self._buckets[key]

    def _get_or_create_bucket(self, key: str, now: float) -> Dict[str, Any]:
        """Get or create a bucket"""
        if key not in self._buckets:
            self._buckets[key] = {
                "water": 0.0,
                "last_update": now,
            }
        return self._buckets[key]

    def _calculate_reset_time(self, bucket: Dict[str, Any]) -> datetime:
        """Calculate when bucket will be empty"""
        seconds_to_empty = bucket["water"] / self.config.requests_per_second
        return datetime.utcnow() + timedelta(seconds=seconds_to_empty)


class AdaptiveLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts limits based on system load.

    Automatically backs off when errors increase and recovers when
    the system is healthy.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        config = config or RateLimitConfig(enable_adaptive=True)
        super().__init__(config)
        self._base_limiter = TokenBucketLimiter(config)
        self._current_rate = config.requests_per_second
        self._error_count = 0
        self._success_count = 0
        self._last_adjustment = time.time()
        self._adjustment_interval = 10.0  # seconds
        self._lock = threading.Lock()

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check with adaptive rate"""
        self._maybe_adjust_rate()

        # Use modified config
        with self._lock:
            self._base_limiter.config.requests_per_second = self._current_rate

        result = await self._base_limiter.check(key, cost)
        result.metadata["adaptive_rate"] = self._current_rate
        return result

    async def reset(self, key: str) -> None:
        """Reset for a key"""
        await self._base_limiter.reset(key)

    def record_success(self) -> None:
        """Record a successful request"""
        with self._lock:
            self._success_count += 1

    def record_error(self) -> None:
        """Record a failed request"""
        with self._lock:
            self._error_count += 1

    def _maybe_adjust_rate(self) -> None:
        """Adjust rate based on error ratio"""
        now = time.time()

        with self._lock:
            if now - self._last_adjustment < self._adjustment_interval:
                return

            total = self._success_count + self._error_count
            if total < 10:
                return

            error_ratio = self._error_count / total

            if error_ratio > 0.1:
                # High error rate - decrease limit
                self._current_rate = max(
                    self.config.min_rate,
                    self._current_rate * 0.8,
                )
            elif error_ratio < 0.01:
                # Low error rate - increase limit
                self._current_rate = min(
                    self.config.max_rate,
                    self._current_rate * 1.1,
                )

            # Reset counters
            self._success_count = 0
            self._error_count = 0
            self._last_adjustment = now


class DistributedLimiter(RateLimiter):
    """
    Distributed rate limiter using Redis for coordination.

    Supports multiple instances sharing the same rate limits.
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        redis_client: Optional[Any] = None,
        key_prefix: str = "ratelimit:",
    ):
        super().__init__(config or RateLimitConfig())
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._local_limiter = SlidingWindowLimiter(self.config)
        self._max_requests = int(
            self.config.requests_per_second * self.config.window_seconds
        )

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check rate limit using distributed storage"""
        if self._redis is None:
            # Fallback to local limiter
            return await self._local_limiter.check(key, cost)

        redis_key = f"{self._key_prefix}{key}"
        now = time.time()
        window_start = now - self.config.window_seconds

        try:
            # Use Lua script for atomic operation
            result = await self._redis_check(redis_key, now, window_start, cost)
            return result
        except Exception as e:
            self._logger.warning(
                "redis_limiter_fallback",
                error=str(e),
            )
            return await self._local_limiter.check(key, cost)

    async def _redis_check(
        self,
        key: str,
        now: float,
        window_start: float,
        cost: int,
    ) -> RateLimitResult:
        """Perform Redis rate limit check"""
        # This would use a Lua script in production
        # Simplified implementation for demonstration

        # In a real implementation:
        # 1. Remove expired entries: ZREMRANGEBYSCORE key 0 window_start
        # 2. Count current entries: ZCARD key
        # 3. If under limit, add new entry: ZADD key now member
        # 4. Set expiry: EXPIRE key window_seconds

        pipe = self._redis.pipeline()

        # Remove old entries
        await pipe.zremrangebyscore(key, 0, window_start)

        # Get current count
        await pipe.zcard(key)

        results = await pipe.execute()
        current_count = results[1]

        if current_count + cost <= self._max_requests:
            # Add new entry
            member = f"{now}:{secrets.token_hex(4)}"
            await self._redis.zadd(key, {member: now})
            await self._redis.expire(key, int(self.config.window_seconds) + 1)

            return RateLimitResult(
                allowed=True,
                limit=self._max_requests,
                remaining=self._max_requests - current_count - cost,
                reset_at=datetime.fromtimestamp(now + self.config.window_seconds),
            )
        else:
            # Get oldest entry for retry calculation
            oldest = await self._redis.zrange(key, 0, 0, withscores=True)
            if oldest:
                oldest_time = oldest[0][1]
                retry_after = (oldest_time + self.config.window_seconds) - now
            else:
                retry_after = 0

            return RateLimitResult(
                allowed=False,
                limit=self._max_requests,
                remaining=0,
                reset_at=datetime.fromtimestamp(now + self.config.window_seconds),
                retry_after=max(0, retry_after),
            )

    async def reset(self, key: str) -> None:
        """Reset rate limit for a key"""
        if self._redis:
            redis_key = f"{self._key_prefix}{key}"
            await self._redis.delete(redis_key)
        await self._local_limiter.reset(key)


# Import for secrets
import secrets


# =============================================================================
# RATE LIMIT POLICIES
# =============================================================================


@dataclass
class RateLimitPolicy:
    """A rate limit policy with conditions"""

    name: str
    limiter: RateLimiter
    key_func: Callable[[Dict[str, Any]], str]
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    priority: int = 0


class RateLimitPolicyManager:
    """
    Manages multiple rate limit policies.

    Applies different limits based on user tier, endpoint, etc.
    """

    def __init__(self):
        self._policies: List[RateLimitPolicy] = []
        self._logger = structlog.get_logger("rate_limit_policy_manager")

    def add_policy(self, policy: RateLimitPolicy) -> None:
        """Add a policy"""
        self._policies.append(policy)
        self._policies.sort(key=lambda p: p.priority, reverse=True)

    async def check(self, context: Dict[str, Any], cost: int = 1) -> RateLimitResult:
        """Check all applicable policies"""
        results = []

        for policy in self._policies:
            # Check condition
            if policy.condition and not policy.condition(context):
                continue

            # Get key
            key = policy.key_func(context)

            # Check limit
            result = await policy.limiter.check(key, cost)
            results.append((policy, result))

            # If any policy denies, return that result
            if not result.allowed:
                return result

        # All policies allowed
        if results:
            # Return the most restrictive remaining count
            return min(results, key=lambda r: r[1].remaining)[1]

        # No policies matched
        return RateLimitResult(
            allowed=True,
            limit=0,
            remaining=0,
            reset_at=datetime.utcnow(),
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_token_bucket(
    rate: float = 10.0,
    burst: int = 20,
) -> TokenBucketLimiter:
    """Create a token bucket limiter"""
    return TokenBucketLimiter(
        RateLimitConfig(
            requests_per_second=rate,
            burst_size=burst,
        )
    )


def create_sliding_window(
    rate: float = 10.0,
    window: float = 60.0,
) -> SlidingWindowLimiter:
    """Create a sliding window limiter"""
    return SlidingWindowLimiter(
        RateLimitConfig(
            requests_per_second=rate,
            window_seconds=window,
        )
    )


def create_fixed_window(
    rate: float = 10.0,
    window: float = 60.0,
) -> FixedWindowLimiter:
    """Create a fixed window limiter"""
    return FixedWindowLimiter(
        RateLimitConfig(
            requests_per_second=rate,
            window_seconds=window,
        )
    )
