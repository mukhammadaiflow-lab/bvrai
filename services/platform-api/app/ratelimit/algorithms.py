"""
Rate Limiting Algorithms

Implementation of various rate limiting algorithms:
- Token Bucket: Smooth bursting
- Sliding Window: Accurate counting
- Fixed Window: Simple counting
- Leaky Bucket: Constant rate
"""

from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from collections import defaultdict
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int
    reset_at: datetime
    limit: int
    retry_after: Optional[float] = None  # Seconds until retry
    current: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "remaining": self.remaining,
            "reset_at": self.reset_at.isoformat(),
            "limit": self.limit,
            "retry_after": self.retry_after,
            "current": self.current,
        }

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(self.reset_at.timestamp())),
        }
        if self.retry_after is not None:
            headers["Retry-After"] = str(int(self.retry_after))
        return headers


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """
        Check if request is allowed.

        Args:
            key: Identifier for rate limiting (e.g., user_id, IP)
            cost: Cost of this request (default 1)

        Returns:
            RateLimitResult indicating if allowed and remaining quota
        """
        pass

    @abstractmethod
    async def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        pass

    async def is_allowed(self, key: str, cost: int = 1) -> bool:
        """Simple check if request is allowed."""
        result = await self.check(key, cost)
        return result.allowed


class TokenBucket(RateLimiter):
    """
    Token Bucket rate limiter.

    Tokens are added at a constant rate. Each request consumes tokens.
    Allows bursting up to bucket capacity.

    Good for: APIs that allow occasional bursts
    """

    def __init__(
        self,
        rate: float,  # Tokens per second
        capacity: int,  # Maximum tokens (burst size)
    ):
        """
        Args:
            rate: Number of tokens added per second
            capacity: Maximum tokens the bucket can hold
        """
        self.rate = rate
        self.capacity = capacity
        self._buckets: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check and consume tokens."""
        async with self._lock:
            now = time.time()

            # Get or create bucket
            if key not in self._buckets:
                self._buckets[key] = {
                    "tokens": self.capacity,
                    "last_update": now,
                }

            bucket = self._buckets[key]

            # Calculate tokens to add since last update
            elapsed = now - bucket["last_update"]
            tokens_to_add = elapsed * self.rate
            bucket["tokens"] = min(self.capacity, bucket["tokens"] + tokens_to_add)
            bucket["last_update"] = now

            # Check if enough tokens
            if bucket["tokens"] >= cost:
                bucket["tokens"] -= cost
                remaining = int(bucket["tokens"])
                return RateLimitResult(
                    allowed=True,
                    remaining=remaining,
                    reset_at=datetime.utcnow() + timedelta(seconds=(self.capacity - remaining) / self.rate),
                    limit=self.capacity,
                    current=self.capacity - remaining,
                )
            else:
                # Calculate wait time
                tokens_needed = cost - bucket["tokens"]
                wait_time = tokens_needed / self.rate

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=datetime.utcnow() + timedelta(seconds=wait_time),
                    limit=self.capacity,
                    retry_after=wait_time,
                    current=self.capacity,
                )

    async def reset(self, key: str) -> None:
        """Reset bucket to full capacity."""
        async with self._lock:
            self._buckets[key] = {
                "tokens": self.capacity,
                "last_update": time.time(),
            }

    async def get_tokens(self, key: str) -> float:
        """Get current token count for a key."""
        async with self._lock:
            if key not in self._buckets:
                return float(self.capacity)

            bucket = self._buckets[key]
            elapsed = time.time() - bucket["last_update"]
            tokens = min(self.capacity, bucket["tokens"] + elapsed * self.rate)
            return tokens


class SlidingWindow(RateLimiter):
    """
    Sliding Window rate limiter.

    Counts requests in a sliding time window.
    More accurate than fixed window, prevents boundary issues.

    Good for: Accurate rate limiting without boundary spikes
    """

    def __init__(
        self,
        limit: int,
        window_seconds: int,
        precision: int = 10,  # Number of sub-windows
    ):
        """
        Args:
            limit: Maximum requests per window
            window_seconds: Window duration in seconds
            precision: Number of sub-windows for accuracy
        """
        self.limit = limit
        self.window_seconds = window_seconds
        self.precision = precision
        self.sub_window_seconds = window_seconds / precision

        self._counters: Dict[str, Dict[int, int]] = defaultdict(dict)
        self._lock = asyncio.Lock()

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check and count request."""
        async with self._lock:
            now = time.time()
            current_sub_window = int(now / self.sub_window_seconds)
            window_start = current_sub_window - self.precision + 1

            # Clean old sub-windows
            counters = self._counters[key]
            old_keys = [k for k in counters if k < window_start]
            for k in old_keys:
                del counters[k]

            # Calculate current count with weighted sub-windows
            total = 0
            for sub_window, count in counters.items():
                if sub_window >= window_start:
                    # Weight by how much of this sub-window is in our window
                    weight = 1.0
                    if sub_window == window_start:
                        # Partial weight for first sub-window
                        elapsed_in_window = (now % self.sub_window_seconds) / self.sub_window_seconds
                        weight = elapsed_in_window
                    total += count * weight

            # Check limit
            if total + cost <= self.limit:
                # Increment counter
                counters[current_sub_window] = counters.get(current_sub_window, 0) + cost

                return RateLimitResult(
                    allowed=True,
                    remaining=self.limit - int(total + cost),
                    reset_at=datetime.utcnow() + timedelta(seconds=self.window_seconds),
                    limit=self.limit,
                    current=int(total + cost),
                )
            else:
                # Calculate when oldest requests expire
                retry_after = self.sub_window_seconds * (1 - (now % self.sub_window_seconds) / self.sub_window_seconds)

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=datetime.utcnow() + timedelta(seconds=retry_after),
                    limit=self.limit,
                    retry_after=retry_after,
                    current=int(total),
                )

    async def reset(self, key: str) -> None:
        """Reset counters for a key."""
        async with self._lock:
            self._counters[key] = {}


class FixedWindow(RateLimiter):
    """
    Fixed Window rate limiter.

    Simple counter that resets at fixed intervals.
    Can have boundary issues (2x burst at window boundary).

    Good for: Simple use cases, low memory
    """

    def __init__(
        self,
        limit: int,
        window_seconds: int,
    ):
        """
        Args:
            limit: Maximum requests per window
            window_seconds: Window duration in seconds
        """
        self.limit = limit
        self.window_seconds = window_seconds
        self._windows: Dict[str, Tuple[int, int]] = {}  # key -> (window_id, count)
        self._lock = asyncio.Lock()

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check and count request."""
        async with self._lock:
            now = time.time()
            current_window = int(now / self.window_seconds)
            window_end = (current_window + 1) * self.window_seconds

            # Get current window data
            if key in self._windows:
                window_id, count = self._windows[key]
                if window_id != current_window:
                    # New window, reset count
                    window_id = current_window
                    count = 0
            else:
                window_id = current_window
                count = 0

            # Check limit
            if count + cost <= self.limit:
                count += cost
                self._windows[key] = (window_id, count)

                return RateLimitResult(
                    allowed=True,
                    remaining=self.limit - count,
                    reset_at=datetime.fromtimestamp(window_end),
                    limit=self.limit,
                    current=count,
                )
            else:
                retry_after = window_end - now

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=datetime.fromtimestamp(window_end),
                    limit=self.limit,
                    retry_after=retry_after,
                    current=count,
                )

    async def reset(self, key: str) -> None:
        """Reset counter for a key."""
        async with self._lock:
            if key in self._windows:
                del self._windows[key]


class LeakyBucket(RateLimiter):
    """
    Leaky Bucket rate limiter.

    Requests queue up and are processed at a constant rate.
    Good for smoothing out traffic.

    Good for: Constant throughput, traffic shaping
    """

    def __init__(
        self,
        rate: float,  # Requests per second (leak rate)
        capacity: int,  # Queue capacity
    ):
        """
        Args:
            rate: Requests processed per second
            capacity: Maximum queued requests
        """
        self.rate = rate
        self.capacity = capacity
        self._buckets: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check if request can be queued."""
        async with self._lock:
            now = time.time()

            # Get or create bucket
            if key not in self._buckets:
                self._buckets[key] = {
                    "level": 0.0,
                    "last_update": now,
                }

            bucket = self._buckets[key]

            # Calculate leaked amount since last update
            elapsed = now - bucket["last_update"]
            leaked = elapsed * self.rate
            bucket["level"] = max(0, bucket["level"] - leaked)
            bucket["last_update"] = now

            # Check if room in bucket
            if bucket["level"] + cost <= self.capacity:
                bucket["level"] += cost
                remaining = int(self.capacity - bucket["level"])

                # Calculate when bucket will be empty
                drain_time = bucket["level"] / self.rate

                return RateLimitResult(
                    allowed=True,
                    remaining=remaining,
                    reset_at=datetime.utcnow() + timedelta(seconds=drain_time),
                    limit=self.capacity,
                    current=int(bucket["level"]),
                )
            else:
                # Calculate wait time
                overflow = bucket["level"] + cost - self.capacity
                wait_time = overflow / self.rate

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=datetime.utcnow() + timedelta(seconds=wait_time),
                    limit=self.capacity,
                    retry_after=wait_time,
                    current=int(bucket["level"]),
                )

    async def reset(self, key: str) -> None:
        """Reset bucket for a key."""
        async with self._lock:
            self._buckets[key] = {
                "level": 0.0,
                "last_update": time.time(),
            }


class CompositeRateLimiter(RateLimiter):
    """
    Combines multiple rate limiters.

    Request is allowed only if all limiters allow it.
    Useful for multi-tier limits (per-second, per-minute, per-day).
    """

    def __init__(self, limiters: Dict[str, RateLimiter]):
        """
        Args:
            limiters: Named rate limiters to combine
        """
        self.limiters = limiters

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check all limiters."""
        results = []
        for name, limiter in self.limiters.items():
            result = await limiter.check(key, cost)
            results.append((name, result))

        # Find the most restrictive result
        denied_results = [(n, r) for n, r in results if not r.allowed]

        if denied_results:
            # Return the one with longest retry time
            name, worst = max(denied_results, key=lambda x: x[1].retry_after or 0)
            return worst

        # All allowed, return the one with least remaining
        name, least = min(results, key=lambda x: x[1].remaining)
        return least

    async def reset(self, key: str) -> None:
        """Reset all limiters."""
        for limiter in self.limiters.values():
            await limiter.reset(key)


class RateLimiterRegistry:
    """
    Registry for managing multiple rate limiters.

    Usage:
        registry = RateLimiterRegistry()

        # Register limiters
        registry.register("api", TokenBucket(rate=10, capacity=100))
        registry.register("auth", FixedWindow(limit=5, window_seconds=60))

        # Check limits
        result = await registry.check("api", user_id)
    """

    def __init__(self):
        self._limiters: Dict[str, RateLimiter] = {}

    def register(self, name: str, limiter: RateLimiter) -> None:
        """Register a rate limiter."""
        self._limiters[name] = limiter

    def get(self, name: str) -> Optional[RateLimiter]:
        """Get a rate limiter by name."""
        return self._limiters.get(name)

    async def check(self, name: str, key: str, cost: int = 1) -> RateLimitResult:
        """Check a specific limiter."""
        limiter = self._limiters.get(name)
        if limiter is None:
            raise ValueError(f"Unknown rate limiter: {name}")
        return await limiter.check(key, cost)

    async def check_all(self, key: str, cost: int = 1) -> Dict[str, RateLimitResult]:
        """Check all limiters."""
        results = {}
        for name, limiter in self._limiters.items():
            results[name] = await limiter.check(key, cost)
        return results
