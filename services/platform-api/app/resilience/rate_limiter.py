"""Rate limiting for API and resource protection."""

from typing import Optional, Dict, Any, Callable, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import time
import logging
import functools
import hashlib

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        limit: int,
        window: int,
        retry_after: float,
    ):
        super().__init__(message)
        self.limit = limit
        self.window = window
        self.retry_after = retry_after


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int
    reset_at: datetime
    retry_after: Optional[float] = None


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests: int  # Number of requests allowed
    window_seconds: int  # Time window in seconds
    burst: Optional[int] = None  # Burst allowance (token bucket)
    key_prefix: str = ""  # Prefix for keys


class TokenBucket:
    """
    Token bucket rate limiter.

    Allows bursting up to bucket capacity while maintaining
    average rate over time.
    """

    def __init__(
        self,
        rate: float,  # Tokens per second
        capacity: int,  # Maximum tokens
    ):
        self.rate = rate
        self.capacity = capacity
        self._tokens = float(capacity)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update

            # Add tokens based on elapsed time
            self._tokens = min(
                self.capacity,
                self._tokens + elapsed * self.rate,
            )
            self._last_update = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    @property
    def available(self) -> float:
        """Get available tokens."""
        return self._tokens

    def time_until_available(self, tokens: int = 1) -> float:
        """Calculate time until tokens are available."""
        if self._tokens >= tokens:
            return 0.0
        needed = tokens - self._tokens
        return needed / self.rate


class SlidingWindowCounter:
    """
    Sliding window rate limiter using counters.

    More memory efficient than sliding log but slightly less accurate.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: int,
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self._current_count = 0
        self._previous_count = 0
        self._current_window_start = 0
        self._lock = asyncio.Lock()

    def _get_window_start(self) -> int:
        """Get current window start timestamp."""
        now = int(time.time())
        return now - (now % self.window_seconds)

    async def check(self) -> RateLimitResult:
        """Check if request is allowed."""
        async with self._lock:
            now = int(time.time())
            window_start = self._get_window_start()

            # Check if we need to roll the window
            if window_start != self._current_window_start:
                if window_start - self._current_window_start >= self.window_seconds:
                    # Completely new window
                    self._previous_count = 0
                else:
                    # Roll to next window
                    self._previous_count = self._current_count
                self._current_count = 0
                self._current_window_start = window_start

            # Calculate weighted count
            elapsed_ratio = (now - window_start) / self.window_seconds
            weighted_count = (
                self._previous_count * (1 - elapsed_ratio) +
                self._current_count
            )

            remaining = max(0, self.limit - int(weighted_count))
            reset_at = datetime.fromtimestamp(window_start + self.window_seconds)

            if weighted_count < self.limit:
                self._current_count += 1
                return RateLimitResult(
                    allowed=True,
                    remaining=remaining - 1,
                    reset_at=reset_at,
                )
            else:
                retry_after = self.window_seconds - (now - window_start)
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=reset_at,
                    retry_after=retry_after,
                )


class FixedWindowCounter:
    """
    Simple fixed window rate limiter.

    Most memory efficient but can allow 2x burst at window boundaries.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: int,
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self._count = 0
        self._window_start = 0
        self._lock = asyncio.Lock()

    async def check(self) -> RateLimitResult:
        """Check if request is allowed."""
        async with self._lock:
            now = int(time.time())
            window_start = now - (now % self.window_seconds)

            # Reset if new window
            if window_start != self._window_start:
                self._count = 0
                self._window_start = window_start

            reset_at = datetime.fromtimestamp(window_start + self.window_seconds)

            if self._count < self.limit:
                self._count += 1
                return RateLimitResult(
                    allowed=True,
                    remaining=self.limit - self._count,
                    reset_at=reset_at,
                )
            else:
                retry_after = self.window_seconds - (now - window_start)
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=reset_at,
                    retry_after=retry_after,
                )


class SlidingWindowLog:
    """
    Sliding window rate limiter using a log of timestamps.

    Most accurate but uses more memory.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: int,
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self._timestamps: List[float] = []
        self._lock = asyncio.Lock()

    async def check(self) -> RateLimitResult:
        """Check if request is allowed."""
        async with self._lock:
            now = time.time()
            window_start = now - self.window_seconds

            # Remove old timestamps
            self._timestamps = [
                ts for ts in self._timestamps
                if ts > window_start
            ]

            reset_at = datetime.fromtimestamp(now + self.window_seconds)

            if len(self._timestamps) < self.limit:
                self._timestamps.append(now)
                return RateLimitResult(
                    allowed=True,
                    remaining=self.limit - len(self._timestamps),
                    reset_at=reset_at,
                )
            else:
                # Calculate when oldest will expire
                oldest = min(self._timestamps)
                retry_after = oldest + self.window_seconds - now
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=reset_at,
                    retry_after=retry_after,
                )


class RateLimiter:
    """
    Main rate limiter with multiple strategies and key support.

    Usage:
        limiter = RateLimiter(
            default_config=RateLimitConfig(requests=100, window_seconds=60)
        )

        # Check rate limit for a key
        result = await limiter.check("user:123")
        if not result.allowed:
            raise RateLimitExceeded(...)

        # Or use decorator
        @limiter.limit("user:{user_id}")
        async def api_handler(user_id: str):
            ...
    """

    def __init__(
        self,
        default_config: RateLimitConfig,
        strategy: str = "sliding_window",  # "token_bucket", "fixed_window", "sliding_log"
    ):
        self.default_config = default_config
        self.strategy = strategy
        self._limiters: Dict[str, Any] = {}
        self._configs: Dict[str, RateLimitConfig] = {}
        self._lock = asyncio.Lock()

    def set_config(self, key_pattern: str, config: RateLimitConfig) -> None:
        """Set custom config for a key pattern."""
        self._configs[key_pattern] = config

    def _get_config(self, key: str) -> RateLimitConfig:
        """Get config for a key."""
        # Check for pattern matches
        for pattern, config in self._configs.items():
            if pattern in key or key.startswith(pattern):
                return config
        return self.default_config

    async def _get_limiter(self, key: str) -> Any:
        """Get or create a limiter for a key."""
        async with self._lock:
            if key not in self._limiters:
                config = self._get_config(key)

                if self.strategy == "token_bucket":
                    rate = config.requests / config.window_seconds
                    capacity = config.burst or config.requests
                    self._limiters[key] = TokenBucket(rate, capacity)
                elif self.strategy == "fixed_window":
                    self._limiters[key] = FixedWindowCounter(
                        config.requests,
                        config.window_seconds,
                    )
                elif self.strategy == "sliding_log":
                    self._limiters[key] = SlidingWindowLog(
                        config.requests,
                        config.window_seconds,
                    )
                else:  # sliding_window (default)
                    self._limiters[key] = SlidingWindowCounter(
                        config.requests,
                        config.window_seconds,
                    )

            return self._limiters[key]

    async def check(self, key: str) -> RateLimitResult:
        """Check rate limit for a key."""
        limiter = await self._get_limiter(key)

        if isinstance(limiter, TokenBucket):
            allowed = await limiter.acquire()
            return RateLimitResult(
                allowed=allowed,
                remaining=int(limiter.available),
                reset_at=datetime.utcnow() + timedelta(seconds=1),
                retry_after=limiter.time_until_available() if not allowed else None,
            )
        else:
            return await limiter.check()

    async def is_allowed(self, key: str) -> bool:
        """Simple check if request is allowed."""
        result = await self.check(key)
        return result.allowed

    def limit(
        self,
        key_pattern: str,
        config: Optional[RateLimitConfig] = None,
    ) -> Callable:
        """
        Decorator for rate limiting.

        Key pattern can include format placeholders that will be
        filled from function arguments.

        Usage:
            @limiter.limit("user:{user_id}")
            async def handler(user_id: str):
                ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Build key from pattern and arguments
                key = key_pattern.format(**kwargs)

                result = await self.check(key)
                if not result.allowed:
                    cfg = config or self._get_config(key)
                    raise RateLimitExceeded(
                        f"Rate limit exceeded for {key}",
                        limit=cfg.requests,
                        window=cfg.window_seconds,
                        retry_after=result.retry_after or 0,
                    )

                return await func(*args, **kwargs)
            return wrapper
        return decorator


class MultiTierRateLimiter:
    """
    Multi-tier rate limiter with different limits for different tiers.

    Usage:
        limiter = MultiTierRateLimiter()
        limiter.add_tier("free", RateLimitConfig(requests=100, window_seconds=3600))
        limiter.add_tier("pro", RateLimitConfig(requests=1000, window_seconds=3600))
        limiter.add_tier("enterprise", RateLimitConfig(requests=10000, window_seconds=3600))

        result = await limiter.check("user:123", tier="pro")
    """

    def __init__(self):
        self._tiers: Dict[str, RateLimiter] = {}

    def add_tier(self, name: str, config: RateLimitConfig) -> None:
        """Add a rate limit tier."""
        self._tiers[name] = RateLimiter(config)

    async def check(self, key: str, tier: str) -> RateLimitResult:
        """Check rate limit for a key in a specific tier."""
        if tier not in self._tiers:
            raise ValueError(f"Unknown tier: {tier}")
        return await self._tiers[tier].check(key)

    async def is_allowed(self, key: str, tier: str) -> bool:
        """Simple check if request is allowed."""
        result = await self.check(key, tier)
        return result.allowed


class IPRateLimiter:
    """
    Rate limiter specifically for IP-based limiting.

    Includes support for trusted proxies and X-Forwarded-For.
    """

    def __init__(
        self,
        config: RateLimitConfig,
        trusted_proxies: Optional[List[str]] = None,
        whitelist: Optional[List[str]] = None,
        blacklist: Optional[List[str]] = None,
    ):
        self._limiter = RateLimiter(config)
        self.trusted_proxies = set(trusted_proxies or [])
        self.whitelist = set(whitelist or [])
        self.blacklist = set(blacklist or [])

    def get_client_ip(
        self,
        remote_addr: str,
        x_forwarded_for: Optional[str] = None,
    ) -> str:
        """Extract client IP from request."""
        if x_forwarded_for and remote_addr in self.trusted_proxies:
            # Get first non-trusted IP from X-Forwarded-For
            ips = [ip.strip() for ip in x_forwarded_for.split(",")]
            for ip in ips:
                if ip not in self.trusted_proxies:
                    return ip
        return remote_addr

    async def check(
        self,
        remote_addr: str,
        x_forwarded_for: Optional[str] = None,
    ) -> RateLimitResult:
        """Check rate limit for IP."""
        ip = self.get_client_ip(remote_addr, x_forwarded_for)

        # Check whitelist
        if ip in self.whitelist:
            return RateLimitResult(
                allowed=True,
                remaining=999999,
                reset_at=datetime.utcnow() + timedelta(hours=1),
            )

        # Check blacklist
        if ip in self.blacklist:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=datetime.utcnow() + timedelta(hours=24),
                retry_after=86400,
            )

        return await self._limiter.check(f"ip:{ip}")


# Predefined configurations
API_RATE_LIMIT = RateLimitConfig(
    requests=1000,
    window_seconds=3600,
    key_prefix="api:",
)

CALL_RATE_LIMIT = RateLimitConfig(
    requests=100,
    window_seconds=60,
    key_prefix="calls:",
)

WEBHOOK_RATE_LIMIT = RateLimitConfig(
    requests=10000,
    window_seconds=3600,
    key_prefix="webhooks:",
)

AUTH_RATE_LIMIT = RateLimitConfig(
    requests=10,
    window_seconds=60,
    key_prefix="auth:",
)
