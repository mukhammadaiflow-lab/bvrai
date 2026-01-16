"""
Rate Limiting Middleware

This module provides sophisticated rate limiting for the API,
supporting multiple strategies and storage backends.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitScope(str, Enum):
    """Scope for rate limiting."""

    GLOBAL = "global"
    ORGANIZATION = "organization"
    USER = "user"
    API_KEY = "api_key"
    IP = "ip"
    ENDPOINT = "endpoint"


@dataclass
class RateLimitRule:
    """A rate limiting rule."""

    name: str
    requests: int  # Max requests
    window_seconds: int  # Time window
    scope: RateLimitScope = RateLimitScope.API_KEY
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW

    # Optional endpoint matching
    endpoints: Optional[List[str]] = None  # None = all endpoints
    methods: Optional[List[str]] = None  # None = all methods

    # Burst allowance
    burst_size: Optional[int] = None

    # Cost per request (for weighted limiting)
    cost: int = 1

    def matches_endpoint(self, endpoint: str, method: str) -> bool:
        """Check if rule applies to an endpoint."""
        if self.endpoints and endpoint not in self.endpoints:
            return False
        if self.methods and method.upper() not in self.methods:
            return False
        return True


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int
    limit: int
    reset_at: datetime
    retry_after_seconds: Optional[int] = None
    rule_name: Optional[str] = None

    def to_headers(self) -> Dict[str, str]:
        """Convert to response headers."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(self.reset_at.timestamp())),
        }
        if self.retry_after_seconds:
            headers["Retry-After"] = str(self.retry_after_seconds)
        return headers


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    enabled: bool = Field(default=True, description="Enable rate limiting")
    default_requests_per_minute: int = Field(
        default=60,
        description="Default requests per minute",
    )
    default_requests_per_day: int = Field(
        default=10000,
        description="Default requests per day",
    )
    burst_multiplier: float = Field(
        default=1.5,
        description="Burst multiplier for token bucket",
    )
    include_headers: bool = Field(
        default=True,
        description="Include rate limit headers in responses",
    )

    # Strategy settings
    strategy: RateLimitStrategy = Field(
        default=RateLimitStrategy.SLIDING_WINDOW,
        description="Default rate limiting strategy",
    )

    # Exemptions
    exempt_ips: List[str] = Field(
        default_factory=list,
        description="IPs exempt from rate limiting",
    )
    exempt_api_keys: List[str] = Field(
        default_factory=list,
        description="API key prefixes exempt from rate limiting",
    )


class RateLimitStore(ABC):
    """Abstract base for rate limit storage."""

    @abstractmethod
    async def get_count(self, key: str, window_start: float) -> int:
        """Get request count for a key in the current window."""
        pass

    @abstractmethod
    async def increment(
        self,
        key: str,
        window_start: float,
        ttl_seconds: int,
        cost: int = 1,
    ) -> int:
        """Increment and return new count."""
        pass

    @abstractmethod
    async def get_token_bucket(
        self,
        key: str,
    ) -> Tuple[float, float]:
        """Get token bucket state (tokens, last_update)."""
        pass

    @abstractmethod
    async def set_token_bucket(
        self,
        key: str,
        tokens: float,
        last_update: float,
        ttl_seconds: int,
    ) -> None:
        """Set token bucket state."""
        pass


class InMemoryRateLimitStore(RateLimitStore):
    """In-memory rate limit store."""

    def __init__(self):
        """Initialize store."""
        self._counts: Dict[str, Dict[float, int]] = {}
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock = asyncio.Lock()

    async def get_count(self, key: str, window_start: float) -> int:
        """Get request count for a key."""
        async with self._lock:
            if key not in self._counts:
                return 0

            # Clean old windows
            self._counts[key] = {
                ws: count
                for ws, count in self._counts[key].items()
                if ws >= window_start - 3600  # Keep 1 hour of history
            }

            return self._counts[key].get(window_start, 0)

    async def increment(
        self,
        key: str,
        window_start: float,
        ttl_seconds: int,
        cost: int = 1,
    ) -> int:
        """Increment and return new count."""
        async with self._lock:
            if key not in self._counts:
                self._counts[key] = {}

            current = self._counts[key].get(window_start, 0)
            self._counts[key][window_start] = current + cost

            return self._counts[key][window_start]

    async def get_token_bucket(self, key: str) -> Tuple[float, float]:
        """Get token bucket state."""
        async with self._lock:
            return self._buckets.get(key, (0.0, 0.0))

    async def set_token_bucket(
        self,
        key: str,
        tokens: float,
        last_update: float,
        ttl_seconds: int,
    ) -> None:
        """Set token bucket state."""
        async with self._lock:
            self._buckets[key] = (tokens, last_update)

    async def cleanup(self) -> int:
        """Clean up expired entries."""
        current_time = time.time()
        cleaned = 0

        async with self._lock:
            # Clean counts
            keys_to_delete = []
            for key, windows in self._counts.items():
                old_windows = [
                    ws for ws in windows
                    if ws < current_time - 3600
                ]
                for ws in old_windows:
                    del windows[ws]
                    cleaned += 1
                if not windows:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del self._counts[key]

        return cleaned


class RedisRateLimitStore(RateLimitStore):
    """Redis-backed rate limit store."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "ratelimit:",
    ):
        """
        Initialize Redis store.

        Args:
            redis_url: Redis connection URL
            key_prefix: Key prefix for rate limit data
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis = None

    async def _get_redis(self):
        """Get Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(self.redis_url)
            except ImportError:
                raise RuntimeError("redis package required for RedisRateLimitStore")
        return self._redis

    async def get_count(self, key: str, window_start: float) -> int:
        """Get request count for a key."""
        redis_client = await self._get_redis()
        redis_key = f"{self.key_prefix}count:{key}:{int(window_start)}"

        count = await redis_client.get(redis_key)
        return int(count) if count else 0

    async def increment(
        self,
        key: str,
        window_start: float,
        ttl_seconds: int,
        cost: int = 1,
    ) -> int:
        """Increment and return new count."""
        redis_client = await self._get_redis()
        redis_key = f"{self.key_prefix}count:{key}:{int(window_start)}"

        pipe = redis_client.pipeline()
        pipe.incrby(redis_key, cost)
        pipe.expire(redis_key, ttl_seconds)
        results = await pipe.execute()

        return results[0]

    async def get_token_bucket(self, key: str) -> Tuple[float, float]:
        """Get token bucket state."""
        redis_client = await self._get_redis()
        redis_key = f"{self.key_prefix}bucket:{key}"

        data = await redis_client.hgetall(redis_key)
        if not data:
            return (0.0, 0.0)

        return (
            float(data.get(b"tokens", 0)),
            float(data.get(b"last_update", 0)),
        )

    async def set_token_bucket(
        self,
        key: str,
        tokens: float,
        last_update: float,
        ttl_seconds: int,
    ) -> None:
        """Set token bucket state."""
        redis_client = await self._get_redis()
        redis_key = f"{self.key_prefix}bucket:{key}"

        pipe = redis_client.pipeline()
        pipe.hset(redis_key, mapping={
            "tokens": str(tokens),
            "last_update": str(last_update),
        })
        pipe.expire(redis_key, ttl_seconds)
        await pipe.execute()


class RateLimiter:
    """
    Sophisticated rate limiter with multiple strategies.

    Supports:
    - Fixed window limiting
    - Sliding window limiting
    - Token bucket algorithm
    - Leaky bucket algorithm
    - Per-endpoint rules
    - Multi-scope limiting
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        store: Optional[RateLimitStore] = None,
    ):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration
            store: Storage backend
        """
        self.config = config or RateLimitConfig()
        self.store = store or InMemoryRateLimitStore()

        # Rules registry
        self._rules: List[RateLimitRule] = []
        self._default_rules_added = False

    def add_rule(self, rule: RateLimitRule) -> None:
        """Add a rate limiting rule."""
        self._rules.append(rule)

    def add_default_rules(self) -> None:
        """Add default rate limiting rules."""
        if self._default_rules_added:
            return

        # Per-minute limit
        self._rules.append(RateLimitRule(
            name="default_per_minute",
            requests=self.config.default_requests_per_minute,
            window_seconds=60,
            scope=RateLimitScope.API_KEY,
            strategy=self.config.strategy,
        ))

        # Per-day limit
        self._rules.append(RateLimitRule(
            name="default_per_day",
            requests=self.config.default_requests_per_day,
            window_seconds=86400,
            scope=RateLimitScope.API_KEY,
            strategy=RateLimitStrategy.FIXED_WINDOW,
        ))

        self._default_rules_added = True

    async def check(
        self,
        identifier: str,
        scope: RateLimitScope = RateLimitScope.API_KEY,
        endpoint: Optional[str] = None,
        method: str = "GET",
        cost: int = 1,
    ) -> RateLimitResult:
        """
        Check if request is allowed.

        Args:
            identifier: Identifier for the scope (API key, user ID, etc.)
            scope: Rate limit scope
            endpoint: Request endpoint
            method: HTTP method
            cost: Request cost

        Returns:
            Rate limit result
        """
        if not self.config.enabled:
            return RateLimitResult(
                allowed=True,
                remaining=999999,
                limit=999999,
                reset_at=datetime.utcnow() + timedelta(hours=1),
            )

        # Add default rules if not yet added
        self.add_default_rules()

        # Find applicable rules
        applicable_rules = [
            rule for rule in self._rules
            if rule.scope == scope and rule.matches_endpoint(endpoint or "", method)
        ]

        if not applicable_rules:
            return RateLimitResult(
                allowed=True,
                remaining=999999,
                limit=999999,
                reset_at=datetime.utcnow() + timedelta(hours=1),
            )

        # Check each rule
        results = []
        for rule in applicable_rules:
            result = await self._check_rule(identifier, rule, cost)
            results.append(result)

            # If any rule denies, deny immediately
            if not result.allowed:
                return result

        # Return the result with lowest remaining
        return min(results, key=lambda r: r.remaining)

    async def _check_rule(
        self,
        identifier: str,
        rule: RateLimitRule,
        cost: int,
    ) -> RateLimitResult:
        """Check a single rule."""
        key = f"{rule.scope.value}:{identifier}:{rule.name}"

        if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._check_fixed_window(key, rule, cost)
        elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._check_sliding_window(key, rule, cost)
        elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._check_token_bucket(key, rule, cost)
        else:
            return await self._check_fixed_window(key, rule, cost)

    async def _check_fixed_window(
        self,
        key: str,
        rule: RateLimitRule,
        cost: int,
    ) -> RateLimitResult:
        """Fixed window rate limiting."""
        current_time = time.time()
        window_start = current_time - (current_time % rule.window_seconds)
        reset_at = datetime.fromtimestamp(window_start + rule.window_seconds)

        # Get current count
        current_count = await self.store.get_count(key, window_start)

        if current_count + cost > rule.requests:
            retry_after = int(reset_at.timestamp() - current_time)
            return RateLimitResult(
                allowed=False,
                remaining=max(0, rule.requests - current_count),
                limit=rule.requests,
                reset_at=reset_at,
                retry_after_seconds=retry_after,
                rule_name=rule.name,
            )

        # Increment count
        new_count = await self.store.increment(
            key,
            window_start,
            rule.window_seconds * 2,
            cost,
        )

        return RateLimitResult(
            allowed=True,
            remaining=max(0, rule.requests - new_count),
            limit=rule.requests,
            reset_at=reset_at,
            rule_name=rule.name,
        )

    async def _check_sliding_window(
        self,
        key: str,
        rule: RateLimitRule,
        cost: int,
    ) -> RateLimitResult:
        """Sliding window rate limiting."""
        current_time = time.time()
        window_size = rule.window_seconds

        # Get counts from current and previous windows
        current_window_start = current_time - (current_time % window_size)
        previous_window_start = current_window_start - window_size

        current_count = await self.store.get_count(key, current_window_start)
        previous_count = await self.store.get_count(key, previous_window_start)

        # Calculate weighted count based on position in current window
        window_position = (current_time - current_window_start) / window_size
        weighted_count = (
            previous_count * (1 - window_position) +
            current_count
        )

        reset_at = datetime.fromtimestamp(current_window_start + window_size)

        if weighted_count + cost > rule.requests:
            retry_after = int(reset_at.timestamp() - current_time)
            return RateLimitResult(
                allowed=False,
                remaining=max(0, int(rule.requests - weighted_count)),
                limit=rule.requests,
                reset_at=reset_at,
                retry_after_seconds=retry_after,
                rule_name=rule.name,
            )

        # Increment current window
        new_count = await self.store.increment(
            key,
            current_window_start,
            window_size * 2,
            cost,
        )

        new_weighted = previous_count * (1 - window_position) + new_count

        return RateLimitResult(
            allowed=True,
            remaining=max(0, int(rule.requests - new_weighted)),
            limit=rule.requests,
            reset_at=reset_at,
            rule_name=rule.name,
        )

    async def _check_token_bucket(
        self,
        key: str,
        rule: RateLimitRule,
        cost: int,
    ) -> RateLimitResult:
        """Token bucket rate limiting."""
        current_time = time.time()

        # Get bucket state
        tokens, last_update = await self.store.get_token_bucket(key)

        # Initialize bucket if new
        if last_update == 0:
            tokens = float(rule.requests)
            last_update = current_time

        # Refill tokens based on time elapsed
        time_elapsed = current_time - last_update
        refill_rate = rule.requests / rule.window_seconds
        tokens = min(
            rule.requests,
            tokens + (time_elapsed * refill_rate),
        )

        # Calculate burst size
        burst_size = rule.burst_size or int(rule.requests * self.config.burst_multiplier)

        reset_at = datetime.fromtimestamp(
            current_time + (rule.requests - tokens) / refill_rate
        )

        if tokens < cost:
            # Not enough tokens
            wait_time = (cost - tokens) / refill_rate
            return RateLimitResult(
                allowed=False,
                remaining=max(0, int(tokens)),
                limit=rule.requests,
                reset_at=reset_at,
                retry_after_seconds=int(wait_time) + 1,
                rule_name=rule.name,
            )

        # Consume tokens
        tokens -= cost

        # Save bucket state
        await self.store.set_token_bucket(
            key,
            tokens,
            current_time,
            rule.window_seconds * 2,
        )

        return RateLimitResult(
            allowed=True,
            remaining=int(tokens),
            limit=rule.requests,
            reset_at=reset_at,
            rule_name=rule.name,
        )


class RateLimitMiddleware:
    """
    FastAPI middleware for rate limiting.

    Usage:
        app = FastAPI()
        limiter = RateLimiter(config)
        app.add_middleware(RateLimitMiddleware, limiter=limiter)
    """

    def __init__(
        self,
        app,
        limiter: RateLimiter,
        get_identifier: Optional[Callable] = None,
    ):
        """
        Initialize middleware.

        Args:
            app: FastAPI application
            limiter: Rate limiter instance
            get_identifier: Function to extract identifier from request
        """
        self.app = app
        self.limiter = limiter
        self.get_identifier = get_identifier or self._default_get_identifier

    def _default_get_identifier(self, request) -> Tuple[str, RateLimitScope]:
        """Default identifier extraction."""
        # Try API key from header
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer ", "")
        if api_key:
            return api_key, RateLimitScope.API_KEY

        # Fall back to IP
        client_ip = request.client.host if request.client else "unknown"
        return client_ip, RateLimitScope.IP

    async def __call__(self, scope, receive, send):
        """Process request."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Build request object for identifier extraction
        from starlette.requests import Request
        request = Request(scope, receive)

        # Check exemptions
        client_ip = request.client.host if request.client else None
        if client_ip and client_ip in self.limiter.config.exempt_ips:
            await self.app(scope, receive, send)
            return

        # Get identifier
        identifier, limit_scope = self.get_identifier(request)

        # Check exemptions for API keys
        if limit_scope == RateLimitScope.API_KEY:
            for prefix in self.limiter.config.exempt_api_keys:
                if identifier.startswith(prefix):
                    await self.app(scope, receive, send)
                    return

        # Check rate limit
        result = await self.limiter.check(
            identifier=identifier,
            scope=limit_scope,
            endpoint=request.url.path,
            method=request.method,
        )

        if not result.allowed:
            # Return 429 response
            from starlette.responses import JSONResponse
            response = JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": {
                        "code": "RATE_4001",
                        "message": "Rate limit exceeded",
                        "details": {
                            "retry_after_seconds": result.retry_after_seconds,
                            "limit": result.limit,
                            "remaining": result.remaining,
                        },
                    },
                },
                headers=result.to_headers() if self.limiter.config.include_headers else {},
            )
            await response(scope, receive, send)
            return

        # Add headers to response
        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                if self.limiter.config.include_headers:
                    for key, value in result.to_headers().items():
                        headers[key.lower().encode()] = value.encode()
                message["headers"] = list(headers.items())
            await send(message)

        await self.app(scope, receive, send_with_headers)


def create_rate_limiter(
    config: Optional[RateLimitConfig] = None,
    use_redis: bool = False,
    redis_url: str = "redis://localhost:6379",
) -> RateLimiter:
    """
    Create a rate limiter instance.

    Args:
        config: Rate limit configuration
        use_redis: Use Redis for storage
        redis_url: Redis connection URL

    Returns:
        Configured rate limiter
    """
    if use_redis:
        store = RedisRateLimitStore(redis_url=redis_url)
    else:
        store = InMemoryRateLimitStore()

    return RateLimiter(config=config, store=store)


__all__ = [
    "RateLimitStrategy",
    "RateLimitScope",
    "RateLimitRule",
    "RateLimitResult",
    "RateLimitConfig",
    "RateLimitStore",
    "InMemoryRateLimitStore",
    "RedisRateLimitStore",
    "RateLimiter",
    "RateLimitMiddleware",
    "create_rate_limiter",
]
