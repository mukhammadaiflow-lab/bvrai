"""
Rate Limiting Middleware for FastAPI

Provides both middleware and dependency-based rate limiting:
- Global rate limits via middleware
- Route-specific limits via dependencies
- User/API key based identification
- Redis-backed distributed limiting with local fallback
"""

import logging
import os
from typing import Callable, Optional
from functools import wraps

from fastapi import HTTPException, Request, Response, Depends
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from bvrai_core.ratelimit.distributed import (
    DistributedRateLimiter,
    RateLimitAlgorithm,
    RateLimitResult,
)

logger = logging.getLogger(__name__)


# Global limiter instance (lazy initialization)
_limiter: Optional[DistributedRateLimiter] = None


def get_limiter() -> DistributedRateLimiter:
    """Get or create the global rate limiter instance."""
    global _limiter
    if _limiter is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _limiter = DistributedRateLimiter(
            redis_url=redis_url,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            key_prefix="bvrai:ratelimit:",
        )
    return _limiter


def extract_client_id(request: Request) -> str:
    """
    Extract client identifier for rate limiting.

    Priority:
    1. API Key (X-API-Key header)
    2. User ID from JWT
    3. IP address (fallback)
    """
    # Try API key first
    api_key = request.headers.get("X-API-Key")
    if api_key:
        # Use first 16 chars for privacy
        return f"api:{api_key[:16]}"

    # Try user from auth state
    user = getattr(request.state, "user", None)
    if user and hasattr(user, "id"):
        return f"user:{user.id}"

    # Try organization from auth
    org_id = getattr(request.state, "organization_id", None)
    if org_id:
        return f"org:{org_id}"

    # Fallback to IP
    client_ip = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    return f"ip:{client_ip}"


def add_rate_limit_headers(response: Response, result: RateLimitResult) -> None:
    """Add standard rate limit headers to response."""
    response.headers["X-RateLimit-Limit"] = str(result.limit)
    response.headers["X-RateLimit-Remaining"] = str(result.remaining)
    response.headers["X-RateLimit-Reset"] = str(int(result.reset_at))

    if not result.allowed:
        response.headers["Retry-After"] = str(int(result.retry_after))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Global rate limiting middleware.

    Applies default rate limits to all requests.
    Use route-specific dependencies for finer control.
    """

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        exclude_paths: Optional[list[str]] = None,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.exclude_paths = exclude_paths or [
            "/health",
            "/healthz",
            "/ready",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        # Skip excluded paths
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)

        limiter = get_limiter()
        client_id = extract_client_id(request)

        # Check per-minute limit
        minute_key = f"{client_id}:minute"
        minute_result = await limiter.check_rate_limit(
            key=minute_key,
            limit=self.requests_per_minute,
            window_seconds=60,
        )

        if not minute_result.allowed:
            response = JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please slow down.",
                    "retry_after": minute_result.retry_after,
                },
            )
            add_rate_limit_headers(response, minute_result)
            return response

        # Check per-hour limit (for sustained abuse)
        hour_key = f"{client_id}:hour"
        hour_result = await limiter.check_rate_limit(
            key=hour_key,
            limit=self.requests_per_hour,
            window_seconds=3600,
        )

        if not hour_result.allowed:
            response = JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Hourly rate limit exceeded. Please try again later.",
                    "retry_after": hour_result.retry_after,
                },
            )
            add_rate_limit_headers(response, hour_result)
            return response

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        add_rate_limit_headers(response, minute_result)

        return response


class RateLimitDependency:
    """
    Route-specific rate limit dependency.

    Usage:
        @router.post("/expensive-operation")
        async def expensive(
            rate_limit: None = Depends(RateLimitDependency(limit=10, window=60))
        ):
            ...
    """

    def __init__(
        self,
        limit: int,
        window_seconds: int = 60,
        key_suffix: Optional[str] = None,
        error_message: str = "Rate limit exceeded",
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self.key_suffix = key_suffix
        self.error_message = error_message

    async def __call__(self, request: Request) -> None:
        """Check rate limit for this route."""
        limiter = get_limiter()
        client_id = extract_client_id(request)

        # Build key with route path for route-specific limits
        suffix = self.key_suffix or request.url.path.replace("/", ":")
        key = f"{client_id}:{suffix}"

        result = await limiter.check_rate_limit(
            key=key,
            limit=self.limit,
            window_seconds=self.window_seconds,
        )

        if not result.allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": self.error_message,
                    "retry_after": result.retry_after,
                },
                headers={
                    "X-RateLimit-Limit": str(result.limit),
                    "X-RateLimit-Remaining": str(result.remaining),
                    "X-RateLimit-Reset": str(int(result.reset_at)),
                    "Retry-After": str(int(result.retry_after)),
                },
            )


def rate_limit(
    limit: int,
    window_seconds: int = 60,
    key_suffix: Optional[str] = None,
) -> Callable:
    """
    Decorator for rate limiting individual routes.

    Usage:
        @router.post("/webhook")
        @rate_limit(limit=100, window_seconds=60)
        async def webhook():
            ...
    """
    dependency = RateLimitDependency(
        limit=limit,
        window_seconds=window_seconds,
        key_suffix=key_suffix,
    )

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, request: Request, **kwargs):
            await dependency(request)
            return await func(*args, request=request, **kwargs)
        return wrapper
    return decorator


# Pre-configured rate limit dependencies for common use cases

# Standard API endpoints: 60/min
standard_rate_limit = RateLimitDependency(limit=60, window_seconds=60)

# Auth endpoints: 10/min (prevent brute force)
auth_rate_limit = RateLimitDependency(
    limit=10,
    window_seconds=60,
    key_suffix="auth",
    error_message="Too many authentication attempts",
)

# Expensive operations: 5/min (LLM calls, voice synthesis)
expensive_rate_limit = RateLimitDependency(
    limit=5,
    window_seconds=60,
    key_suffix="expensive",
    error_message="Rate limit for this operation exceeded",
)

# File uploads: 20/hour
upload_rate_limit = RateLimitDependency(
    limit=20,
    window_seconds=3600,
    key_suffix="upload",
    error_message="Upload rate limit exceeded",
)

# Webhook calls: 1000/min (high volume)
webhook_rate_limit = RateLimitDependency(
    limit=1000,
    window_seconds=60,
    key_suffix="webhook",
)


async def close_limiter() -> None:
    """Close the rate limiter connection on shutdown."""
    global _limiter
    if _limiter:
        await _limiter.close()
        _limiter = None
