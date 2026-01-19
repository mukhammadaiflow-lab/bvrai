"""
Rate Limiting Middleware

FastAPI middleware and decorators for rate limiting:
- Request rate limiting
- Quota enforcement
- Header injection
- Error handling
"""

from typing import Optional, Dict, Any, List, Callable, Union, Awaitable
from functools import wraps
from datetime import datetime
import asyncio
import time
import logging

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from app.ratelimit.algorithms import RateLimiter, RateLimitResult
from app.ratelimit.quota import QuotaManager, QuotaCheckResult

logger = logging.getLogger(__name__)


class RateLimitExceeded(HTTPException):
    """Exception raised when rate limit is exceeded."""

    def __init__(
        self,
        detail: str = "Rate limit exceeded",
        retry_after: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(status_code=429, detail=detail)
        self.retry_after = retry_after
        self.headers = headers or {}

        if retry_after:
            self.headers["Retry-After"] = str(int(retry_after))


class QuotaExceeded(HTTPException):
    """Exception raised when quota is exceeded."""

    def __init__(
        self,
        quota_name: str,
        detail: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        message = detail or f"Quota exceeded: {quota_name}"
        super().__init__(status_code=429, detail=message)
        self.quota_name = quota_name
        self.headers = headers or {}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.

    Applies rate limits based on configurable key extractors.
    """

    def __init__(
        self,
        app: ASGIApp,
        rate_limiter: RateLimiter,
        key_func: Optional[Callable[[Request], str]] = None,
        cost_func: Optional[Callable[[Request], int]] = None,
        skip_paths: Optional[List[str]] = None,
        skip_func: Optional[Callable[[Request], bool]] = None,
        on_limited: Optional[Callable[[Request, RateLimitResult], Awaitable[Response]]] = None,
        include_headers: bool = True,
    ):
        """
        Args:
            app: FastAPI/Starlette application
            rate_limiter: Rate limiter instance
            key_func: Function to extract rate limit key from request
            cost_func: Function to calculate request cost
            skip_paths: Paths to skip rate limiting
            skip_func: Function to determine if request should skip
            on_limited: Custom handler when rate limited
            include_headers: Whether to include rate limit headers
        """
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.key_func = key_func or self._default_key_func
        self.cost_func = cost_func or (lambda r: 1)
        self.skip_paths = set(skip_paths or ["/health", "/metrics", "/ready"])
        self.skip_func = skip_func
        self.on_limited = on_limited
        self.include_headers = include_headers

    def _default_key_func(self, request: Request) -> str:
        """Default key function using client IP."""
        # Try to get real IP from headers (for proxied requests)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the first IP in the chain
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct client IP
        if request.client:
            return request.client.host

        return "unknown"

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process request with rate limiting."""
        # Check skip conditions
        if request.url.path in self.skip_paths:
            return await call_next(request)

        if self.skip_func and self.skip_func(request):
            return await call_next(request)

        # Extract key and cost
        key = self.key_func(request)
        cost = self.cost_func(request)

        # Check rate limit
        result = await self.rate_limiter.check(key, cost)

        if not result.allowed:
            logger.warning(f"Rate limit exceeded for key: {key}")

            if self.on_limited:
                return await self.on_limited(request, result)

            return self._create_rate_limit_response(result)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        if self.include_headers:
            for header_name, header_value in result.to_headers().items():
                response.headers[header_name] = header_value

        return response

    def _create_rate_limit_response(self, result: RateLimitResult) -> Response:
        """Create rate limit exceeded response."""
        content = {
            "error": "rate_limit_exceeded",
            "message": "Too many requests",
            "limit": result.limit,
            "remaining": result.remaining,
            "reset_at": result.reset_at.isoformat(),
        }

        if result.retry_after:
            content["retry_after"] = result.retry_after

        response = JSONResponse(
            status_code=429,
            content=content,
        )

        # Add headers
        for header_name, header_value in result.to_headers().items():
            response.headers[header_name] = header_value

        return response


class QuotaMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for quota enforcement.

    Enforces tenant quotas and tracks usage.
    """

    def __init__(
        self,
        app: ASGIApp,
        quota_manager: QuotaManager,
        tenant_func: Callable[[Request], str],
        quota_name: str = "api_requests",
        cost_func: Optional[Callable[[Request], int]] = None,
        skip_paths: Optional[List[str]] = None,
        include_headers: bool = True,
    ):
        """
        Args:
            app: FastAPI/Starlette application
            quota_manager: Quota manager instance
            tenant_func: Function to extract tenant ID from request
            quota_name: Quota to check
            cost_func: Function to calculate request cost
            skip_paths: Paths to skip quota check
            include_headers: Whether to include quota headers
        """
        super().__init__(app)
        self.quota_manager = quota_manager
        self.tenant_func = tenant_func
        self.quota_name = quota_name
        self.cost_func = cost_func or (lambda r: 1)
        self.skip_paths = set(skip_paths or ["/health", "/metrics", "/ready"])
        self.include_headers = include_headers

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process request with quota enforcement."""
        # Check skip conditions
        if request.url.path in self.skip_paths:
            return await call_next(request)

        # Extract tenant and cost
        try:
            tenant_id = self.tenant_func(request)
        except Exception as e:
            logger.error(f"Failed to extract tenant ID: {e}")
            return await call_next(request)

        cost = self.cost_func(request)

        # Check quota
        result = await self.quota_manager.check_quota(
            tenant_id=tenant_id,
            quota_name=self.quota_name,
            amount=cost,
        )

        if not result.allowed:
            logger.warning(f"Quota exceeded for tenant: {tenant_id}")
            return self._create_quota_response(result)

        # Process request
        response = await call_next(request)

        # Add quota headers
        if self.include_headers:
            response.headers["X-Quota-Limit"] = str(result.usage.limit)
            response.headers["X-Quota-Remaining"] = str(result.usage.remaining)
            response.headers["X-Quota-Reset"] = result.usage.period_end.isoformat()

            if result.warning:
                response.headers["X-Quota-Warning"] = result.warning

        return response

    def _create_quota_response(self, result: QuotaCheckResult) -> Response:
        """Create quota exceeded response."""
        content = {
            "error": "quota_exceeded",
            "message": f"Quota exceeded: {result.quota_name}",
            "quota": result.quota_name,
            "used": result.usage.used,
            "limit": result.usage.limit,
            "reset_at": result.usage.period_end.isoformat(),
        }

        return JSONResponse(
            status_code=429,
            content=content,
            headers={
                "X-Quota-Limit": str(result.usage.limit),
                "X-Quota-Remaining": "0",
                "X-Quota-Reset": result.usage.period_end.isoformat(),
            },
        )


def rate_limit(
    rate_limiter: RateLimiter,
    key_func: Optional[Callable[..., str]] = None,
    cost: int = 1,
    cost_func: Optional[Callable[..., int]] = None,
) -> Callable:
    """
    Decorator for rate limiting individual endpoints.

    Usage:
        @app.get("/api/resource")
        @rate_limit(limiter, key_func=lambda request: request.state.user_id)
        async def get_resource(request: Request):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find request in args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if request is None:
                request = kwargs.get("request")

            if request is None:
                # No request available, skip rate limiting
                logger.warning("rate_limit decorator: no Request found")
                return await func(*args, **kwargs)

            # Extract key
            if key_func:
                if asyncio.iscoroutinefunction(key_func):
                    key = await key_func(request)
                else:
                    key = key_func(request)
            else:
                # Default to IP
                key = request.client.host if request.client else "unknown"

            # Calculate cost
            if cost_func:
                if asyncio.iscoroutinefunction(cost_func):
                    request_cost = await cost_func(request)
                else:
                    request_cost = cost_func(request)
            else:
                request_cost = cost

            # Check rate limit
            result = await rate_limiter.check(key, request_cost)

            if not result.allowed:
                raise RateLimitExceeded(
                    detail="Rate limit exceeded",
                    retry_after=result.retry_after,
                    headers=result.to_headers(),
                )

            # Execute function
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def quota_limit(
    quota_manager: QuotaManager,
    quota_name: str,
    tenant_func: Callable[..., str],
    cost: int = 1,
    cost_func: Optional[Callable[..., int]] = None,
) -> Callable:
    """
    Decorator for quota limiting individual endpoints.

    Usage:
        @app.post("/api/expensive-operation")
        @quota_limit(
            quota_manager,
            quota_name="ai_tokens",
            tenant_func=lambda request: request.state.tenant_id,
            cost_func=lambda request: request.json().get("tokens", 100)
        )
        async def expensive_operation(request: Request):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find request
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if request is None:
                request = kwargs.get("request")

            if request is None:
                logger.warning("quota_limit decorator: no Request found")
                return await func(*args, **kwargs)

            # Extract tenant
            if asyncio.iscoroutinefunction(tenant_func):
                tenant_id = await tenant_func(request)
            else:
                tenant_id = tenant_func(request)

            # Calculate cost
            if cost_func:
                if asyncio.iscoroutinefunction(cost_func):
                    request_cost = await cost_func(request)
                else:
                    request_cost = cost_func(request)
            else:
                request_cost = cost

            # Check quota
            result = await quota_manager.check_quota(
                tenant_id=tenant_id,
                quota_name=quota_name,
                amount=request_cost,
            )

            if not result.allowed:
                raise QuotaExceeded(
                    quota_name=quota_name,
                    detail=f"Quota exceeded: {quota_name}",
                    headers={
                        "X-Quota-Limit": str(result.usage.limit),
                        "X-Quota-Remaining": "0",
                    },
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


class RateLimitManager:
    """
    Centralized rate limit management for FastAPI.

    Usage:
        rate_manager = RateLimitManager()

        # Configure limits
        rate_manager.add_limiter("api", TokenBucket(rate=10, capacity=100))
        rate_manager.add_limiter("auth", FixedWindow(limit=5, window_seconds=60))

        # Apply middleware
        rate_manager.install(app)

        # Or use decorator
        @rate_manager.limit("api")
        async def my_endpoint():
            ...
    """

    def __init__(
        self,
        default_key_func: Optional[Callable[[Request], str]] = None,
    ):
        self._limiters: Dict[str, RateLimiter] = {}
        self._default_key_func = default_key_func or self._extract_ip

    def _extract_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def add_limiter(self, name: str, limiter: RateLimiter) -> None:
        """Add a rate limiter."""
        self._limiters[name] = limiter

    def get_limiter(self, name: str) -> Optional[RateLimiter]:
        """Get a rate limiter by name."""
        return self._limiters.get(name)

    def install(
        self,
        app: ASGIApp,
        limiter_name: str = "api",
        **kwargs,
    ) -> None:
        """Install rate limit middleware on app."""
        limiter = self._limiters.get(limiter_name)
        if not limiter:
            raise ValueError(f"Unknown limiter: {limiter_name}")

        app.add_middleware(
            RateLimitMiddleware,
            rate_limiter=limiter,
            key_func=self._default_key_func,
            **kwargs,
        )

    def limit(
        self,
        limiter_name: str,
        key_func: Optional[Callable] = None,
        cost: int = 1,
    ) -> Callable:
        """Decorator to rate limit an endpoint."""
        limiter = self._limiters.get(limiter_name)
        if not limiter:
            raise ValueError(f"Unknown limiter: {limiter_name}")

        return rate_limit(
            rate_limiter=limiter,
            key_func=key_func or self._default_key_func,
            cost=cost,
        )

    async def check(
        self,
        limiter_name: str,
        key: str,
        cost: int = 1,
    ) -> RateLimitResult:
        """Check rate limit programmatically."""
        limiter = self._limiters.get(limiter_name)
        if not limiter:
            raise ValueError(f"Unknown limiter: {limiter_name}")

        return await limiter.check(key, cost)


class CompositeMiddleware:
    """
    Combines rate limiting and quota enforcement.

    Applies multiple checks in order, failing fast on any denial.
    """

    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        quota_manager: Optional[QuotaManager] = None,
        rate_key_func: Optional[Callable[[Request], str]] = None,
        tenant_func: Optional[Callable[[Request], str]] = None,
        quota_name: str = "api_requests",
    ):
        self.rate_limiter = rate_limiter
        self.quota_manager = quota_manager
        self.rate_key_func = rate_key_func
        self.tenant_func = tenant_func
        self.quota_name = quota_name

    async def __call__(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process request through all checks."""
        # Rate limit check
        if self.rate_limiter:
            key = self.rate_key_func(request) if self.rate_key_func else "global"
            result = await self.rate_limiter.check(key)

            if not result.allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limit_exceeded",
                        "retry_after": result.retry_after,
                    },
                    headers=result.to_headers(),
                )

        # Quota check
        if self.quota_manager and self.tenant_func:
            try:
                tenant_id = self.tenant_func(request)
                quota_result = await self.quota_manager.check_quota(
                    tenant_id=tenant_id,
                    quota_name=self.quota_name,
                )

                if not quota_result.allowed:
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "quota_exceeded",
                            "quota": self.quota_name,
                        },
                    )
            except Exception as e:
                logger.error(f"Quota check error: {e}")

        return await call_next(request)


class SlidingWindowRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limit middleware with precise timing.

    Uses a more memory-efficient sliding window implementation
    for high-traffic scenarios.
    """

    def __init__(
        self,
        app: ASGIApp,
        requests_per_window: int = 100,
        window_seconds: int = 60,
        key_func: Optional[Callable[[Request], str]] = None,
    ):
        super().__init__(app)
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.key_func = key_func or self._default_key

        # In-memory storage (use Redis for production)
        self._requests: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    def _default_key(self, request: Request) -> str:
        """Extract client key from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process request with sliding window rate limit."""
        key = self.key_func(request)
        now = time.time()
        window_start = now - self.window_seconds

        async with self._lock:
            # Get or create request list
            if key not in self._requests:
                self._requests[key] = []

            # Remove old requests
            self._requests[key] = [
                ts for ts in self._requests[key]
                if ts > window_start
            ]

            # Check limit
            current_count = len(self._requests[key])

            if current_count >= self.requests_per_window:
                # Calculate retry after
                oldest = min(self._requests[key])
                retry_after = oldest + self.window_seconds - now

                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limit_exceeded",
                        "message": "Too many requests",
                        "limit": self.requests_per_window,
                        "window_seconds": self.window_seconds,
                        "retry_after": max(0, retry_after),
                    },
                    headers={
                        "X-RateLimit-Limit": str(self.requests_per_window),
                        "X-RateLimit-Remaining": "0",
                        "Retry-After": str(int(max(0, retry_after))),
                    },
                )

            # Record request
            self._requests[key].append(now)
            remaining = self.requests_per_window - current_count - 1

        # Process request
        response = await call_next(request)

        # Add headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response


def create_rate_limit_handler(
    rate_limiter: RateLimiter,
    key_func: Optional[Callable[[Request], str]] = None,
) -> Callable:
    """
    Create a FastAPI dependency for rate limiting.

    Usage:
        limiter = TokenBucket(rate=10, capacity=100)
        rate_limit_check = create_rate_limit_handler(limiter)

        @app.get("/api/resource")
        async def get_resource(
            request: Request,
            _: None = Depends(rate_limit_check)
        ):
            ...
    """

    async def rate_limit_dependency(request: Request) -> None:
        # Extract key
        if key_func:
            key = key_func(request)
        else:
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                key = forwarded.split(",")[0].strip()
            else:
                key = request.client.host if request.client else "unknown"

        # Check limit
        result = await rate_limiter.check(key)

        if not result.allowed:
            raise RateLimitExceeded(
                detail="Rate limit exceeded",
                retry_after=result.retry_after,
                headers=result.to_headers(),
            )

        # Store result in request state for header injection
        request.state.rate_limit_result = result

    return rate_limit_dependency


async def inject_rate_limit_headers(
    request: Request,
    response: Response,
) -> Response:
    """
    Response callback to inject rate limit headers.

    Use with FastAPI's response model.
    """
    if hasattr(request.state, "rate_limit_result"):
        result: RateLimitResult = request.state.rate_limit_result
        for header_name, header_value in result.to_headers().items():
            response.headers[header_name] = header_value

    return response
