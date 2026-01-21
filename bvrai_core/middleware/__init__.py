"""
BVRAI Core Middleware

Provides reusable middleware components for FastAPI services:
- Rate limiting (Redis-backed with local fallback)
- Request logging
- Authentication
- CORS (see bvrai_core.security.cors)
"""

from .rate_limit import (
    RateLimitMiddleware,
    RateLimitDependency,
    rate_limit,
    get_limiter,
    close_limiter,
    extract_client_id,
    add_rate_limit_headers,
    # Pre-configured dependencies
    standard_rate_limit,
    auth_rate_limit,
    expensive_rate_limit,
    upload_rate_limit,
    webhook_rate_limit,
)

__all__ = [
    # Middleware
    "RateLimitMiddleware",
    # Dependencies
    "RateLimitDependency",
    "rate_limit",
    # Utilities
    "get_limiter",
    "close_limiter",
    "extract_client_id",
    "add_rate_limit_headers",
    # Pre-configured
    "standard_rate_limit",
    "auth_rate_limit",
    "expensive_rate_limit",
    "upload_rate_limit",
    "webhook_rate_limit",
]
