"""
API Middleware Module

This module provides middleware components for the REST API,
including rate limiting, logging, CORS, and request tracking.
"""

from .rate_limit import (
    RateLimiter,
    RateLimitConfig,
    RateLimitStrategy,
    InMemoryRateLimitStore,
    RedisRateLimitStore,
    RateLimitMiddleware,
)

from .logging import (
    RequestLogger,
    LogConfig,
    RequestLogMiddleware,
    AccessLogEntry,
)

from .tracking import (
    RequestTracker,
    RequestContext,
    RequestTrackingMiddleware,
)

from .org_context import (
    OrgContextMiddleware,
    set_org_context_for_session,
    async_set_org_context,
    get_org_context_dependency,
)


__all__ = [
    # Rate limiting
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitStrategy",
    "InMemoryRateLimitStore",
    "RedisRateLimitStore",
    "RateLimitMiddleware",
    # Logging
    "RequestLogger",
    "LogConfig",
    "RequestLogMiddleware",
    "AccessLogEntry",
    # Tracking
    "RequestTracker",
    "RequestContext",
    "RequestTrackingMiddleware",
    # Organization Context (RLS)
    "OrgContextMiddleware",
    "set_org_context_for_session",
    "async_set_org_context",
    "get_org_context_dependency",
]
