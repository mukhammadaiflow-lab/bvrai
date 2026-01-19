"""
Rate Limit Middleware
=====================

HTTP middleware for rate limiting and quota enforcement.

Author: Platform Engineering Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import structlog

from bvrai_core.ratelimit.limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitExceeded,
    RateLimitResult,
    TokenBucketLimiter,
    SlidingWindowLimiter,
    AdaptiveLimiter,
    DistributedLimiter,
)
from bvrai_core.ratelimit.quota import (
    QuotaManager,
    QuotaConfig,
    QuotaExceeded,
    QuotaLimit,
    QuotaPeriod,
    QuotaStore,
    InMemoryQuotaStore,
)

logger = structlog.get_logger(__name__)


@dataclass
class RateLimitResponse:
    """Response for rate-limited requests"""

    status_code: int
    body: Dict[str, Any]
    headers: Dict[str, str]

    @classmethod
    def from_result(cls, result: RateLimitResult) -> "RateLimitResponse":
        """Create response from rate limit result"""
        return cls(
            status_code=429,
            body={
                "error": "rate_limit_exceeded",
                "message": "Too many requests",
                "limit": result.limit,
                "remaining": max(0, result.remaining),
                "reset_at": result.reset_at.isoformat(),
                "retry_after": int(result.retry_after),
            },
            headers=result.headers,
        )

    @classmethod
    def from_quota_exceeded(cls, exc: QuotaExceeded) -> "RateLimitResponse":
        """Create response from quota exceeded"""
        return cls(
            status_code=429,
            body={
                "error": "quota_exceeded",
                "message": f"Quota exceeded for {exc.resource}",
                "resource": exc.resource,
                "limit": exc.limit,
                "used": exc.used,
                "reset_at": exc.reset_at.isoformat(),
            },
            headers={
                "X-Quota-Limit": str(exc.limit),
                "X-Quota-Used": str(exc.used),
                "X-Quota-Reset": str(int(exc.reset_at.timestamp())),
            },
        )


@dataclass
class RateLimitRule:
    """A rate limit rule"""

    name: str
    requests_per_second: float
    burst_size: int = 20
    window_seconds: float = 60.0
    key_func: Optional[Callable[[Dict[str, Any]], str]] = None
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    endpoints: Optional[List[str]] = None
    methods: Optional[List[str]] = None
    user_tiers: Optional[List[str]] = None


@dataclass
class MiddlewareConfig:
    """Configuration for rate limit middleware"""

    enabled: bool = True
    default_rate: float = 100.0
    default_burst: int = 200
    default_window: float = 60.0
    use_distributed: bool = False
    redis_client: Optional[Any] = None
    rules: List[RateLimitRule] = field(default_factory=list)
    excluded_paths: List[str] = field(default_factory=list)
    include_headers: bool = True
    log_rejections: bool = True


class RateLimitMiddleware:
    """
    HTTP middleware for rate limiting.

    Usage:
        middleware = RateLimitMiddleware(config)

        # In request handler
        result = await middleware.process(request_context)
        if not result.allowed:
            return rate_limit_response(result)
    """

    def __init__(self, config: Optional[MiddlewareConfig] = None):
        self.config = config or MiddlewareConfig()
        self._limiters: Dict[str, RateLimiter] = {}
        self._default_limiter = self._create_default_limiter()
        self._logger = structlog.get_logger("rate_limit_middleware")
        self._setup_limiters()

    def _create_default_limiter(self) -> RateLimiter:
        """Create the default limiter"""
        limiter_config = RateLimitConfig(
            requests_per_second=self.config.default_rate,
            burst_size=self.config.default_burst,
            window_seconds=self.config.default_window,
        )

        if self.config.use_distributed and self.config.redis_client:
            return DistributedLimiter(
                config=limiter_config,
                redis_client=self.config.redis_client,
            )
        return TokenBucketLimiter(limiter_config)

    def _setup_limiters(self) -> None:
        """Setup limiters for rules"""
        for rule in self.config.rules:
            limiter_config = RateLimitConfig(
                requests_per_second=rule.requests_per_second,
                burst_size=rule.burst_size,
                window_seconds=rule.window_seconds,
            )

            if self.config.use_distributed and self.config.redis_client:
                limiter = DistributedLimiter(
                    config=limiter_config,
                    redis_client=self.config.redis_client,
                    key_prefix=f"ratelimit:{rule.name}:",
                )
            else:
                limiter = TokenBucketLimiter(limiter_config)

            self._limiters[rule.name] = limiter

    async def process(
        self,
        context: Dict[str, Any],
        cost: int = 1,
    ) -> RateLimitResult:
        """Process a request through rate limiting"""
        if not self.config.enabled:
            return RateLimitResult(
                allowed=True,
                limit=0,
                remaining=0,
                reset_at=datetime.utcnow(),
            )

        # Check excluded paths
        path = context.get("path", "")
        if self._is_excluded(path):
            return RateLimitResult(
                allowed=True,
                limit=0,
                remaining=0,
                reset_at=datetime.utcnow(),
            )

        # Find matching rules
        matching_rules = self._find_matching_rules(context)

        # Check all matching rules
        results = []
        for rule in matching_rules:
            limiter = self._limiters.get(rule.name, self._default_limiter)
            key = self._get_key(rule, context)
            result = await limiter.check(key, cost)
            results.append((rule, result))

            if not result.allowed:
                if self.config.log_rejections:
                    self._logger.warning(
                        "rate_limit_rejected",
                        rule=rule.name,
                        key=key,
                        limit=result.limit,
                        remaining=result.remaining,
                    )
                return result

        # If no rules matched, use default limiter
        if not results:
            key = self._get_default_key(context)
            result = await self._default_limiter.check(key, cost)

            if not result.allowed and self.config.log_rejections:
                self._logger.warning(
                    "rate_limit_rejected",
                    rule="default",
                    key=key,
                    limit=result.limit,
                    remaining=result.remaining,
                )

            return result

        # Return most restrictive result
        return min(results, key=lambda r: r[1].remaining)[1]

    def _is_excluded(self, path: str) -> bool:
        """Check if path is excluded from rate limiting"""
        for excluded in self.config.excluded_paths:
            if path.startswith(excluded):
                return True
        return False

    def _find_matching_rules(self, context: Dict[str, Any]) -> List[RateLimitRule]:
        """Find all rules that match the request"""
        matching = []
        path = context.get("path", "")
        method = context.get("method", "GET")
        user_tier = context.get("user_tier", "default")

        for rule in self.config.rules:
            # Check endpoint match
            if rule.endpoints:
                endpoint_match = False
                for endpoint in rule.endpoints:
                    if path.startswith(endpoint):
                        endpoint_match = True
                        break
                if not endpoint_match:
                    continue

            # Check method match
            if rule.methods and method not in rule.methods:
                continue

            # Check user tier match
            if rule.user_tiers and user_tier not in rule.user_tiers:
                continue

            # Check custom condition
            if rule.condition and not rule.condition(context):
                continue

            matching.append(rule)

        return matching

    def _get_key(self, rule: RateLimitRule, context: Dict[str, Any]) -> str:
        """Get rate limit key for a rule"""
        if rule.key_func:
            return rule.key_func(context)
        return self._get_default_key(context)

    def _get_default_key(self, context: Dict[str, Any]) -> str:
        """Get default rate limit key"""
        # Priority: API key > User ID > Organization ID > IP
        api_key = context.get("api_key")
        if api_key:
            return f"apikey:{api_key}"

        user_id = context.get("user_id")
        if user_id:
            return f"user:{user_id}"

        org_id = context.get("organization_id")
        if org_id:
            return f"org:{org_id}"

        ip = context.get("ip", context.get("client_ip", "unknown"))
        return f"ip:{ip}"

    def add_rule(self, rule: RateLimitRule) -> None:
        """Add a rate limit rule"""
        self.config.rules.append(rule)

        limiter_config = RateLimitConfig(
            requests_per_second=rule.requests_per_second,
            burst_size=rule.burst_size,
            window_seconds=rule.window_seconds,
        )
        self._limiters[rule.name] = TokenBucketLimiter(limiter_config)


class QuotaMiddleware:
    """
    HTTP middleware for quota enforcement.

    Usage:
        middleware = QuotaMiddleware(manager)

        # In request handler
        usage = await middleware.process(context, "api_calls")
        if usage.hard_limit_reached:
            return quota_exceeded_response(usage)
    """

    def __init__(
        self,
        manager: Optional[QuotaManager] = None,
        config: Optional[QuotaConfig] = None,
    ):
        self._manager = manager or QuotaManager(config)
        self._logger = structlog.get_logger("quota_middleware")
        self._resource_mappings: Dict[str, str] = {}

    @property
    def manager(self) -> QuotaManager:
        """Get the quota manager"""
        return self._manager

    async def process(
        self,
        context: Dict[str, Any],
        resource: str,
        amount: int = 1,
    ) -> None:
        """Process a request through quota checking"""
        entity_id = self._get_entity_id(context)
        if not entity_id:
            return

        # Check quota
        await self._manager.check(entity_id, resource, amount)

    async def increment(
        self,
        context: Dict[str, Any],
        resource: str,
        amount: int = 1,
    ) -> None:
        """Increment usage after successful request"""
        entity_id = self._get_entity_id(context)
        if not entity_id:
            return

        await self._manager.increment(entity_id, resource, amount)

    async def get_usage_headers(
        self,
        context: Dict[str, Any],
        resource: str,
    ) -> Dict[str, str]:
        """Get quota usage headers"""
        entity_id = self._get_entity_id(context)
        if not entity_id:
            return {}

        usages = await self._manager.get_usage(entity_id, resource)
        if not usages:
            return {}

        usage = usages[0]
        return {
            "X-Quota-Limit": str(usage.limit),
            "X-Quota-Remaining": str(usage.remaining),
            "X-Quota-Used": str(usage.used),
            "X-Quota-Reset": str(int(usage.period_end.timestamp())),
        }

    def map_resource(self, path_pattern: str, resource: str) -> None:
        """Map a path pattern to a quota resource"""
        self._resource_mappings[path_pattern] = resource

    def get_resource(self, path: str) -> Optional[str]:
        """Get quota resource for a path"""
        for pattern, resource in self._resource_mappings.items():
            if path.startswith(pattern):
                return resource
        return None

    def _get_entity_id(self, context: Dict[str, Any]) -> Optional[str]:
        """Get entity ID from context"""
        # Priority: Organization > User > API Key
        org_id = context.get("organization_id")
        if org_id:
            return f"org:{org_id}"

        user_id = context.get("user_id")
        if user_id:
            return f"user:{user_id}"

        api_key = context.get("api_key_id")
        if api_key:
            return f"apikey:{api_key}"

        return None


# =============================================================================
# COMBINED MIDDLEWARE
# =============================================================================


class CombinedLimitMiddleware:
    """
    Combined rate limiting and quota middleware.

    Applies both rate limiting and quota checks in a single pass.
    """

    def __init__(
        self,
        rate_limiter: Optional[RateLimitMiddleware] = None,
        quota_middleware: Optional[QuotaMiddleware] = None,
    ):
        self._rate_limiter = rate_limiter or RateLimitMiddleware()
        self._quota_middleware = quota_middleware or QuotaMiddleware()
        self._logger = structlog.get_logger("combined_limit_middleware")

    async def process(
        self,
        context: Dict[str, Any],
        quota_resource: Optional[str] = None,
    ) -> Tuple[RateLimitResult, Optional[RateLimitResponse]]:
        """
        Process request through both rate limit and quota checks.

        Returns tuple of (result, error_response).
        error_response is None if request is allowed.
        """
        # Check rate limit first
        result = await self._rate_limiter.process(context)
        if not result.allowed:
            return result, RateLimitResponse.from_result(result)

        # Check quota if resource specified
        if quota_resource:
            try:
                await self._quota_middleware.process(context, quota_resource)
            except QuotaExceeded as e:
                return result, RateLimitResponse.from_quota_exceeded(e)

        return result, None

    async def on_success(
        self,
        context: Dict[str, Any],
        quota_resource: Optional[str] = None,
    ) -> None:
        """Called after successful request to increment quota"""
        if quota_resource:
            await self._quota_middleware.increment(context, quota_resource)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_rate_limiter(
    rate: float = 100.0,
    burst: int = 200,
    use_distributed: bool = False,
    redis_client: Optional[Any] = None,
) -> RateLimitMiddleware:
    """Create a rate limit middleware with defaults"""
    config = MiddlewareConfig(
        default_rate=rate,
        default_burst=burst,
        use_distributed=use_distributed,
        redis_client=redis_client,
    )
    return RateLimitMiddleware(config)


def create_quota_manager(
    default_limits: Optional[List[Tuple[str, int, str]]] = None,
    store: Optional[QuotaStore] = None,
) -> QuotaManager:
    """Create a quota manager with default limits"""
    limits = []
    for item in (default_limits or []):
        resource, limit, period_str = item
        period = QuotaPeriod(period_str) if isinstance(period_str, str) else period_str
        limits.append(
            QuotaLimit(
                resource=resource,
                limit=limit,
                period=period,
            )
        )

    config = QuotaConfig(default_limits=limits)
    return QuotaManager(config, store)


# =============================================================================
# TIER-BASED LIMITS
# =============================================================================


@dataclass
class TierLimits:
    """Rate limits and quotas for a pricing tier"""

    name: str
    rate_limit: float
    burst_size: int
    quotas: Dict[str, Tuple[int, QuotaPeriod]]  # resource -> (limit, period)


# Predefined tiers
TIER_FREE = TierLimits(
    name="free",
    rate_limit=10.0,
    burst_size=20,
    quotas={
        "api_calls": (1000, QuotaPeriod.DAY),
        "voice_minutes": (60, QuotaPeriod.MONTH),
        "agents": (1, QuotaPeriod.LIFETIME),
    },
)

TIER_STARTER = TierLimits(
    name="starter",
    rate_limit=50.0,
    burst_size=100,
    quotas={
        "api_calls": (10000, QuotaPeriod.DAY),
        "voice_minutes": (500, QuotaPeriod.MONTH),
        "agents": (5, QuotaPeriod.LIFETIME),
    },
)

TIER_PROFESSIONAL = TierLimits(
    name="professional",
    rate_limit=200.0,
    burst_size=500,
    quotas={
        "api_calls": (100000, QuotaPeriod.DAY),
        "voice_minutes": (5000, QuotaPeriod.MONTH),
        "agents": (25, QuotaPeriod.LIFETIME),
    },
)

TIER_ENTERPRISE = TierLimits(
    name="enterprise",
    rate_limit=1000.0,
    burst_size=2000,
    quotas={
        "api_calls": (1000000, QuotaPeriod.DAY),
        "voice_minutes": (50000, QuotaPeriod.MONTH),
        "agents": (100, QuotaPeriod.LIFETIME),
    },
)


class TierManager:
    """
    Manages tier-based rate limits and quotas.

    Usage:
        manager = TierManager()
        manager.set_entity_tier("org_123", "professional")
        limits = manager.get_limits("org_123")
    """

    def __init__(self):
        self._tiers: Dict[str, TierLimits] = {
            "free": TIER_FREE,
            "starter": TIER_STARTER,
            "professional": TIER_PROFESSIONAL,
            "enterprise": TIER_ENTERPRISE,
        }
        self._entity_tiers: Dict[str, str] = {}
        self._logger = structlog.get_logger("tier_manager")

    def register_tier(self, tier: TierLimits) -> None:
        """Register a custom tier"""
        self._tiers[tier.name] = tier

    def set_entity_tier(self, entity_id: str, tier_name: str) -> None:
        """Set tier for an entity"""
        if tier_name not in self._tiers:
            raise ValueError(f"Unknown tier: {tier_name}")
        self._entity_tiers[entity_id] = tier_name

    def get_entity_tier(self, entity_id: str) -> str:
        """Get tier for an entity"""
        return self._entity_tiers.get(entity_id, "free")

    def get_limits(self, entity_id: str) -> TierLimits:
        """Get limits for an entity"""
        tier_name = self.get_entity_tier(entity_id)
        return self._tiers[tier_name]

    def get_rate_limit_config(self, entity_id: str) -> RateLimitConfig:
        """Get rate limit config for entity"""
        tier = self.get_limits(entity_id)
        return RateLimitConfig(
            requests_per_second=tier.rate_limit,
            burst_size=tier.burst_size,
        )

    def get_quota_limits(self, entity_id: str) -> List[QuotaLimit]:
        """Get quota limits for entity"""
        tier = self.get_limits(entity_id)
        return [
            QuotaLimit(resource=resource, limit=limit, period=period)
            for resource, (limit, period) in tier.quotas.items()
        ]
