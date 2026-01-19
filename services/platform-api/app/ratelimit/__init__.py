"""
Advanced Rate Limiting System

Enterprise rate limiting with:
- Token bucket algorithm
- Sliding window counter
- Leaky bucket
- Adaptive rate limiting
- Distributed limiting with Redis
- Per-tenant quotas
- Middleware integration
"""

from app.ratelimit.algorithms import (
    RateLimiter,
    TokenBucket,
    SlidingWindow,
    FixedWindow,
    LeakyBucket,
    CompositeRateLimiter,
    RateLimitResult,
    RateLimiterRegistry,
)

from app.ratelimit.distributed import (
    DistributedRateLimiter,
    RedisRateLimiter,
    RedisTokenBucket,
    RedisSlidingWindow,
    RedisFixedWindow,
    RedisLeakyBucket,
    ClusterAwareRateLimiter,
    ReplicatedRateLimiter,
    FailoverStrategy,
    RedisConfig,
    DistributedMetrics,
)

from app.ratelimit.adaptive import (
    AdaptiveRateLimiter,
    AdaptiveConfig,
    LoadBasedLimiter,
    ConcurrencyLimiter,
    ConcurrencyLimitExceeded,
    AIMLLimiter,
    GradualDegradationLimiter,
    FairShareLimiter,
    MetricsCollector,
    DefaultMetricsCollector,
    SystemMetrics,
    LoadLevel,
    LoadThresholds,
)

from app.ratelimit.quota import (
    QuotaManager,
    Quota,
    QuotaPeriod,
    QuotaUnit,
    QuotaUsage,
    QuotaCheckResult,
    QuotaDefinition,
    TenantQuota,
    QuotaStorage,
    InMemoryQuotaStorage,
    RedisQuotaStorage,
    QuotaReporter,
    OveragePolicy,
)

from app.ratelimit.middleware import (
    RateLimitMiddleware,
    QuotaMiddleware,
    CompositeMiddleware,
    SlidingWindowRateLimitMiddleware,
    RateLimitManager,
    rate_limit,
    quota_limit,
    RateLimitExceeded,
    QuotaExceeded,
    create_rate_limit_handler,
    inject_rate_limit_headers,
)

__all__ = [
    # Algorithms
    "RateLimiter",
    "TokenBucket",
    "SlidingWindow",
    "FixedWindow",
    "LeakyBucket",
    "CompositeRateLimiter",
    "RateLimitResult",
    "RateLimiterRegistry",
    # Distributed
    "DistributedRateLimiter",
    "RedisRateLimiter",
    "RedisTokenBucket",
    "RedisSlidingWindow",
    "RedisFixedWindow",
    "RedisLeakyBucket",
    "ClusterAwareRateLimiter",
    "ReplicatedRateLimiter",
    "FailoverStrategy",
    "RedisConfig",
    "DistributedMetrics",
    # Adaptive
    "AdaptiveRateLimiter",
    "AdaptiveConfig",
    "LoadBasedLimiter",
    "ConcurrencyLimiter",
    "ConcurrencyLimitExceeded",
    "AIMLLimiter",
    "GradualDegradationLimiter",
    "FairShareLimiter",
    "MetricsCollector",
    "DefaultMetricsCollector",
    "SystemMetrics",
    "LoadLevel",
    "LoadThresholds",
    # Quota
    "QuotaManager",
    "Quota",
    "QuotaPeriod",
    "QuotaUnit",
    "QuotaUsage",
    "QuotaCheckResult",
    "QuotaDefinition",
    "TenantQuota",
    "QuotaStorage",
    "InMemoryQuotaStorage",
    "RedisQuotaStorage",
    "QuotaReporter",
    "OveragePolicy",
    # Middleware
    "RateLimitMiddleware",
    "QuotaMiddleware",
    "CompositeMiddleware",
    "SlidingWindowRateLimitMiddleware",
    "RateLimitManager",
    "rate_limit",
    "quota_limit",
    "RateLimitExceeded",
    "QuotaExceeded",
    "create_rate_limit_handler",
    "inject_rate_limit_headers",
]
