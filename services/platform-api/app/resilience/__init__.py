"""
Enterprise Resilience Patterns

Production-grade fault tolerance with:
- Circuit breaker pattern
- Retry with backoff
- Bulkhead isolation
- Rate limiting
- Timeout management
- Fallback strategies
- Health monitoring
"""

from app.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
    CircuitState,
    CircuitBreakerStats,
    circuit_breaker,
    get_circuit_breaker,
    get_circuit_breaker_async,
)
from app.resilience.retry import (
    RetryPolicy,
    RetryConfig,
    RetryExhausted,
    RetryStats,
    RetryWithCircuitBreaker,
    AdaptiveRetry,
    retry,
    QUICK_RETRY,
    STANDARD_RETRY,
    AGGRESSIVE_RETRY,
    NETWORK_RETRY,
)
from app.resilience.bulkhead import (
    Bulkhead,
    BulkheadFullError,
    BulkheadStats,
    BulkheadRegistry,
    ThreadPoolBulkhead,
    AdaptiveBulkhead,
    bulkhead,
    get_bulkhead,
)
from app.resilience.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitExceeded,
    RateLimitResult,
    TokenBucket,
    SlidingWindowCounter,
    FixedWindowCounter,
    SlidingWindowLog,
    MultiTierRateLimiter,
    IPRateLimiter,
    API_RATE_LIMIT,
    CALL_RATE_LIMIT,
    WEBHOOK_RATE_LIMIT,
    AUTH_RATE_LIMIT,
)
from app.resilience.timeout import (
    TimeoutManager,
    TimeoutConfig,
    TimeoutStats,
    TimeoutExceeded,
    AdaptiveTimeout,
    DeadlineContext,
    CascadingTimeout,
    timeout,
    with_timeout,
    get_current_deadline,
    set_deadline,
    clear_deadline,
)
from app.resilience.fallback import (
    FallbackStrategy,
    DefaultFallback,
    CacheFallback,
    CircuitFallback,
    FallbackChain,
    CallbackFallback,
    GracefulDegradation,
    FeatureFallback,
    RetryWithFallback,
    ShedLoadFallback,
    fallback,
)
from app.resilience.health import (
    HealthChecker,
    HealthStatus,
    HealthCheck,
    HealthCheckResult,
    ComponentHealth,
    DependencyHealth,
    HealthRegistry,
    DatabaseHealthCheck,
    RedisHealthCheck,
    HTTPHealthCheck,
    CallbackHealthCheck,
    CompositeHealthCheck,
    MemoryHealthCheck,
    DiskHealthCheck,
    health_check,
)

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitBreakerRegistry",
    "CircuitState",
    "CircuitBreakerStats",
    "circuit_breaker",
    "get_circuit_breaker",
    "get_circuit_breaker_async",

    # Retry
    "RetryPolicy",
    "RetryConfig",
    "RetryExhausted",
    "RetryStats",
    "RetryWithCircuitBreaker",
    "AdaptiveRetry",
    "retry",
    "QUICK_RETRY",
    "STANDARD_RETRY",
    "AGGRESSIVE_RETRY",
    "NETWORK_RETRY",

    # Bulkhead
    "Bulkhead",
    "BulkheadFullError",
    "BulkheadStats",
    "BulkheadRegistry",
    "ThreadPoolBulkhead",
    "AdaptiveBulkhead",
    "bulkhead",
    "get_bulkhead",

    # Rate Limiter
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitExceeded",
    "RateLimitResult",
    "TokenBucket",
    "SlidingWindowCounter",
    "FixedWindowCounter",
    "SlidingWindowLog",
    "MultiTierRateLimiter",
    "IPRateLimiter",
    "API_RATE_LIMIT",
    "CALL_RATE_LIMIT",
    "WEBHOOK_RATE_LIMIT",
    "AUTH_RATE_LIMIT",

    # Timeout
    "TimeoutManager",
    "TimeoutConfig",
    "TimeoutStats",
    "TimeoutExceeded",
    "AdaptiveTimeout",
    "DeadlineContext",
    "CascadingTimeout",
    "timeout",
    "with_timeout",
    "get_current_deadline",
    "set_deadline",
    "clear_deadline",

    # Fallback
    "FallbackStrategy",
    "DefaultFallback",
    "CacheFallback",
    "CircuitFallback",
    "FallbackChain",
    "CallbackFallback",
    "GracefulDegradation",
    "FeatureFallback",
    "RetryWithFallback",
    "ShedLoadFallback",
    "fallback",

    # Health
    "HealthChecker",
    "HealthStatus",
    "HealthCheck",
    "HealthCheckResult",
    "ComponentHealth",
    "DependencyHealth",
    "HealthRegistry",
    "DatabaseHealthCheck",
    "RedisHealthCheck",
    "HTTPHealthCheck",
    "CallbackHealthCheck",
    "CompositeHealthCheck",
    "MemoryHealthCheck",
    "DiskHealthCheck",
    "health_check",
]
