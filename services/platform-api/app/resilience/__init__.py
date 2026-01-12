"""Resilience patterns for fault tolerance."""

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
]
