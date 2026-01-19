"""
API Gateway
===========

Comprehensive API gateway with request routing, load balancing,
circuit breaking, and service discovery.

Author: Platform Engineering Team
Version: 2.0.0
"""

from bvrai_core.gateway.router import (
    Router,
    Route,
    RouteConfig,
    RouteMatch,
    PathMatcher,
    MethodMatcher,
    HeaderMatcher,
    QueryMatcher,
    LoadBalancer,
    RoundRobinBalancer,
    WeightedBalancer,
    LeastConnectionsBalancer,
    ConsistentHashBalancer,
    ServiceEndpoint,
    ServiceRegistry,
    InMemoryServiceRegistry,
)
from bvrai_core.gateway.circuit import (
    CircuitBreaker,
    CircuitState,
    CircuitConfig,
    CircuitMetrics,
    CircuitBreakerRegistry,
    CircuitOpenError,
)
from bvrai_core.gateway.gateway import (
    APIGateway,
    GatewayConfig,
    GatewayRequest,
    GatewayResponse,
    GatewayMiddleware,
    AuthMiddleware,
    LoggingMiddleware,
    CORSMiddleware,
    RequestTransformer,
    ResponseTransformer,
)

__all__ = [
    # Router
    "Router",
    "Route",
    "RouteConfig",
    "RouteMatch",
    "PathMatcher",
    "MethodMatcher",
    "HeaderMatcher",
    "QueryMatcher",
    "LoadBalancer",
    "RoundRobinBalancer",
    "WeightedBalancer",
    "LeastConnectionsBalancer",
    "ConsistentHashBalancer",
    "ServiceEndpoint",
    "ServiceRegistry",
    "InMemoryServiceRegistry",
    # Circuit
    "CircuitBreaker",
    "CircuitState",
    "CircuitConfig",
    "CircuitMetrics",
    "CircuitBreakerRegistry",
    "CircuitOpenError",
    # Gateway
    "APIGateway",
    "GatewayConfig",
    "GatewayRequest",
    "GatewayResponse",
    "GatewayMiddleware",
    "AuthMiddleware",
    "LoggingMiddleware",
    "CORSMiddleware",
    "RequestTransformer",
    "ResponseTransformer",
]
