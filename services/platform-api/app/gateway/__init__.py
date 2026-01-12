"""
API Gateway System

Enterprise API gateway with:
- Load balancing
- Service discovery
- Request routing
- API versioning
- Rate limiting integration
- Request/response transformation
- Authentication and authorization
"""

from app.gateway.router import (
    Route,
    RouteConfig,
    PathRouter,
    VersionedRouter,
    HeaderRouter,
    CanaryRouter,
    CompositeRouter,
    RouteRegistry,
)

from app.gateway.loadbalancer import (
    LoadBalancer,
    ServiceInstance,
    InstanceHealth,
    RoundRobinBalancer,
    WeightedBalancer,
    LeastConnectionsBalancer,
    ConsistentHashBalancer,
    LeastResponseTimeBalancer,
    AdaptiveBalancer,
)

from app.gateway.discovery import (
    ServiceRegistry,
    ServiceDefinition,
    ServiceStatus,
    ServiceHealth,
    StaticServiceDiscovery,
    ConsulServiceDiscovery,
    HealthChecker,
    ServiceDiscoveryManager,
)

from app.gateway.proxy import (
    ProxyHandler,
    ProxyConfig,
    ProxyRequest,
    ProxyResponse,
    RequestTransformer,
    ResponseTransformer,
    HeaderTransformer,
    PathTransformer,
    BodyTransformer,
    StreamingProxyHandler,
    ProxyManager,
)

from app.gateway.middleware import (
    GatewayMiddleware,
    GatewayRequest,
    GatewayResponse,
    MiddlewareChain,
    AuthenticationMiddleware,
    RateLimitingMiddleware,
    RequestLoggingMiddleware,
    CORSMiddleware,
    SecurityMiddleware,
    CompressionMiddleware,
    RequestIdMiddleware,
    TimeoutMiddleware,
    IPFilterMiddleware,
    CircuitBreakerMiddleware,
    MiddlewareFactory,
)

__all__ = [
    # Router
    "Route",
    "RouteConfig",
    "PathRouter",
    "VersionedRouter",
    "HeaderRouter",
    "CanaryRouter",
    "CompositeRouter",
    "RouteRegistry",
    # Load balancer
    "LoadBalancer",
    "ServiceInstance",
    "InstanceHealth",
    "RoundRobinBalancer",
    "WeightedBalancer",
    "LeastConnectionsBalancer",
    "ConsistentHashBalancer",
    "LeastResponseTimeBalancer",
    "AdaptiveBalancer",
    # Service discovery
    "ServiceRegistry",
    "ServiceDefinition",
    "ServiceStatus",
    "ServiceHealth",
    "StaticServiceDiscovery",
    "ConsulServiceDiscovery",
    "HealthChecker",
    "ServiceDiscoveryManager",
    # Proxy
    "ProxyHandler",
    "ProxyConfig",
    "ProxyRequest",
    "ProxyResponse",
    "RequestTransformer",
    "ResponseTransformer",
    "HeaderTransformer",
    "PathTransformer",
    "BodyTransformer",
    "StreamingProxyHandler",
    "ProxyManager",
    # Middleware
    "GatewayMiddleware",
    "GatewayRequest",
    "GatewayResponse",
    "MiddlewareChain",
    "AuthenticationMiddleware",
    "RateLimitingMiddleware",
    "RequestLoggingMiddleware",
    "CORSMiddleware",
    "SecurityMiddleware",
    "CompressionMiddleware",
    "RequestIdMiddleware",
    "TimeoutMiddleware",
    "IPFilterMiddleware",
    "CircuitBreakerMiddleware",
    "MiddlewareFactory",
]
