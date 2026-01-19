"""
Request Router
==============

Request routing with pattern matching, load balancing, and service discovery.

Author: Platform Engineering Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# SERVICE DISCOVERY
# =============================================================================


@dataclass
class ServiceEndpoint:
    """A service endpoint"""

    id: str
    host: str
    port: int
    weight: int = 1
    healthy: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_health_check: Optional[datetime] = None
    connections: int = 0

    @property
    def url(self) -> str:
        """Get endpoint URL"""
        return f"http://{self.host}:{self.port}"

    def mark_healthy(self) -> None:
        """Mark endpoint as healthy"""
        self.healthy = True
        self.last_health_check = datetime.utcnow()

    def mark_unhealthy(self) -> None:
        """Mark endpoint as unhealthy"""
        self.healthy = False
        self.last_health_check = datetime.utcnow()


@dataclass
class ServiceConfig:
    """Service configuration"""

    name: str
    endpoints: List[ServiceEndpoint] = field(default_factory=list)
    health_check_path: str = "/health"
    health_check_interval: int = 30
    timeout_seconds: float = 30.0
    retry_count: int = 3
    circuit_breaker_enabled: bool = True


class ServiceRegistry(ABC):
    """Base class for service registries"""

    @abstractmethod
    async def register(self, service: str, endpoint: ServiceEndpoint) -> None:
        """Register a service endpoint"""
        pass

    @abstractmethod
    async def deregister(self, service: str, endpoint_id: str) -> None:
        """Deregister a service endpoint"""
        pass

    @abstractmethod
    async def get_endpoints(self, service: str) -> List[ServiceEndpoint]:
        """Get all endpoints for a service"""
        pass

    @abstractmethod
    async def get_healthy_endpoints(self, service: str) -> List[ServiceEndpoint]:
        """Get healthy endpoints for a service"""
        pass


class InMemoryServiceRegistry(ServiceRegistry):
    """In-memory service registry"""

    def __init__(self):
        self._services: Dict[str, Dict[str, ServiceEndpoint]] = defaultdict(dict)
        self._configs: Dict[str, ServiceConfig] = {}
        self._lock = threading.RLock()
        self._logger = structlog.get_logger("service_registry")

    async def register(self, service: str, endpoint: ServiceEndpoint) -> None:
        """Register a service endpoint"""
        with self._lock:
            self._services[service][endpoint.id] = endpoint
            self._logger.info(
                "endpoint_registered",
                service=service,
                endpoint_id=endpoint.id,
                url=endpoint.url,
            )

    async def deregister(self, service: str, endpoint_id: str) -> None:
        """Deregister a service endpoint"""
        with self._lock:
            if service in self._services:
                self._services[service].pop(endpoint_id, None)
                self._logger.info(
                    "endpoint_deregistered",
                    service=service,
                    endpoint_id=endpoint_id,
                )

    async def get_endpoints(self, service: str) -> List[ServiceEndpoint]:
        """Get all endpoints for a service"""
        with self._lock:
            return list(self._services.get(service, {}).values())

    async def get_healthy_endpoints(self, service: str) -> List[ServiceEndpoint]:
        """Get healthy endpoints for a service"""
        endpoints = await self.get_endpoints(service)
        return [e for e in endpoints if e.healthy]

    def set_config(self, config: ServiceConfig) -> None:
        """Set service configuration"""
        with self._lock:
            self._configs[config.name] = config

    def get_config(self, service: str) -> Optional[ServiceConfig]:
        """Get service configuration"""
        with self._lock:
            return self._configs.get(service)


# =============================================================================
# LOAD BALANCING
# =============================================================================


class LoadBalancer(ABC):
    """Base class for load balancers"""

    @abstractmethod
    def select(
        self,
        endpoints: List[ServiceEndpoint],
        request_key: Optional[str] = None,
    ) -> Optional[ServiceEndpoint]:
        """Select an endpoint"""
        pass


class RoundRobinBalancer(LoadBalancer):
    """Round-robin load balancer"""

    def __init__(self):
        self._indices: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def select(
        self,
        endpoints: List[ServiceEndpoint],
        request_key: Optional[str] = None,
    ) -> Optional[ServiceEndpoint]:
        """Select next endpoint in rotation"""
        if not endpoints:
            return None

        # Use service name as key for rotation tracking
        service_key = endpoints[0].metadata.get("service", "default")

        with self._lock:
            index = self._indices[service_key]
            self._indices[service_key] = (index + 1) % len(endpoints)

        return endpoints[index % len(endpoints)]


class WeightedBalancer(LoadBalancer):
    """Weighted random load balancer"""

    def __init__(self):
        import random
        self._random = random

    def select(
        self,
        endpoints: List[ServiceEndpoint],
        request_key: Optional[str] = None,
    ) -> Optional[ServiceEndpoint]:
        """Select endpoint based on weights"""
        if not endpoints:
            return None

        total_weight = sum(e.weight for e in endpoints)
        if total_weight == 0:
            return self._random.choice(endpoints)

        rand = self._random.randint(1, total_weight)
        cumulative = 0

        for endpoint in endpoints:
            cumulative += endpoint.weight
            if rand <= cumulative:
                return endpoint

        return endpoints[-1]


class LeastConnectionsBalancer(LoadBalancer):
    """Least connections load balancer"""

    def select(
        self,
        endpoints: List[ServiceEndpoint],
        request_key: Optional[str] = None,
    ) -> Optional[ServiceEndpoint]:
        """Select endpoint with fewest connections"""
        if not endpoints:
            return None

        return min(endpoints, key=lambda e: e.connections)


class ConsistentHashBalancer(LoadBalancer):
    """Consistent hash load balancer for session affinity"""

    def __init__(self, replicas: int = 100):
        self._replicas = replicas
        self._ring: Dict[int, ServiceEndpoint] = {}
        self._sorted_keys: List[int] = []

    def select(
        self,
        endpoints: List[ServiceEndpoint],
        request_key: Optional[str] = None,
    ) -> Optional[ServiceEndpoint]:
        """Select endpoint using consistent hashing"""
        if not endpoints:
            return None

        if not request_key:
            # Fall back to random selection
            import random
            return random.choice(endpoints)

        # Rebuild ring if needed
        self._build_ring(endpoints)

        # Hash the request key
        key_hash = self._hash(request_key)

        # Find the first endpoint with hash >= key_hash
        for ring_hash in self._sorted_keys:
            if ring_hash >= key_hash:
                return self._ring[ring_hash]

        # Wrap around to first endpoint
        if self._sorted_keys:
            return self._ring[self._sorted_keys[0]]

        return endpoints[0]

    def _build_ring(self, endpoints: List[ServiceEndpoint]) -> None:
        """Build the hash ring"""
        self._ring.clear()
        self._sorted_keys.clear()

        for endpoint in endpoints:
            for i in range(self._replicas):
                key = f"{endpoint.id}:{i}"
                hash_val = self._hash(key)
                self._ring[hash_val] = endpoint
                self._sorted_keys.append(hash_val)

        self._sorted_keys.sort()

    def _hash(self, key: str) -> int:
        """Hash a key to an integer"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)


# =============================================================================
# ROUTE MATCHING
# =============================================================================


class RouteMatcher(ABC):
    """Base class for route matchers"""

    @abstractmethod
    def match(self, request: Dict[str, Any]) -> bool:
        """Check if request matches"""
        pass


class PathMatcher(RouteMatcher):
    """Match requests by path pattern"""

    def __init__(self, pattern: str):
        self.pattern = pattern
        self._regex = self._compile_pattern(pattern)
        self._param_names: List[str] = []

        # Extract parameter names
        for match in re.finditer(r"\{(\w+)\}", pattern):
            self._param_names.append(match.group(1))

    def match(self, request: Dict[str, Any]) -> bool:
        """Check if path matches"""
        path = request.get("path", "")
        return self._regex.match(path) is not None

    def extract_params(self, path: str) -> Dict[str, str]:
        """Extract path parameters"""
        match = self._regex.match(path)
        if not match:
            return {}
        return dict(zip(self._param_names, match.groups()))

    def _compile_pattern(self, pattern: str) -> Pattern:
        """Compile pattern to regex"""
        # Convert {param} to named capture groups
        regex = re.sub(r"\{(\w+)\}", r"([^/]+)", pattern)
        # Add anchors
        regex = f"^{regex}$"
        return re.compile(regex)


class MethodMatcher(RouteMatcher):
    """Match requests by HTTP method"""

    def __init__(self, methods: List[str]):
        self.methods = [m.upper() for m in methods]

    def match(self, request: Dict[str, Any]) -> bool:
        """Check if method matches"""
        method = request.get("method", "GET").upper()
        return method in self.methods


class HeaderMatcher(RouteMatcher):
    """Match requests by headers"""

    def __init__(self, headers: Dict[str, str]):
        self.headers = {k.lower(): v for k, v in headers.items()}

    def match(self, request: Dict[str, Any]) -> bool:
        """Check if headers match"""
        req_headers = {
            k.lower(): v for k, v in request.get("headers", {}).items()
        }

        for key, value in self.headers.items():
            if key not in req_headers:
                return False
            if value != "*" and req_headers[key] != value:
                return False

        return True


class QueryMatcher(RouteMatcher):
    """Match requests by query parameters"""

    def __init__(self, params: Dict[str, str]):
        self.params = params

    def match(self, request: Dict[str, Any]) -> bool:
        """Check if query params match"""
        req_params = request.get("query", {})

        for key, value in self.params.items():
            if key not in req_params:
                return False
            if value != "*" and req_params[key] != value:
                return False

        return True


# =============================================================================
# ROUTING
# =============================================================================


@dataclass
class RouteConfig:
    """Route configuration"""

    timeout_seconds: float = 30.0
    retry_count: int = 3
    retry_delay_seconds: float = 1.0
    strip_prefix: Optional[str] = None
    add_prefix: Optional[str] = None
    preserve_host: bool = False
    circuit_breaker: bool = True
    rate_limit: Optional[str] = None
    auth_required: bool = True


@dataclass
class Route:
    """A route definition"""

    id: str
    service: str
    matchers: List[RouteMatcher] = field(default_factory=list)
    config: RouteConfig = field(default_factory=RouteConfig)
    priority: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, request: Dict[str, Any]) -> bool:
        """Check if route matches request"""
        if not self.enabled:
            return False
        return all(m.match(request) for m in self.matchers)


@dataclass
class RouteMatch:
    """Result of route matching"""

    route: Route
    params: Dict[str, str] = field(default_factory=dict)
    endpoint: Optional[ServiceEndpoint] = None


class Router:
    """
    Request router with pattern matching and load balancing.

    Usage:
        router = Router(registry)

        router.add_route(Route(
            id="users_api",
            service="users",
            matchers=[
                PathMatcher("/api/v1/users/{user_id}"),
                MethodMatcher(["GET", "PUT", "DELETE"]),
            ],
        ))

        match = await router.match(request)
        if match:
            response = await router.forward(request, match)
    """

    def __init__(
        self,
        registry: Optional[ServiceRegistry] = None,
        balancer: Optional[LoadBalancer] = None,
    ):
        self._registry = registry or InMemoryServiceRegistry()
        self._balancer = balancer or RoundRobinBalancer()
        self._routes: List[Route] = []
        self._logger = structlog.get_logger("router")

    @property
    def registry(self) -> ServiceRegistry:
        """Get the service registry"""
        return self._registry

    def add_route(self, route: Route) -> None:
        """Add a route"""
        self._routes.append(route)
        # Sort by priority (higher first)
        self._routes.sort(key=lambda r: r.priority, reverse=True)
        self._logger.info(
            "route_added",
            route_id=route.id,
            service=route.service,
        )

    def remove_route(self, route_id: str) -> bool:
        """Remove a route"""
        for i, route in enumerate(self._routes):
            if route.id == route_id:
                self._routes.pop(i)
                self._logger.info("route_removed", route_id=route_id)
                return True
        return False

    def get_route(self, route_id: str) -> Optional[Route]:
        """Get a route by ID"""
        for route in self._routes:
            if route.id == route_id:
                return route
        return None

    async def match(self, request: Dict[str, Any]) -> Optional[RouteMatch]:
        """Match a request to a route"""
        for route in self._routes:
            if route.matches(request):
                # Extract path params if path matcher exists
                params = {}
                for matcher in route.matchers:
                    if isinstance(matcher, PathMatcher):
                        params = matcher.extract_params(request.get("path", ""))
                        break

                # Select endpoint
                endpoints = await self._registry.get_healthy_endpoints(route.service)
                endpoint = self._balancer.select(
                    endpoints,
                    request_key=self._get_affinity_key(request),
                )

                return RouteMatch(
                    route=route,
                    params=params,
                    endpoint=endpoint,
                )

        return None

    def _get_affinity_key(self, request: Dict[str, Any]) -> Optional[str]:
        """Get affinity key for consistent hashing"""
        # Use user ID or session ID for affinity
        user_id = request.get("user_id")
        if user_id:
            return f"user:{user_id}"

        session_id = request.get("headers", {}).get("x-session-id")
        if session_id:
            return f"session:{session_id}"

        return None

    def list_routes(self) -> List[Route]:
        """List all routes"""
        return list(self._routes)


# =============================================================================
# ROUTE BUILDER
# =============================================================================


class RouteBuilder:
    """
    Fluent builder for routes.

    Usage:
        route = (RouteBuilder("users_api", "users")
            .path("/api/v1/users/{user_id}")
            .methods(["GET", "PUT", "DELETE"])
            .header("Content-Type", "application/json")
            .timeout(30)
            .build())
    """

    def __init__(self, route_id: str, service: str):
        self._route_id = route_id
        self._service = service
        self._matchers: List[RouteMatcher] = []
        self._config = RouteConfig()
        self._priority = 0
        self._metadata: Dict[str, Any] = {}

    def path(self, pattern: str) -> "RouteBuilder":
        """Add path matcher"""
        self._matchers.append(PathMatcher(pattern))
        return self

    def methods(self, methods: List[str]) -> "RouteBuilder":
        """Add method matcher"""
        self._matchers.append(MethodMatcher(methods))
        return self

    def header(self, key: str, value: str) -> "RouteBuilder":
        """Add header matcher"""
        # Find existing header matcher or create new
        for matcher in self._matchers:
            if isinstance(matcher, HeaderMatcher):
                matcher.headers[key.lower()] = value
                return self
        self._matchers.append(HeaderMatcher({key: value}))
        return self

    def query(self, key: str, value: str) -> "RouteBuilder":
        """Add query param matcher"""
        for matcher in self._matchers:
            if isinstance(matcher, QueryMatcher):
                matcher.params[key] = value
                return self
        self._matchers.append(QueryMatcher({key: value}))
        return self

    def timeout(self, seconds: float) -> "RouteBuilder":
        """Set timeout"""
        self._config.timeout_seconds = seconds
        return self

    def retries(self, count: int, delay: float = 1.0) -> "RouteBuilder":
        """Set retry configuration"""
        self._config.retry_count = count
        self._config.retry_delay_seconds = delay
        return self

    def strip_prefix(self, prefix: str) -> "RouteBuilder":
        """Strip prefix from path"""
        self._config.strip_prefix = prefix
        return self

    def add_prefix(self, prefix: str) -> "RouteBuilder":
        """Add prefix to path"""
        self._config.add_prefix = prefix
        return self

    def no_auth(self) -> "RouteBuilder":
        """Disable authentication requirement"""
        self._config.auth_required = False
        return self

    def priority(self, priority: int) -> "RouteBuilder":
        """Set route priority"""
        self._priority = priority
        return self

    def metadata(self, key: str, value: Any) -> "RouteBuilder":
        """Add metadata"""
        self._metadata[key] = value
        return self

    def build(self) -> Route:
        """Build the route"""
        return Route(
            id=self._route_id,
            service=self._service,
            matchers=self._matchers,
            config=self._config,
            priority=self._priority,
            metadata=self._metadata,
        )
