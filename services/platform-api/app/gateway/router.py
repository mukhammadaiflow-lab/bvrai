"""
API Gateway Router

Request routing with:
- Path-based routing
- Header-based routing
- Version-based routing
- Dynamic route configuration
"""

from typing import Optional, Dict, Any, List, Callable, Pattern, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import re
import logging

from fastapi import Request

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """Routing strategies."""
    PATH = "path"
    HEADER = "header"
    QUERY = "query"
    VERSION = "version"
    WEIGHTED = "weighted"
    CANARY = "canary"


class RouteStatus(str, Enum):
    """Route status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    MAINTENANCE = "maintenance"


@dataclass
class RouteConfig:
    """Route configuration."""
    timeout_seconds: float = 30.0
    retries: int = 3
    retry_delay_seconds: float = 1.0
    circuit_breaker_enabled: bool = True
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    cache_enabled: bool = False
    cache_ttl_seconds: int = 60
    auth_required: bool = True
    allowed_methods: Set[str] = field(default_factory=lambda: {"GET", "POST", "PUT", "DELETE", "PATCH"})
    cors_enabled: bool = True
    strip_prefix: bool = False
    add_prefix: Optional[str] = None


@dataclass
class Route:
    """Route definition."""
    id: str
    path_pattern: str
    service_name: str
    methods: Set[str] = field(default_factory=lambda: {"GET"})
    config: RouteConfig = field(default_factory=RouteConfig)
    status: RouteStatus = RouteStatus.ACTIVE
    priority: int = 0
    version: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    _compiled_pattern: Optional[Pattern] = field(default=None, repr=False)

    def __post_init__(self):
        """Compile pattern."""
        self._compile_pattern()

    def _compile_pattern(self) -> None:
        """Compile path pattern to regex."""
        # Convert path pattern to regex
        # Supports: /users/{id}, /api/v{version}/*, /prefix/**
        pattern = self.path_pattern

        # Escape special regex characters except our placeholders
        pattern = re.sub(r'([.+?^${}()|[\]\\])', r'\\\1', pattern)

        # Convert {param} to named groups
        pattern = re.sub(r'\\\{(\w+)\\\}', r'(?P<\1>[^/]+)', pattern)

        # Convert ** to match any path
        pattern = pattern.replace('\\*\\*', '.*')

        # Convert * to match single segment
        pattern = pattern.replace('\\*', '[^/]*')

        # Add anchors
        pattern = f'^{pattern}$'

        self._compiled_pattern = re.compile(pattern)

    def matches(self, path: str, method: str) -> Tuple[bool, Dict[str, str]]:
        """Check if route matches path and method."""
        if method.upper() not in self.methods:
            return False, {}

        if self.status != RouteStatus.ACTIVE:
            return False, {}

        if self._compiled_pattern:
            match = self._compiled_pattern.match(path)
            if match:
                return True, match.groupdict()

        return False, {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "path_pattern": self.path_pattern,
            "service_name": self.service_name,
            "methods": list(self.methods),
            "status": self.status.value,
            "priority": self.priority,
            "version": self.version,
            "tags": list(self.tags),
            "config": {
                "timeout_seconds": self.config.timeout_seconds,
                "retries": self.config.retries,
                "auth_required": self.config.auth_required,
                "rate_limit_enabled": self.config.rate_limit_enabled,
            },
        }


class Router(ABC):
    """Abstract base router."""

    @abstractmethod
    def add_route(self, route: Route) -> None:
        """Add route to router."""
        pass

    @abstractmethod
    def remove_route(self, route_id: str) -> bool:
        """Remove route from router."""
        pass

    @abstractmethod
    def match(self, request: Request) -> Optional[Tuple[Route, Dict[str, str]]]:
        """Match request to route."""
        pass


class PathRouter(Router):
    """
    Path-based router.

    Routes requests based on URL path patterns.
    """

    def __init__(self):
        self._routes: Dict[str, Route] = {}
        self._route_order: List[str] = []  # Sorted by priority

    def add_route(self, route: Route) -> None:
        """Add route to router."""
        self._routes[route.id] = route

        # Maintain sorted order by priority (higher first)
        self._route_order.append(route.id)
        self._route_order.sort(
            key=lambda rid: -self._routes[rid].priority
        )

        logger.info(f"Added route: {route.id} -> {route.service_name}")

    def remove_route(self, route_id: str) -> bool:
        """Remove route from router."""
        if route_id in self._routes:
            del self._routes[route_id]
            self._route_order.remove(route_id)
            logger.info(f"Removed route: {route_id}")
            return True
        return False

    def match(self, request: Request) -> Optional[Tuple[Route, Dict[str, str]]]:
        """Match request to route."""
        path = request.url.path
        method = request.method

        for route_id in self._route_order:
            route = self._routes.get(route_id)
            if not route:
                continue

            matches, params = route.matches(path, method)
            if matches:
                logger.debug(f"Route matched: {route_id} for {method} {path}")
                return route, params

        return None

    def get_route(self, route_id: str) -> Optional[Route]:
        """Get route by ID."""
        return self._routes.get(route_id)

    def list_routes(
        self,
        service_name: Optional[str] = None,
        status: Optional[RouteStatus] = None,
    ) -> List[Route]:
        """List routes with optional filtering."""
        routes = list(self._routes.values())

        if service_name:
            routes = [r for r in routes if r.service_name == service_name]

        if status:
            routes = [r for r in routes if r.status == status]

        return sorted(routes, key=lambda r: -r.priority)

    def update_route_status(
        self,
        route_id: str,
        status: RouteStatus,
    ) -> bool:
        """Update route status."""
        if route_id in self._routes:
            self._routes[route_id].status = status
            self._routes[route_id].updated_at = datetime.utcnow()
            return True
        return False


class VersionedRouter(Router):
    """
    Version-aware router.

    Routes requests based on API version.
    """

    def __init__(
        self,
        version_header: str = "API-Version",
        version_param: str = "version",
        default_version: str = "v1",
    ):
        self.version_header = version_header
        self.version_param = version_param
        self.default_version = default_version
        self._routers: Dict[str, PathRouter] = {}

    def add_route(self, route: Route) -> None:
        """Add route to appropriate version router."""
        version = route.version or self.default_version

        if version not in self._routers:
            self._routers[version] = PathRouter()

        self._routers[version].add_route(route)

    def remove_route(self, route_id: str) -> bool:
        """Remove route from all version routers."""
        removed = False
        for router in self._routers.values():
            if router.remove_route(route_id):
                removed = True
        return removed

    def match(self, request: Request) -> Optional[Tuple[Route, Dict[str, str]]]:
        """Match request to route based on version."""
        # Extract version from header
        version = request.headers.get(self.version_header)

        # Try from path
        if not version:
            path = request.url.path
            version_match = re.match(r'^/v(\d+)/', path)
            if version_match:
                version = f"v{version_match.group(1)}"

        # Try from query param
        if not version:
            version = request.query_params.get(self.version_param)

        # Use default
        version = version or self.default_version

        # Get version router
        router = self._routers.get(version)
        if router:
            result = router.match(request)
            if result:
                return result

        # Fall back to default version
        if version != self.default_version:
            router = self._routers.get(self.default_version)
            if router:
                return router.match(request)

        return None

    def get_versions(self) -> List[str]:
        """Get all available versions."""
        return list(self._routers.keys())


class HeaderRouter(Router):
    """
    Header-based router.

    Routes based on request headers.
    """

    def __init__(
        self,
        header_name: str = "X-Route-To",
        fallback_router: Optional[Router] = None,
    ):
        self.header_name = header_name
        self.fallback_router = fallback_router or PathRouter()
        self._header_routes: Dict[str, Route] = {}

    def add_route(self, route: Route) -> None:
        """Add route."""
        # Check if route has header routing metadata
        header_value = route.metadata.get("route_header_value")
        if header_value:
            self._header_routes[header_value] = route
        else:
            self.fallback_router.add_route(route)

    def remove_route(self, route_id: str) -> bool:
        """Remove route."""
        # Check header routes
        for header_val, route in list(self._header_routes.items()):
            if route.id == route_id:
                del self._header_routes[header_val]
                return True

        return self.fallback_router.remove_route(route_id)

    def match(self, request: Request) -> Optional[Tuple[Route, Dict[str, str]]]:
        """Match based on header."""
        header_value = request.headers.get(self.header_name)

        if header_value and header_value in self._header_routes:
            route = self._header_routes[header_value]
            if route.status == RouteStatus.ACTIVE:
                return route, {}

        return self.fallback_router.match(request)


class CanaryRouter(Router):
    """
    Canary deployment router.

    Routes percentage of traffic to canary service.
    """

    def __init__(
        self,
        canary_percentage: float = 10.0,
        canary_header: str = "X-Canary",
    ):
        self.canary_percentage = canary_percentage
        self.canary_header = canary_header
        self._stable_router = PathRouter()
        self._canary_router = PathRouter()

    def add_route(self, route: Route) -> None:
        """Add route to appropriate router."""
        is_canary = route.metadata.get("canary", False)
        if is_canary:
            self._canary_router.add_route(route)
        else:
            self._stable_router.add_route(route)

    def remove_route(self, route_id: str) -> bool:
        """Remove route."""
        if self._canary_router.remove_route(route_id):
            return True
        return self._stable_router.remove_route(route_id)

    def match(self, request: Request) -> Optional[Tuple[Route, Dict[str, str]]]:
        """Match with canary routing."""
        import random

        # Check explicit canary header
        force_canary = request.headers.get(self.canary_header)
        if force_canary == "true":
            result = self._canary_router.match(request)
            if result:
                return result

        # Random canary selection
        if random.random() * 100 < self.canary_percentage:
            result = self._canary_router.match(request)
            if result:
                return result

        # Fall back to stable
        return self._stable_router.match(request)

    def set_canary_percentage(self, percentage: float) -> None:
        """Update canary percentage."""
        self.canary_percentage = max(0, min(100, percentage))
        logger.info(f"Canary percentage set to {self.canary_percentage}%")


class CompositeRouter(Router):
    """
    Composite router combining multiple routing strategies.
    """

    def __init__(self, routers: Optional[List[Router]] = None):
        self.routers = routers or []

    def add_router(self, router: Router) -> None:
        """Add router to composite."""
        self.routers.append(router)

    def add_route(self, route: Route) -> None:
        """Add route to first router."""
        if self.routers:
            self.routers[0].add_route(route)

    def remove_route(self, route_id: str) -> bool:
        """Remove route from all routers."""
        removed = False
        for router in self.routers:
            if router.remove_route(route_id):
                removed = True
        return removed

    def match(self, request: Request) -> Optional[Tuple[Route, Dict[str, str]]]:
        """Try all routers in order."""
        for router in self.routers:
            result = router.match(request)
            if result:
                return result
        return None


class RouteRegistry:
    """
    Central route registry.

    Manages route configuration and updates.
    """

    def __init__(self, router: Optional[Router] = None):
        self.router = router or PathRouter()
        self._routes: Dict[str, Route] = {}
        self._update_callbacks: List[Callable[[Route, str], None]] = []

    def register(
        self,
        path_pattern: str,
        service_name: str,
        methods: Optional[Set[str]] = None,
        **kwargs,
    ) -> Route:
        """Register a new route."""
        import uuid

        route = Route(
            id=str(uuid.uuid4()),
            path_pattern=path_pattern,
            service_name=service_name,
            methods=methods or {"GET"},
            **kwargs,
        )

        self._routes[route.id] = route
        self.router.add_route(route)

        self._notify_update(route, "added")
        return route

    def unregister(self, route_id: str) -> bool:
        """Unregister a route."""
        if route_id in self._routes:
            route = self._routes.pop(route_id)
            self.router.remove_route(route_id)
            self._notify_update(route, "removed")
            return True
        return False

    def update(
        self,
        route_id: str,
        **updates,
    ) -> Optional[Route]:
        """Update route configuration."""
        route = self._routes.get(route_id)
        if not route:
            return None

        for key, value in updates.items():
            if hasattr(route, key):
                setattr(route, key, value)

        route.updated_at = datetime.utcnow()
        self._notify_update(route, "updated")
        return route

    def get(self, route_id: str) -> Optional[Route]:
        """Get route by ID."""
        return self._routes.get(route_id)

    def list_all(self) -> List[Route]:
        """List all routes."""
        return list(self._routes.values())

    def on_update(self, callback: Callable[[Route, str], None]) -> None:
        """Register update callback."""
        self._update_callbacks.append(callback)

    def _notify_update(self, route: Route, action: str) -> None:
        """Notify callbacks of route update."""
        for callback in self._update_callbacks:
            try:
                callback(route, action)
            except Exception as e:
                logger.error(f"Route update callback error: {e}")

    def match(self, request: Request) -> Optional[Tuple[Route, Dict[str, str]]]:
        """Match request to route."""
        return self.router.match(request)
