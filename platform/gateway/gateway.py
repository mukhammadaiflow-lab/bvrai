"""
API Gateway
===========

Main gateway orchestration with middleware, transformers, and request handling.

Author: Platform Engineering Team
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import structlog

from platform.gateway.router import (
    Router,
    Route,
    RouteMatch,
    ServiceRegistry,
    InMemoryServiceRegistry,
    ServiceEndpoint,
    RoundRobinBalancer,
)
from platform.gateway.circuit import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    Bulkhead,
)

logger = structlog.get_logger(__name__)

T = TypeVar("T")


@dataclass
class GatewayRequest:
    """Incoming gateway request"""

    id: str
    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    query: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    client_ip: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for routing"""
        return {
            "id": self.id,
            "method": self.method,
            "path": self.path,
            "headers": self.headers,
            "query": self.query,
            "client_ip": self.client_ip,
            "user_id": self.metadata.get("user_id"),
            "organization_id": self.metadata.get("organization_id"),
        }


@dataclass
class GatewayResponse:
    """Gateway response"""

    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    @classmethod
    def error(
        cls,
        status_code: int,
        message: str,
        error_code: Optional[str] = None,
    ) -> "GatewayResponse":
        """Create error response"""
        body = {
            "error": error_code or "error",
            "message": message,
            "status_code": status_code,
        }
        return cls(
            status_code=status_code,
            headers={"Content-Type": "application/json"},
            body=json.dumps(body).encode(),
        )

    @classmethod
    def not_found(cls, path: str) -> "GatewayResponse":
        """Create 404 response"""
        return cls.error(404, f"Route not found: {path}", "not_found")

    @classmethod
    def bad_gateway(cls, service: str) -> "GatewayResponse":
        """Create 502 response"""
        return cls.error(
            502,
            f"Service unavailable: {service}",
            "bad_gateway",
        )

    @classmethod
    def service_unavailable(
        cls,
        service: str,
        retry_after: float = 30.0,
    ) -> "GatewayResponse":
        """Create 503 response"""
        response = cls.error(
            503,
            f"Service temporarily unavailable: {service}",
            "service_unavailable",
        )
        response.headers["Retry-After"] = str(int(retry_after))
        return response


# =============================================================================
# MIDDLEWARE
# =============================================================================


class GatewayMiddleware(ABC):
    """Base class for gateway middleware"""

    @abstractmethod
    async def process_request(
        self,
        request: GatewayRequest,
    ) -> Tuple[GatewayRequest, Optional[GatewayResponse]]:
        """
        Process incoming request.

        Returns:
            Tuple of (modified_request, optional_response).
            If response is not None, request processing stops.
        """
        pass

    @abstractmethod
    async def process_response(
        self,
        request: GatewayRequest,
        response: GatewayResponse,
    ) -> GatewayResponse:
        """Process outgoing response"""
        pass


class AuthMiddleware(GatewayMiddleware):
    """Authentication middleware"""

    def __init__(
        self,
        validator: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
        excluded_paths: Optional[List[str]] = None,
    ):
        self._validator = validator
        self._excluded_paths = excluded_paths or ["/health", "/metrics"]
        self._logger = structlog.get_logger("auth_middleware")

    async def process_request(
        self,
        request: GatewayRequest,
    ) -> Tuple[GatewayRequest, Optional[GatewayResponse]]:
        """Validate authentication"""
        # Check if path is excluded
        for excluded in self._excluded_paths:
            if request.path.startswith(excluded):
                return request, None

        # Get auth token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return request, GatewayResponse.error(
                401,
                "Missing or invalid authorization header",
                "unauthorized",
            )

        token = auth_header[7:]  # Remove "Bearer " prefix

        # Validate token
        if self._validator:
            try:
                claims = self._validator(token)
                if claims is None:
                    return request, GatewayResponse.error(
                        401,
                        "Invalid token",
                        "unauthorized",
                    )

                # Add claims to request metadata
                request.metadata.update(claims)
            except Exception as e:
                self._logger.warning("auth_validation_error", error=str(e))
                return request, GatewayResponse.error(
                    401,
                    "Token validation failed",
                    "unauthorized",
                )

        return request, None

    async def process_response(
        self,
        request: GatewayRequest,
        response: GatewayResponse,
    ) -> GatewayResponse:
        """Pass through response"""
        return response


class LoggingMiddleware(GatewayMiddleware):
    """Request/response logging middleware"""

    def __init__(
        self,
        log_body: bool = False,
        excluded_paths: Optional[List[str]] = None,
    ):
        self._log_body = log_body
        self._excluded_paths = excluded_paths or ["/health"]
        self._logger = structlog.get_logger("gateway_access")

    async def process_request(
        self,
        request: GatewayRequest,
    ) -> Tuple[GatewayRequest, Optional[GatewayResponse]]:
        """Log incoming request"""
        # Check if path is excluded
        for excluded in self._excluded_paths:
            if request.path.startswith(excluded):
                return request, None

        log_data = {
            "request_id": request.id,
            "method": request.method,
            "path": request.path,
            "client_ip": request.client_ip,
        }

        if self._log_body and request.body:
            log_data["body_size"] = len(request.body)

        self._logger.info("request_received", **log_data)
        return request, None

    async def process_response(
        self,
        request: GatewayRequest,
        response: GatewayResponse,
    ) -> GatewayResponse:
        """Log outgoing response"""
        # Check if path is excluded
        for excluded in self._excluded_paths:
            if request.path.startswith(excluded):
                return response

        self._logger.info(
            "request_completed",
            request_id=request.id,
            method=request.method,
            path=request.path,
            status_code=response.status_code,
            duration_ms=response.duration_ms,
        )
        return response


class CORSMiddleware(GatewayMiddleware):
    """CORS handling middleware"""

    def __init__(
        self,
        allowed_origins: Optional[List[str]] = None,
        allowed_methods: Optional[List[str]] = None,
        allowed_headers: Optional[List[str]] = None,
        expose_headers: Optional[List[str]] = None,
        max_age: int = 86400,
        allow_credentials: bool = True,
    ):
        self._allowed_origins = allowed_origins or ["*"]
        self._allowed_methods = allowed_methods or [
            "GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"
        ]
        self._allowed_headers = allowed_headers or [
            "Content-Type", "Authorization", "X-Request-ID"
        ]
        self._expose_headers = expose_headers or [
            "X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"
        ]
        self._max_age = max_age
        self._allow_credentials = allow_credentials

    async def process_request(
        self,
        request: GatewayRequest,
    ) -> Tuple[GatewayRequest, Optional[GatewayResponse]]:
        """Handle CORS preflight"""
        if request.method == "OPTIONS":
            origin = request.headers.get("Origin", "*")
            response = GatewayResponse(
                status_code=204,
                headers=self._get_cors_headers(origin),
            )
            return request, response

        return request, None

    async def process_response(
        self,
        request: GatewayRequest,
        response: GatewayResponse,
    ) -> GatewayResponse:
        """Add CORS headers to response"""
        origin = request.headers.get("Origin", "*")
        cors_headers = self._get_cors_headers(origin)
        response.headers.update(cors_headers)
        return response

    def _get_cors_headers(self, origin: str) -> Dict[str, str]:
        """Get CORS headers"""
        # Check if origin is allowed
        if "*" in self._allowed_origins:
            allowed_origin = origin if self._allow_credentials else "*"
        elif origin in self._allowed_origins:
            allowed_origin = origin
        else:
            allowed_origin = self._allowed_origins[0] if self._allowed_origins else ""

        headers = {
            "Access-Control-Allow-Origin": allowed_origin,
            "Access-Control-Allow-Methods": ", ".join(self._allowed_methods),
            "Access-Control-Allow-Headers": ", ".join(self._allowed_headers),
            "Access-Control-Expose-Headers": ", ".join(self._expose_headers),
            "Access-Control-Max-Age": str(self._max_age),
        }

        if self._allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"

        return headers


# =============================================================================
# TRANSFORMERS
# =============================================================================


class RequestTransformer(ABC):
    """Base class for request transformers"""

    @abstractmethod
    async def transform(
        self,
        request: GatewayRequest,
        route: Route,
        params: Dict[str, str],
    ) -> GatewayRequest:
        """Transform the request before forwarding"""
        pass


class ResponseTransformer(ABC):
    """Base class for response transformers"""

    @abstractmethod
    async def transform(
        self,
        response: GatewayResponse,
        request: GatewayRequest,
        route: Route,
    ) -> GatewayResponse:
        """Transform the response before returning"""
        pass


class PathTransformer(RequestTransformer):
    """Transform request path based on route config"""

    async def transform(
        self,
        request: GatewayRequest,
        route: Route,
        params: Dict[str, str],
    ) -> GatewayRequest:
        """Apply path transformations"""
        path = request.path

        # Strip prefix
        if route.config.strip_prefix:
            if path.startswith(route.config.strip_prefix):
                path = path[len(route.config.strip_prefix):]
                if not path.startswith("/"):
                    path = "/" + path

        # Add prefix
        if route.config.add_prefix:
            path = route.config.add_prefix + path

        request.path = path
        return request


class HeaderTransformer(RequestTransformer):
    """Add/modify request headers"""

    def __init__(
        self,
        add_headers: Optional[Dict[str, str]] = None,
        remove_headers: Optional[List[str]] = None,
    ):
        self._add_headers = add_headers or {}
        self._remove_headers = [h.lower() for h in (remove_headers or [])]

    async def transform(
        self,
        request: GatewayRequest,
        route: Route,
        params: Dict[str, str],
    ) -> GatewayRequest:
        """Transform headers"""
        # Remove headers
        request.headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in self._remove_headers
        }

        # Add headers
        request.headers.update(self._add_headers)

        # Add request ID if not present
        if "x-request-id" not in {k.lower() for k in request.headers}:
            request.headers["X-Request-ID"] = request.id

        return request


# =============================================================================
# GATEWAY
# =============================================================================


@dataclass
class GatewayConfig:
    """Gateway configuration"""

    name: str = "api_gateway"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    enable_circuit_breaker: bool = True
    enable_bulkhead: bool = True
    bulkhead_max_concurrent: int = 100


class APIGateway:
    """
    Main API Gateway orchestrator.

    Handles request routing, middleware, circuit breaking, and response transformation.

    Usage:
        gateway = APIGateway(config)

        # Add middleware
        gateway.add_middleware(AuthMiddleware())
        gateway.add_middleware(LoggingMiddleware())

        # Add routes
        gateway.router.add_route(Route(...))

        # Process request
        response = await gateway.handle(request)
    """

    def __init__(
        self,
        config: Optional[GatewayConfig] = None,
        registry: Optional[ServiceRegistry] = None,
    ):
        self.config = config or GatewayConfig()
        self._registry = registry or InMemoryServiceRegistry()
        self._router = Router(self._registry)
        self._middleware: List[GatewayMiddleware] = []
        self._request_transformers: List[RequestTransformer] = []
        self._response_transformers: List[ResponseTransformer] = []
        self._circuit_registry = CircuitBreakerRegistry()
        self._bulkheads: Dict[str, Bulkhead] = {}
        self._logger = structlog.get_logger("api_gateway")

        # Add default transformers
        self._request_transformers.append(PathTransformer())
        self._request_transformers.append(HeaderTransformer())

    @property
    def router(self) -> Router:
        """Get the router"""
        return self._router

    @property
    def registry(self) -> ServiceRegistry:
        """Get the service registry"""
        return self._registry

    def add_middleware(self, middleware: GatewayMiddleware) -> None:
        """Add middleware to the pipeline"""
        self._middleware.append(middleware)

    def add_request_transformer(self, transformer: RequestTransformer) -> None:
        """Add request transformer"""
        self._request_transformers.append(transformer)

    def add_response_transformer(self, transformer: ResponseTransformer) -> None:
        """Add response transformer"""
        self._response_transformers.append(transformer)

    async def handle(self, request: GatewayRequest) -> GatewayResponse:
        """
        Handle an incoming request through the gateway.

        Args:
            request: The incoming request

        Returns:
            The response from the backend service
        """
        start_time = time.time()

        try:
            # Process request through middleware
            for mw in self._middleware:
                request, response = await mw.process_request(request)
                if response is not None:
                    response.duration_ms = (time.time() - start_time) * 1000
                    return await self._process_response_middleware(request, response)

            # Match route
            match = await self._router.match(request.to_dict())
            if match is None:
                response = GatewayResponse.not_found(request.path)
                response.duration_ms = (time.time() - start_time) * 1000
                return await self._process_response_middleware(request, response)

            # Check endpoint availability
            if match.endpoint is None:
                response = GatewayResponse.bad_gateway(match.route.service)
                response.duration_ms = (time.time() - start_time) * 1000
                return await self._process_response_middleware(request, response)

            # Transform request
            for transformer in self._request_transformers:
                request = await transformer.transform(request, match.route, match.params)

            # Forward request
            response = await self._forward_request(request, match)

            # Transform response
            for transformer in self._response_transformers:
                response = await transformer.transform(response, request, match.route)

            response.duration_ms = (time.time() - start_time) * 1000

            # Process response through middleware
            return await self._process_response_middleware(request, response)

        except Exception as e:
            self._logger.error(
                "gateway_error",
                request_id=request.id,
                error=str(e),
            )
            response = GatewayResponse.error(500, "Internal gateway error", "internal_error")
            response.duration_ms = (time.time() - start_time) * 1000
            return response

    async def _forward_request(
        self,
        request: GatewayRequest,
        match: RouteMatch,
    ) -> GatewayResponse:
        """Forward request to backend service"""
        service = match.route.service
        endpoint = match.endpoint

        if endpoint is None:
            return GatewayResponse.bad_gateway(service)

        # Get or create circuit breaker
        if self.config.enable_circuit_breaker and match.route.config.circuit_breaker:
            breaker = self._circuit_registry.get(service)

            try:
                return await breaker.call(
                    self._make_request,
                    request,
                    endpoint,
                    match.route.config,
                )
            except CircuitOpenError as e:
                return GatewayResponse.service_unavailable(
                    service,
                    retry_after=e.retry_after,
                )
        else:
            return await self._make_request(request, endpoint, match.route.config)

    async def _make_request(
        self,
        request: GatewayRequest,
        endpoint: ServiceEndpoint,
        config: Any,
    ) -> GatewayResponse:
        """
        Make the actual HTTP request to the backend.

        This is a simplified implementation. In production, you would use
        aiohttp or httpx for actual HTTP calls.
        """
        # Increment connection count
        endpoint.connections += 1

        try:
            # Simulate HTTP call
            # In real implementation:
            # async with aiohttp.ClientSession() as session:
            #     async with session.request(
            #         method=request.method,
            #         url=f"{endpoint.url}{request.path}",
            #         headers=request.headers,
            #         data=request.body,
            #         timeout=aiohttp.ClientTimeout(total=config.timeout_seconds),
            #     ) as resp:
            #         return GatewayResponse(
            #             status_code=resp.status,
            #             headers=dict(resp.headers),
            #             body=await resp.read(),
            #         )

            # Placeholder response
            return GatewayResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                body=json.dumps({
                    "message": "Request forwarded successfully",
                    "service": endpoint.metadata.get("service", "unknown"),
                    "endpoint": endpoint.url,
                    "path": request.path,
                }).encode(),
            )

        finally:
            endpoint.connections -= 1

    async def _process_response_middleware(
        self,
        request: GatewayRequest,
        response: GatewayResponse,
    ) -> GatewayResponse:
        """Process response through middleware in reverse order"""
        for mw in reversed(self._middleware):
            response = await mw.process_response(request, response)
        return response

    def _get_bulkhead(self, service: str) -> Bulkhead:
        """Get or create bulkhead for a service"""
        if service not in self._bulkheads:
            self._bulkheads[service] = Bulkhead(
                name=service,
                max_concurrent=self.config.bulkhead_max_concurrent,
            )
        return self._bulkheads[service]


# =============================================================================
# HEALTH CHECK
# =============================================================================


class HealthChecker:
    """Health checker for backend services"""

    def __init__(
        self,
        registry: ServiceRegistry,
        interval_seconds: int = 30,
    ):
        self._registry = registry
        self._interval = interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._logger = structlog.get_logger("health_checker")

    async def start(self) -> None:
        """Start health checking"""
        self._running = True
        self._task = asyncio.create_task(self._check_loop())
        self._logger.info("health_checker_started")

    async def stop(self) -> None:
        """Stop health checking"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._logger.info("health_checker_stopped")

    async def _check_loop(self) -> None:
        """Main health check loop"""
        while self._running:
            try:
                await self._check_all_services()
            except Exception as e:
                self._logger.error("health_check_error", error=str(e))

            await asyncio.sleep(self._interval)

    async def _check_all_services(self) -> None:
        """Check health of all registered services"""
        # This would iterate through all services and check their health
        # Implementation depends on how services are registered
        pass

    async def check_endpoint(self, endpoint: ServiceEndpoint, path: str = "/health") -> bool:
        """Check health of a single endpoint"""
        # In production, make actual HTTP request
        # For now, simulate health check
        return True


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_gateway(
    name: str = "api_gateway",
    enable_auth: bool = True,
    enable_logging: bool = True,
    enable_cors: bool = True,
) -> APIGateway:
    """Create a gateway with common middleware"""
    gateway = APIGateway(GatewayConfig(name=name))

    if enable_cors:
        gateway.add_middleware(CORSMiddleware())

    if enable_auth:
        gateway.add_middleware(AuthMiddleware())

    if enable_logging:
        gateway.add_middleware(LoggingMiddleware())

    return gateway
