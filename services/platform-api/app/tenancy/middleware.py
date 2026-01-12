"""
Tenant Middleware and Resolution

FastAPI middleware for multi-tenant handling:
- Tenant resolution strategies
- Request tenant context injection
- Cross-tenant protection
"""

from typing import Optional, Dict, Any, List, Callable, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re
import logging
import jwt

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.tenancy.context import (
    TenantContext,
    set_current_tenant,
    clear_tenant_context,
)
from app.tenancy.tenant import Tenant, TenantManager, TenantStatus

logger = logging.getLogger(__name__)


class TenantResolver(ABC):
    """Abstract base for tenant resolution."""

    @abstractmethod
    async def resolve(self, request: Request) -> Optional[str]:
        """Resolve tenant ID from request."""
        pass


class HeaderTenantResolver(TenantResolver):
    """
    Resolve tenant from HTTP header.

    Common pattern for API-first applications.
    """

    def __init__(
        self,
        header_name: str = "X-Tenant-ID",
        alternative_headers: Optional[List[str]] = None,
    ):
        self.header_name = header_name
        self.alternative_headers = alternative_headers or ["X-Tenant", "Tenant-ID"]

    async def resolve(self, request: Request) -> Optional[str]:
        """Get tenant ID from header."""
        # Check primary header
        tenant_id = request.headers.get(self.header_name)
        if tenant_id:
            return tenant_id

        # Check alternative headers
        for header in self.alternative_headers:
            tenant_id = request.headers.get(header)
            if tenant_id:
                return tenant_id

        return None


class SubdomainTenantResolver(TenantResolver):
    """
    Resolve tenant from subdomain.

    Pattern: tenant-slug.domain.com
    """

    def __init__(
        self,
        base_domain: str = "voiceai.com",
        excluded_subdomains: Optional[Set[str]] = None,
    ):
        self.base_domain = base_domain
        self.excluded_subdomains = excluded_subdomains or {
            "www", "api", "app", "admin", "dashboard",
            "docs", "status", "blog", "support",
        }

    async def resolve(self, request: Request) -> Optional[str]:
        """Extract tenant slug from subdomain."""
        host = request.headers.get("host", "")

        # Remove port if present
        host = host.split(":")[0]

        # Check if it's a subdomain of base domain
        if not host.endswith(self.base_domain):
            return None

        # Extract subdomain
        subdomain = host[:-len(self.base_domain)].rstrip(".")

        if not subdomain or subdomain in self.excluded_subdomains:
            return None

        return subdomain


class PathTenantResolver(TenantResolver):
    """
    Resolve tenant from URL path.

    Pattern: /tenant/{tenant_id}/resource
    """

    def __init__(
        self,
        path_pattern: str = r"^/t/([^/]+)",
        parameter_name: str = "tenant_id",
    ):
        self.path_pattern = re.compile(path_pattern)
        self.parameter_name = parameter_name

    async def resolve(self, request: Request) -> Optional[str]:
        """Extract tenant ID from URL path."""
        # Check route parameters first
        if hasattr(request, "path_params"):
            tenant_id = request.path_params.get(self.parameter_name)
            if tenant_id:
                return tenant_id

        # Fall back to pattern matching
        match = self.path_pattern.match(request.url.path)
        if match:
            return match.group(1)

        return None


class JWTTenantResolver(TenantResolver):
    """
    Resolve tenant from JWT token.

    Extracts tenant_id claim from JWT.
    """

    def __init__(
        self,
        secret_key: str = "",
        algorithm: str = "HS256",
        claim_name: str = "tenant_id",
        token_header: str = "Authorization",
        token_prefix: str = "Bearer",
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.claim_name = claim_name
        self.token_header = token_header
        self.token_prefix = token_prefix

    async def resolve(self, request: Request) -> Optional[str]:
        """Extract tenant ID from JWT."""
        auth_header = request.headers.get(self.token_header)
        if not auth_header:
            return None

        # Extract token
        parts = auth_header.split()
        if len(parts) != 2 or parts[0] != self.token_prefix:
            return None

        token = parts[1]

        try:
            # Decode JWT
            if self.secret_key:
                payload = jwt.decode(
                    token,
                    self.secret_key,
                    algorithms=[self.algorithm],
                )
            else:
                # Decode without verification (for extracting tenant_id only)
                payload = jwt.decode(
                    token,
                    options={"verify_signature": False},
                )

            return payload.get(self.claim_name)

        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None


class QueryParamTenantResolver(TenantResolver):
    """
    Resolve tenant from query parameter.

    Useful for webhooks and callbacks.
    """

    def __init__(self, param_name: str = "tenant_id"):
        self.param_name = param_name

    async def resolve(self, request: Request) -> Optional[str]:
        """Get tenant ID from query parameter."""
        return request.query_params.get(self.param_name)


class CompositeTenantResolver(TenantResolver):
    """
    Composite resolver using multiple strategies.

    Tries resolvers in order until one succeeds.
    """

    def __init__(self, resolvers: Optional[List[TenantResolver]] = None):
        self.resolvers = resolvers or [
            JWTTenantResolver(),
            HeaderTenantResolver(),
            SubdomainTenantResolver(),
            PathTenantResolver(),
            QueryParamTenantResolver(),
        ]

    async def resolve(self, request: Request) -> Optional[str]:
        """Try all resolvers in order."""
        for resolver in self.resolvers:
            tenant_id = await resolver.resolve(request)
            if tenant_id:
                logger.debug(f"Tenant resolved by {resolver.__class__.__name__}: {tenant_id}")
                return tenant_id
        return None


@dataclass
class TenantMiddlewareConfig:
    """Configuration for tenant middleware."""
    # Resolution
    resolver: Optional[TenantResolver] = None

    # Validation
    validate_tenant: bool = True
    require_active_tenant: bool = True

    # Paths
    excluded_paths: Set[str] = None
    public_paths: Set[str] = None

    # Error handling
    raise_on_missing: bool = False
    missing_tenant_status: int = 401
    invalid_tenant_status: int = 403

    # Context enrichment
    enrich_context: bool = True

    def __post_init__(self):
        self.excluded_paths = self.excluded_paths or {
            "/health", "/healthz", "/ready", "/readiness",
            "/metrics", "/docs", "/openapi.json", "/redoc",
        }
        self.public_paths = self.public_paths or {
            "/", "/api/v1/auth/login", "/api/v1/auth/register",
            "/api/v1/auth/forgot-password",
        }


class TenantMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for tenant context.

    Resolves and validates tenant for each request.
    """

    def __init__(
        self,
        app,
        config: Optional[TenantMiddlewareConfig] = None,
        tenant_manager: Optional[TenantManager] = None,
    ):
        super().__init__(app)
        self.config = config or TenantMiddlewareConfig()
        self.resolver = self.config.resolver or CompositeTenantResolver()
        self.tenant_manager = tenant_manager

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with tenant context."""
        # Check if path is excluded
        if self._is_excluded_path(request.url.path):
            return await call_next(request)

        # Check if path is public
        if self._is_public_path(request.url.path):
            return await call_next(request)

        # Resolve tenant
        tenant_id = await self.resolver.resolve(request)

        if not tenant_id:
            if self.config.raise_on_missing:
                raise HTTPException(
                    status_code=self.config.missing_tenant_status,
                    detail="Tenant identification required",
                )
            return await call_next(request)

        # Validate tenant if configured
        tenant = None
        if self.config.validate_tenant and self.tenant_manager:
            tenant = await self._validate_tenant(tenant_id)
            if not tenant:
                raise HTTPException(
                    status_code=self.config.invalid_tenant_status,
                    detail="Invalid or inactive tenant",
                )

        # Create tenant context
        context = await self._create_context(request, tenant_id, tenant)

        # Set tenant context
        token = set_current_tenant(context)

        try:
            # Add tenant info to request state
            request.state.tenant_id = tenant_id
            request.state.tenant_context = context
            if tenant:
                request.state.tenant = tenant

            # Process request
            response = await call_next(request)

            # Add tenant header to response
            response.headers["X-Tenant-ID"] = tenant_id

            return response

        finally:
            # Clear tenant context
            clear_tenant_context(token)

    def _is_excluded_path(self, path: str) -> bool:
        """Check if path is excluded from tenant handling."""
        return path in self.config.excluded_paths

    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (no tenant required)."""
        return path in self.config.public_paths

    async def _validate_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Validate tenant exists and is active."""
        tenant = await self.tenant_manager.get_tenant(tenant_id)

        if not tenant:
            # Try by slug
            tenant = await self.tenant_manager.get_tenant_by_slug(tenant_id)

        if not tenant:
            logger.warning(f"Tenant not found: {tenant_id}")
            return None

        if self.config.require_active_tenant and not tenant.can_access():
            logger.warning(f"Tenant not active: {tenant_id} (status: {tenant.status})")
            return None

        return tenant

    async def _create_context(
        self,
        request: Request,
        tenant_id: str,
        tenant: Optional[Tenant],
    ) -> TenantContext:
        """Create tenant context from request."""
        import uuid

        context = TenantContext(
            tenant_id=tenant_id,
            request_id=request.headers.get("X-Request-ID", str(uuid.uuid4())),
            correlation_id=request.headers.get("X-Correlation-ID"),
        )

        if tenant:
            context.tenant_slug = tenant.slug
            context.tenant_name = tenant.name
            context.tier = tenant.tier.value

        # Extract user info from JWT if available
        if self.config.enrich_context:
            await self._enrich_from_jwt(request, context)

        return context

    async def _enrich_from_jwt(
        self,
        request: Request,
        context: TenantContext,
    ) -> None:
        """Enrich context with JWT claims."""
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return

        token = auth_header[7:]
        try:
            # Decode without verification for enrichment only
            payload = jwt.decode(
                token,
                options={"verify_signature": False},
            )

            context.user_id = payload.get("sub") or payload.get("user_id")
            context.user_email = payload.get("email")
            context.user_roles = payload.get("roles", [])
            context.permissions = set(payload.get("permissions", []))
            context.is_admin = payload.get("is_admin", False)

        except jwt.InvalidTokenError:
            pass


class TenantRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware per tenant.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_size: int = 10,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self._request_counts: Dict[str, List[float]] = {}

    async def dispatch(self, request: Request, call_next) -> Response:
        """Apply rate limiting per tenant."""
        import time

        tenant_id = getattr(request.state, "tenant_id", None)
        if not tenant_id:
            return await call_next(request)

        current_time = time.time()
        window_start = current_time - 60

        # Get request history
        if tenant_id not in self._request_counts:
            self._request_counts[tenant_id] = []

        # Clean old requests
        self._request_counts[tenant_id] = [
            t for t in self._request_counts[tenant_id]
            if t > window_start
        ]

        # Check rate limit
        if len(self._request_counts[tenant_id]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": "60",
                },
            )

        # Record request
        self._request_counts[tenant_id].append(current_time)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        remaining = self.requests_per_minute - len(self._request_counts[tenant_id])
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response


class CrossTenantProtectionMiddleware(BaseHTTPMiddleware):
    """
    Prevents cross-tenant data access.
    """

    def __init__(
        self,
        app,
        protected_paths: Optional[Set[str]] = None,
    ):
        super().__init__(app)
        self.protected_paths = protected_paths or set()

    async def dispatch(self, request: Request, call_next) -> Response:
        """Validate tenant consistency in requests."""
        # Get tenant from context
        context_tenant = getattr(request.state, "tenant_id", None)
        if not context_tenant:
            return await call_next(request)

        # Check for tenant_id in path parameters
        path_params = getattr(request, "path_params", {})
        path_tenant = path_params.get("tenant_id")

        if path_tenant and path_tenant != context_tenant:
            logger.warning(
                f"Cross-tenant access attempt: context={context_tenant}, path={path_tenant}"
            )
            raise HTTPException(
                status_code=403,
                detail="Cross-tenant access not allowed",
            )

        # Check for tenant_id in query parameters
        query_tenant = request.query_params.get("tenant_id")
        if query_tenant and query_tenant != context_tenant:
            logger.warning(
                f"Cross-tenant access attempt: context={context_tenant}, query={query_tenant}"
            )
            raise HTTPException(
                status_code=403,
                detail="Cross-tenant access not allowed",
            )

        return await call_next(request)


def require_tenant_middleware(
    raise_on_missing: bool = True,
    excluded_paths: Optional[Set[str]] = None,
) -> Callable:
    """
    Create tenant middleware with custom configuration.

    Usage:
        app.add_middleware(require_tenant_middleware(raise_on_missing=True))
    """
    config = TenantMiddlewareConfig(
        raise_on_missing=raise_on_missing,
        excluded_paths=excluded_paths or set(),
    )

    def middleware(app):
        return TenantMiddleware(app, config=config)

    return middleware
