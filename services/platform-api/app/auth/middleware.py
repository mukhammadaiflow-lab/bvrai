"""Authentication middleware for FastAPI."""

from typing import Optional, Callable, List, Set
from dataclasses import dataclass
from datetime import datetime
import time
import logging

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.auth.jwt import (
    JWTManager,
    APIKeyGenerator,
    TokenClaims,
    TokenExpiredError,
    InvalidTokenError,
    get_jwt_manager,
    get_api_key_generator,
)
from app.monitoring.logging import (
    set_correlation_id,
    set_request_context,
    clear_request_context,
    get_correlation_id,
)

logger = logging.getLogger(__name__)


@dataclass
class AuthenticatedUser:
    """Authenticated user context."""
    user_id: str
    org_id: Optional[str]
    scopes: List[str]
    token_type: str
    is_test_key: bool
    authenticated_at: datetime
    api_key_id: Optional[str] = None

    def has_scope(self, scope: str) -> bool:
        """Check if user has a specific scope."""
        return scope in self.scopes or "*" in self.scopes

    def has_any_scope(self, *scopes: str) -> bool:
        """Check if user has any of the specified scopes."""
        return any(self.has_scope(s) for s in scopes)

    def has_all_scopes(self, *scopes: str) -> bool:
        """Check if user has all specified scopes."""
        return all(self.has_scope(s) for s in scopes)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware.

    Extracts authentication info from headers and sets user context.
    Supports both JWT tokens and API keys.
    """

    def __init__(
        self,
        app: ASGIApp,
        jwt_manager: Optional[JWTManager] = None,
        api_key_generator: Optional[APIKeyGenerator] = None,
        exclude_paths: Optional[Set[str]] = None,
        exclude_prefixes: Optional[List[str]] = None,
        require_auth: bool = False,
    ):
        super().__init__(app)
        self.jwt_manager = jwt_manager or get_jwt_manager()
        self.api_key_generator = api_key_generator or get_api_key_generator()
        self.exclude_paths = exclude_paths or {
            "/",
            "/health",
            "/healthz",
            "/ready",
            "/readyz",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        }
        self.exclude_prefixes = exclude_prefixes or ["/api/v1/public"]
        self.require_auth = require_auth

    def _should_skip_auth(self, path: str) -> bool:
        """Check if path should skip authentication."""
        if path in self.exclude_paths:
            return True
        for prefix in self.exclude_prefixes:
            if path.startswith(prefix):
                return True
        return False

    def _extract_token(self, request: Request) -> Optional[str]:
        """Extract token from request headers."""
        # Check X-API-Key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key

        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header:
            if auth_header.startswith("Bearer "):
                return auth_header[7:]
            elif auth_header.startswith("ApiKey "):
                return auth_header[7:]

        return None

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request and authenticate."""
        # Set correlation ID
        correlation_id = request.headers.get(
            "X-Correlation-ID",
            request.headers.get("X-Request-ID"),
        )
        set_correlation_id(correlation_id)

        # Set request context for logging
        set_request_context({
            "method": request.method,
            "path": request.url.path,
            "client_ip": self._get_client_ip(request),
        })

        try:
            # Check if auth should be skipped
            if self._should_skip_auth(request.url.path):
                return await call_next(request)

            # Extract token
            token = self._extract_token(request)

            if not token:
                if self.require_auth:
                    return Response(
                        content='{"detail":"Authentication required"}',
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        media_type="application/json",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                return await call_next(request)

            # Authenticate
            try:
                user = await self._authenticate(token, request)
                request.state.user = user
                request.state.authenticated = True
            except (TokenExpiredError, InvalidTokenError) as e:
                if self.require_auth:
                    return Response(
                        content=f'{{"detail":"{str(e)}"}}',
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        media_type="application/json",
                    )
                request.state.user = None
                request.state.authenticated = False

            return await call_next(request)

        finally:
            clear_request_context()

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP from request."""
        # Check X-Forwarded-For header (for proxied requests)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to client host
        return request.client.host if request.client else "unknown"

    async def _authenticate(
        self,
        token: str,
        request: Request,
    ) -> AuthenticatedUser:
        """Authenticate using token or API key."""
        # Check if it's an API key
        if self.api_key_generator.is_valid_format(token):
            return await self._authenticate_api_key(token, request)

        # Otherwise, try JWT
        return await self._authenticate_jwt(token)

    async def _authenticate_jwt(self, token: str) -> AuthenticatedUser:
        """Authenticate using JWT token."""
        claims = self.jwt_manager.decode_token(token)

        return AuthenticatedUser(
            user_id=claims.sub,
            org_id=claims.org_id,
            scopes=claims.scopes,
            token_type=claims.token_type.value,
            is_test_key=False,
            authenticated_at=datetime.utcnow(),
        )

    async def _authenticate_api_key(
        self,
        api_key: str,
        request: Request,
    ) -> AuthenticatedUser:
        """Authenticate using API key."""
        # In production, this would look up the key in the database
        # For now, we just validate format and assume it's valid
        # The actual lookup happens in the dependencies

        # Hash the key for lookup
        key_hash = self.api_key_generator.hash_key(api_key)

        return AuthenticatedUser(
            user_id="pending",  # Will be resolved by dependencies
            org_id=None,
            scopes=["*"],  # Will be resolved by dependencies
            token_type="api_key",
            is_test_key=self.api_key_generator.is_test_key(api_key),
            authenticated_at=datetime.utcnow(),
            api_key_id=key_hash[:16],  # Short identifier
        )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware.

    For more advanced rate limiting, use the resilience module.
    """

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 100,
        exclude_paths: Optional[Set[str]] = None,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.exclude_paths = exclude_paths or {"/health", "/healthz", "/metrics"}
        self._requests: dict = {}
        self._window = 60  # seconds

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Check rate limit and process request."""
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Get client identifier
        client_id = self._get_client_id(request)
        current_time = int(time.time())
        window_start = current_time - self._window

        # Clean old entries and count recent requests
        if client_id in self._requests:
            self._requests[client_id] = [
                ts for ts in self._requests[client_id]
                if ts > window_start
            ]
            request_count = len(self._requests[client_id])
        else:
            self._requests[client_id] = []
            request_count = 0

        # Check limit
        if request_count >= self.requests_per_minute:
            retry_after = self._window - (current_time - min(self._requests[client_id]))
            return Response(
                content='{"detail":"Rate limit exceeded"}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(current_time + retry_after),
                },
            )

        # Record request
        self._requests[client_id].append(current_time)

        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - request_count - 1
        )
        response.headers["X-RateLimit-Reset"] = str(current_time + self._window)

        return response

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use authenticated user if available
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user.user_id}"

        # Fall back to IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"

        if request.client:
            return f"ip:{request.client.host}"

        return "ip:unknown"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Request logging middleware.

    Logs all requests with timing and status information.
    """

    def __init__(
        self,
        app: ASGIApp,
        log_headers: bool = False,
        exclude_paths: Optional[Set[str]] = None,
    ):
        super().__init__(app)
        self.log_headers = log_headers
        self.exclude_paths = exclude_paths or {"/health", "/healthz", "/metrics"}

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Log request and response."""
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        start_time = time.perf_counter()

        # Log request
        log_data = {
            "method": request.method,
            "path": request.url.path,
            "client_ip": self._get_client_ip(request),
            "correlation_id": get_correlation_id(),
        }

        if self.log_headers:
            log_data["headers"] = dict(request.headers)

        logger.info(f"Request started: {request.method} {request.url.path}")

        try:
            response = await call_next(request)

            # Log response
            duration_ms = (time.perf_counter() - start_time) * 1000
            log_data["status_code"] = response.status_code
            log_data["duration_ms"] = round(duration_ms, 2)

            if response.status_code >= 500:
                logger.error(
                    f"Request error: {request.method} {request.url.path} "
                    f"-> {response.status_code} ({duration_ms:.2f}ms)"
                )
            elif response.status_code >= 400:
                logger.warning(
                    f"Request failed: {request.method} {request.url.path} "
                    f"-> {response.status_code} ({duration_ms:.2f}ms)"
                )
            else:
                logger.info(
                    f"Request completed: {request.method} {request.url.path} "
                    f"-> {response.status_code} ({duration_ms:.2f}ms)"
                )

            # Add timing header
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

            return response

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(
                f"Request exception: {request.method} {request.url.path} "
                f"({duration_ms:.2f}ms)"
            )
            raise

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"


class CORSMiddleware:
    """
    Custom CORS middleware.

    More flexible than the standard Starlette CORS middleware.
    """

    def __init__(
        self,
        app: ASGIApp,
        allow_origins: Optional[List[str]] = None,
        allow_methods: Optional[List[str]] = None,
        allow_headers: Optional[List[str]] = None,
        allow_credentials: bool = True,
        max_age: int = 86400,
    ):
        self.app = app
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["*"]
        self.allow_headers = allow_headers or ["*"]
        self.allow_credentials = allow_credentials
        self.max_age = max_age

    async def __call__(self, scope, receive, send):
        """Handle CORS."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        origin = request.headers.get("origin")

        # Handle preflight request
        if request.method == "OPTIONS":
            response = Response(status_code=200)
            self._add_cors_headers(response, origin)
            await response(scope, receive, send)
            return

        # Handle regular request
        async def send_with_cors(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))

                if origin and self._is_allowed_origin(origin):
                    headers.append((b"access-control-allow-origin", origin.encode()))
                    if self.allow_credentials:
                        headers.append((b"access-control-allow-credentials", b"true"))

                message["headers"] = headers

            await send(message)

        await self.app(scope, receive, send_with_cors)

    def _is_allowed_origin(self, origin: str) -> bool:
        """Check if origin is allowed."""
        if "*" in self.allow_origins:
            return True
        return origin in self.allow_origins

    def _add_cors_headers(self, response: Response, origin: Optional[str]) -> None:
        """Add CORS headers to response."""
        if origin and self._is_allowed_origin(origin):
            response.headers["Access-Control-Allow-Origin"] = origin

        if "*" in self.allow_methods:
            response.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, PUT, PATCH, DELETE, OPTIONS"
            )
        else:
            response.headers["Access-Control-Allow-Methods"] = ", ".join(
                self.allow_methods
            )

        if "*" in self.allow_headers:
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization, X-API-Key, X-Correlation-ID"
            )
        else:
            response.headers["Access-Control-Allow-Headers"] = ", ".join(
                self.allow_headers
            )

        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"

        response.headers["Access-Control-Max-Age"] = str(self.max_age)


def get_current_user(request: Request) -> Optional[AuthenticatedUser]:
    """Get current authenticated user from request."""
    return getattr(request.state, "user", None)


def require_auth(request: Request) -> AuthenticatedUser:
    """Require authentication for endpoint."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return user


def require_scopes(*required_scopes: str):
    """
    Require specific scopes for endpoint.

    Usage:
        @app.get("/admin")
        async def admin_endpoint(user: AuthenticatedUser = Depends(require_scopes("admin"))):
            ...
    """
    def dependency(request: Request) -> AuthenticatedUser:
        user = require_auth(request)
        if not user.has_all_scopes(*required_scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scopes: {set(required_scopes) - set(user.scopes)}",
            )
        return user

    return dependency
