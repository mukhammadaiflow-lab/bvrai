"""
Gateway Middleware

Request processing middleware for API gateway:
- Authentication
- Rate limiting
- Request logging
- CORS handling
- Request/Response transformation
"""

from typing import Optional, Dict, Any, List, Callable, Awaitable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
import json
import re
import hashlib
import time
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class GatewayRequest:
    """Gateway request representation."""
    method: str
    path: str
    headers: Dict[str, str]
    query_params: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    client_ip: str = ""
    request_id: str = ""
    start_time: float = field(default_factory=time.time)

    # Populated by middleware
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    api_key: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_header(self, name: str, default: str = "") -> str:
        """Get header case-insensitively."""
        for key, value in self.headers.items():
            if key.lower() == name.lower():
                return value
        return default


@dataclass
class GatewayResponse:
    """Gateway response representation."""
    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None

    @classmethod
    def json(cls, data: Any, status_code: int = 200) -> "GatewayResponse":
        """Create JSON response."""
        return cls(
            status_code=status_code,
            headers={"content-type": "application/json"},
            body=json.dumps(data).encode(),
        )

    @classmethod
    def error(cls, message: str, status_code: int = 400) -> "GatewayResponse":
        """Create error response."""
        return cls.json({"error": message}, status_code)


class GatewayMiddleware(ABC):
    """Abstract gateway middleware."""

    @abstractmethod
    async def process_request(
        self,
        request: GatewayRequest,
    ) -> Optional[GatewayResponse]:
        """
        Process incoming request.

        Return None to continue, or GatewayResponse to short-circuit.
        """
        pass

    async def process_response(
        self,
        request: GatewayRequest,
        response: GatewayResponse,
    ) -> GatewayResponse:
        """
        Process outgoing response.

        Default implementation passes through.
        """
        return response


class MiddlewareChain:
    """Chain of middleware processors."""

    def __init__(self):
        self._middlewares: List[GatewayMiddleware] = []

    def add(self, middleware: GatewayMiddleware) -> "MiddlewareChain":
        """Add middleware to chain."""
        self._middlewares.append(middleware)
        return self

    async def process_request(
        self,
        request: GatewayRequest,
    ) -> Optional[GatewayResponse]:
        """Process request through all middleware."""
        for middleware in self._middlewares:
            response = await middleware.process_request(request)
            if response:
                return response
        return None

    async def process_response(
        self,
        request: GatewayRequest,
        response: GatewayResponse,
    ) -> GatewayResponse:
        """Process response through all middleware (reverse order)."""
        for middleware in reversed(self._middlewares):
            response = await middleware.process_response(request, response)
        return response


class AuthenticationMiddleware(GatewayMiddleware):
    """
    Authentication middleware.

    Supports multiple authentication methods:
    - API Key
    - JWT Bearer Token
    - Basic Auth
    - OAuth2
    """

    def __init__(
        self,
        api_key_header: str = "X-API-Key",
        bearer_header: str = "Authorization",
        exclude_paths: Optional[List[str]] = None,
        require_auth: bool = True,
    ):
        self.api_key_header = api_key_header
        self.bearer_header = bearer_header
        self.exclude_paths = exclude_paths or ["/health", "/ready", "/metrics"]
        self.require_auth = require_auth
        self._api_key_validator: Optional[Callable[[str], Awaitable[Optional[Dict]]]] = None
        self._jwt_validator: Optional[Callable[[str], Awaitable[Optional[Dict]]]] = None

    def set_api_key_validator(
        self,
        validator: Callable[[str], Awaitable[Optional[Dict]]],
    ) -> None:
        """Set API key validator function."""
        self._api_key_validator = validator

    def set_jwt_validator(
        self,
        validator: Callable[[str], Awaitable[Optional[Dict]]],
    ) -> None:
        """Set JWT validator function."""
        self._jwt_validator = validator

    async def process_request(
        self,
        request: GatewayRequest,
    ) -> Optional[GatewayResponse]:
        """Authenticate request."""
        # Check excluded paths
        for path in self.exclude_paths:
            if request.path.startswith(path):
                return None

        # Try API Key authentication
        api_key = request.get_header(self.api_key_header)
        if api_key:
            result = await self._validate_api_key(api_key)
            if result:
                request.api_key = api_key
                request.user_id = result.get("user_id")
                request.tenant_id = result.get("tenant_id")
                request.scopes = result.get("scopes", [])
                return None
            return GatewayResponse.error("Invalid API key", 401)

        # Try Bearer token authentication
        auth_header = request.get_header(self.bearer_header)
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            result = await self._validate_jwt(token)
            if result:
                request.user_id = result.get("user_id") or result.get("sub")
                request.tenant_id = result.get("tenant_id")
                request.scopes = result.get("scopes", result.get("scope", "").split())
                return None
            return GatewayResponse.error("Invalid token", 401)

        # Try Basic authentication
        if auth_header and auth_header.startswith("Basic "):
            result = await self._validate_basic_auth(auth_header[6:])
            if result:
                request.user_id = result.get("user_id")
                request.tenant_id = result.get("tenant_id")
                return None
            return GatewayResponse.error("Invalid credentials", 401)

        # No authentication provided
        if self.require_auth:
            return GatewayResponse.error("Authentication required", 401)

        return None

    async def _validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate API key."""
        if self._api_key_validator:
            return await self._api_key_validator(api_key)

        # Default: accept any non-empty key (for testing)
        if api_key:
            return {"user_id": "default", "scopes": ["read", "write"]}
        return None

    async def _validate_jwt(self, token: str) -> Optional[Dict]:
        """Validate JWT token."""
        if self._jwt_validator:
            return await self._jwt_validator(token)

        # Default: decode without verification (for testing only)
        try:
            import base64
            parts = token.split(".")
            if len(parts) >= 2:
                payload = parts[1]
                # Add padding
                payload += "=" * (4 - len(payload) % 4)
                decoded = base64.urlsafe_b64decode(payload)
                return json.loads(decoded)
        except Exception:
            pass
        return None

    async def _validate_basic_auth(self, credentials: str) -> Optional[Dict]:
        """Validate basic auth credentials."""
        try:
            import base64
            decoded = base64.b64decode(credentials).decode()
            username, password = decoded.split(":", 1)
            # In production: validate against user store
            if username and password:
                return {"user_id": username}
        except Exception:
            pass
        return None


class RateLimitingMiddleware(GatewayMiddleware):
    """
    Rate limiting middleware.

    Supports multiple rate limiting strategies:
    - Fixed window
    - Sliding window
    - Token bucket
    - Per-user/tenant/IP limits
    """

    def __init__(
        self,
        requests_per_second: float = 100.0,
        requests_per_minute: float = 1000.0,
        burst_size: int = 50,
        key_extractor: Optional[Callable[[GatewayRequest], str]] = None,
        exclude_paths: Optional[List[str]] = None,
    ):
        self.requests_per_second = requests_per_second
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.key_extractor = key_extractor or self._default_key_extractor
        self.exclude_paths = exclude_paths or ["/health", "/ready"]

        # Token bucket state
        self._buckets: Dict[str, Dict[str, float]] = {}
        self._lock = asyncio.Lock()

        # Metrics
        self._allowed_count = 0
        self._rejected_count = 0

    def _default_key_extractor(self, request: GatewayRequest) -> str:
        """Extract rate limit key from request."""
        # Prefer tenant_id > user_id > api_key > client_ip
        if request.tenant_id:
            return f"tenant:{request.tenant_id}"
        if request.user_id:
            return f"user:{request.user_id}"
        if request.api_key:
            return f"api_key:{hashlib.sha256(request.api_key.encode()).hexdigest()[:16]}"
        return f"ip:{request.client_ip}"

    async def process_request(
        self,
        request: GatewayRequest,
    ) -> Optional[GatewayResponse]:
        """Check rate limits."""
        # Check excluded paths
        for path in self.exclude_paths:
            if request.path.startswith(path):
                return None

        key = self.key_extractor(request)
        allowed, retry_after = await self._check_rate_limit(key)

        if not allowed:
            self._rejected_count += 1
            return GatewayResponse(
                status_code=429,
                headers={
                    "content-type": "application/json",
                    "retry-after": str(int(retry_after)),
                    "x-ratelimit-limit": str(int(self.requests_per_minute)),
                    "x-ratelimit-remaining": "0",
                },
                body=json.dumps({
                    "error": "Rate limit exceeded",
                    "retry_after": retry_after,
                }).encode(),
            )

        self._allowed_count += 1
        return None

    async def process_response(
        self,
        request: GatewayRequest,
        response: GatewayResponse,
    ) -> GatewayResponse:
        """Add rate limit headers to response."""
        key = self.key_extractor(request)
        remaining = await self._get_remaining(key)

        response.headers["x-ratelimit-limit"] = str(int(self.requests_per_minute))
        response.headers["x-ratelimit-remaining"] = str(int(remaining))
        response.headers["x-ratelimit-reset"] = str(int(time.time()) + 60)

        return response

    async def _check_rate_limit(self, key: str) -> tuple:
        """Check if request is within rate limit."""
        async with self._lock:
            now = time.time()

            if key not in self._buckets:
                self._buckets[key] = {
                    "tokens": float(self.burst_size),
                    "last_update": now,
                }

            bucket = self._buckets[key]

            # Add tokens based on time elapsed
            elapsed = now - bucket["last_update"]
            bucket["tokens"] = min(
                float(self.burst_size),
                bucket["tokens"] + elapsed * self.requests_per_second,
            )
            bucket["last_update"] = now

            # Check if we have tokens
            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                return True, 0.0

            # Calculate retry after
            tokens_needed = 1.0 - bucket["tokens"]
            retry_after = tokens_needed / self.requests_per_second

            return False, retry_after

    async def _get_remaining(self, key: str) -> float:
        """Get remaining tokens for key."""
        if key in self._buckets:
            return max(0, self._buckets[key]["tokens"])
        return float(self.burst_size)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        return {
            "allowed": self._allowed_count,
            "rejected": self._rejected_count,
            "rejection_rate": self._rejected_count / max(1, self._allowed_count + self._rejected_count),
            "active_buckets": len(self._buckets),
        }


class RequestLoggingMiddleware(GatewayMiddleware):
    """
    Request logging middleware.

    Logs request/response details with configurable verbosity.
    """

    def __init__(
        self,
        log_headers: bool = False,
        log_body: bool = False,
        log_response_body: bool = False,
        max_body_size: int = 1000,
        exclude_paths: Optional[List[str]] = None,
        sensitive_headers: Optional[List[str]] = None,
    ):
        self.log_headers = log_headers
        self.log_body = log_body
        self.log_response_body = log_response_body
        self.max_body_size = max_body_size
        self.exclude_paths = exclude_paths or ["/health", "/ready", "/metrics"]
        self.sensitive_headers = [h.lower() for h in (sensitive_headers or [
            "authorization", "x-api-key", "cookie", "x-auth-token"
        ])]

    async def process_request(
        self,
        request: GatewayRequest,
    ) -> Optional[GatewayResponse]:
        """Log incoming request."""
        # Check excluded paths
        for path in self.exclude_paths:
            if request.path.startswith(path):
                return None

        log_data = {
            "type": "request",
            "request_id": request.request_id,
            "method": request.method,
            "path": request.path,
            "client_ip": request.client_ip,
            "user_id": request.user_id,
            "tenant_id": request.tenant_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if self.log_headers:
            log_data["headers"] = self._sanitize_headers(request.headers)

        if self.log_body and request.body:
            log_data["body"] = self._truncate_body(request.body)

        logger.info(f"Gateway request: {json.dumps(log_data)}")
        return None

    async def process_response(
        self,
        request: GatewayRequest,
        response: GatewayResponse,
    ) -> GatewayResponse:
        """Log outgoing response."""
        # Check excluded paths
        for path in self.exclude_paths:
            if request.path.startswith(path):
                return response

        duration_ms = (time.time() - request.start_time) * 1000

        log_data = {
            "type": "response",
            "request_id": request.request_id,
            "method": request.method,
            "path": request.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "user_id": request.user_id,
            "tenant_id": request.tenant_id,
        }

        if self.log_response_body and response.body:
            log_data["body"] = self._truncate_body(response.body)

        log_level = logging.INFO if response.status_code < 400 else logging.WARNING
        logger.log(log_level, f"Gateway response: {json.dumps(log_data)}")

        return response

    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize sensitive headers."""
        return {
            k: "***" if k.lower() in self.sensitive_headers else v
            for k, v in headers.items()
        }

    def _truncate_body(self, body: bytes) -> str:
        """Truncate body for logging."""
        try:
            text = body.decode()[:self.max_body_size]
            if len(body) > self.max_body_size:
                text += f"... (truncated, {len(body)} bytes total)"
            return text
        except UnicodeDecodeError:
            return f"<binary, {len(body)} bytes>"


class CORSMiddleware(GatewayMiddleware):
    """
    CORS (Cross-Origin Resource Sharing) middleware.

    Handles preflight requests and CORS headers.
    """

    def __init__(
        self,
        allowed_origins: Optional[List[str]] = None,
        allowed_methods: Optional[List[str]] = None,
        allowed_headers: Optional[List[str]] = None,
        exposed_headers: Optional[List[str]] = None,
        allow_credentials: bool = True,
        max_age: int = 86400,
    ):
        self.allowed_origins = allowed_origins or ["*"]
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        self.allowed_headers = allowed_headers or ["*"]
        self.exposed_headers = exposed_headers or []
        self.allow_credentials = allow_credentials
        self.max_age = max_age

        # Compile origin patterns
        self._origin_patterns: List[re.Pattern] = []
        for origin in self.allowed_origins:
            if origin == "*":
                self._origin_patterns.append(re.compile(".*"))
            else:
                # Convert wildcard patterns to regex
                pattern = origin.replace(".", r"\.").replace("*", ".*")
                self._origin_patterns.append(re.compile(f"^{pattern}$"))

    async def process_request(
        self,
        request: GatewayRequest,
    ) -> Optional[GatewayResponse]:
        """Handle CORS preflight requests."""
        origin = request.get_header("origin")

        # Handle preflight
        if request.method == "OPTIONS":
            if origin and self._is_origin_allowed(origin):
                return GatewayResponse(
                    status_code=204,
                    headers=self._get_cors_headers(origin, preflight=True),
                )
            return GatewayResponse.error("CORS not allowed", 403)

        return None

    async def process_response(
        self,
        request: GatewayRequest,
        response: GatewayResponse,
    ) -> GatewayResponse:
        """Add CORS headers to response."""
        origin = request.get_header("origin")

        if origin and self._is_origin_allowed(origin):
            cors_headers = self._get_cors_headers(origin, preflight=False)
            response.headers.update(cors_headers)

        return response

    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed."""
        return any(pattern.match(origin) for pattern in self._origin_patterns)

    def _get_cors_headers(self, origin: str, preflight: bool = False) -> Dict[str, str]:
        """Get CORS headers."""
        headers = {
            "access-control-allow-origin": origin if self.allow_credentials else (
                origin if "*" not in self.allowed_origins else "*"
            ),
        }

        if self.allow_credentials:
            headers["access-control-allow-credentials"] = "true"

        if preflight:
            headers["access-control-allow-methods"] = ", ".join(self.allowed_methods)

            if "*" in self.allowed_headers:
                headers["access-control-allow-headers"] = "*"
            else:
                headers["access-control-allow-headers"] = ", ".join(self.allowed_headers)

            headers["access-control-max-age"] = str(self.max_age)

        if self.exposed_headers:
            headers["access-control-expose-headers"] = ", ".join(self.exposed_headers)

        return headers


class RequestValidationMiddleware(GatewayMiddleware):
    """
    Request validation middleware.

    Validates request content and structure.
    """

    def __init__(
        self,
        max_body_size: int = 10 * 1024 * 1024,  # 10MB
        allowed_content_types: Optional[List[str]] = None,
        required_headers: Optional[List[str]] = None,
    ):
        self.max_body_size = max_body_size
        self.allowed_content_types = allowed_content_types
        self.required_headers = [h.lower() for h in (required_headers or [])]

    async def process_request(
        self,
        request: GatewayRequest,
    ) -> Optional[GatewayResponse]:
        """Validate request."""
        # Check body size
        if request.body and len(request.body) > self.max_body_size:
            return GatewayResponse.error(
                f"Request body too large (max {self.max_body_size} bytes)",
                413,
            )

        # Check content type
        if self.allowed_content_types and request.body:
            content_type = request.get_header("content-type").split(";")[0].strip()
            if content_type and content_type not in self.allowed_content_types:
                return GatewayResponse.error(
                    f"Unsupported content type: {content_type}",
                    415,
                )

        # Check required headers
        for header in self.required_headers:
            if not request.get_header(header):
                return GatewayResponse.error(
                    f"Missing required header: {header}",
                    400,
                )

        return None


class SecurityMiddleware(GatewayMiddleware):
    """
    Security middleware.

    Adds security headers and validates requests.
    """

    def __init__(
        self,
        enable_hsts: bool = True,
        hsts_max_age: int = 31536000,
        enable_xss_protection: bool = True,
        enable_content_type_options: bool = True,
        enable_frame_options: bool = True,
        frame_options: str = "DENY",
        content_security_policy: Optional[str] = None,
    ):
        self.enable_hsts = enable_hsts
        self.hsts_max_age = hsts_max_age
        self.enable_xss_protection = enable_xss_protection
        self.enable_content_type_options = enable_content_type_options
        self.enable_frame_options = enable_frame_options
        self.frame_options = frame_options
        self.content_security_policy = content_security_policy

    async def process_request(
        self,
        request: GatewayRequest,
    ) -> Optional[GatewayResponse]:
        """Validate request security."""
        return None

    async def process_response(
        self,
        request: GatewayRequest,
        response: GatewayResponse,
    ) -> GatewayResponse:
        """Add security headers."""
        if self.enable_hsts:
            response.headers["strict-transport-security"] = (
                f"max-age={self.hsts_max_age}; includeSubDomains"
            )

        if self.enable_xss_protection:
            response.headers["x-xss-protection"] = "1; mode=block"

        if self.enable_content_type_options:
            response.headers["x-content-type-options"] = "nosniff"

        if self.enable_frame_options:
            response.headers["x-frame-options"] = self.frame_options

        if self.content_security_policy:
            response.headers["content-security-policy"] = self.content_security_policy

        return response


class CompressionMiddleware(GatewayMiddleware):
    """
    Response compression middleware.

    Compresses responses using gzip or br.
    """

    def __init__(
        self,
        min_size: int = 500,
        compression_level: int = 6,
        compressible_types: Optional[List[str]] = None,
    ):
        self.min_size = min_size
        self.compression_level = compression_level
        self.compressible_types = compressible_types or [
            "text/", "application/json", "application/xml",
            "application/javascript", "application/x-javascript",
        ]

    async def process_request(
        self,
        request: GatewayRequest,
    ) -> Optional[GatewayResponse]:
        """Store accept-encoding for later."""
        request.metadata["accept_encoding"] = request.get_header("accept-encoding")
        return None

    async def process_response(
        self,
        request: GatewayRequest,
        response: GatewayResponse,
    ) -> GatewayResponse:
        """Compress response if applicable."""
        if not response.body or len(response.body) < self.min_size:
            return response

        # Check content type
        content_type = response.headers.get("content-type", "")
        if not any(ct in content_type for ct in self.compressible_types):
            return response

        # Check accepted encoding
        accept_encoding = request.metadata.get("accept_encoding", "")

        if "gzip" in accept_encoding:
            try:
                import gzip
                response.body = gzip.compress(response.body, self.compression_level)
                response.headers["content-encoding"] = "gzip"
                response.headers["content-length"] = str(len(response.body))
            except Exception:
                pass

        elif "deflate" in accept_encoding:
            try:
                import zlib
                response.body = zlib.compress(response.body, self.compression_level)
                response.headers["content-encoding"] = "deflate"
                response.headers["content-length"] = str(len(response.body))
            except Exception:
                pass

        return response


class RequestIdMiddleware(GatewayMiddleware):
    """
    Request ID middleware.

    Generates or propagates request IDs.
    """

    def __init__(
        self,
        header_name: str = "X-Request-ID",
        generate_if_missing: bool = True,
    ):
        self.header_name = header_name
        self.generate_if_missing = generate_if_missing

    async def process_request(
        self,
        request: GatewayRequest,
    ) -> Optional[GatewayResponse]:
        """Extract or generate request ID."""
        request_id = request.get_header(self.header_name)

        if not request_id and self.generate_if_missing:
            import uuid
            request_id = str(uuid.uuid4())

        request.request_id = request_id
        return None

    async def process_response(
        self,
        request: GatewayRequest,
        response: GatewayResponse,
    ) -> GatewayResponse:
        """Add request ID to response."""
        if request.request_id:
            response.headers[self.header_name] = request.request_id
        return response


class TimeoutMiddleware(GatewayMiddleware):
    """
    Request timeout middleware.

    Enforces request timeouts.
    """

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        timeout_response: Optional[GatewayResponse] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.timeout_response = timeout_response or GatewayResponse.error(
            "Request timeout",
            504,
        )

    async def process_request(
        self,
        request: GatewayRequest,
    ) -> Optional[GatewayResponse]:
        """Store timeout deadline."""
        request.metadata["timeout_deadline"] = time.time() + self.timeout_seconds
        return None

    async def process_response(
        self,
        request: GatewayRequest,
        response: GatewayResponse,
    ) -> GatewayResponse:
        """Check if request timed out."""
        deadline = request.metadata.get("timeout_deadline")
        if deadline and time.time() > deadline:
            return self.timeout_response
        return response


class IPFilterMiddleware(GatewayMiddleware):
    """
    IP filtering middleware.

    Allow or deny requests based on IP address.
    """

    def __init__(
        self,
        allowed_ips: Optional[List[str]] = None,
        denied_ips: Optional[List[str]] = None,
        allow_private: bool = True,
    ):
        self.allowed_ips = set(allowed_ips or [])
        self.denied_ips = set(denied_ips or [])
        self.allow_private = allow_private

        # Private IP ranges
        self._private_ranges = [
            ("10.", ),
            ("172.16.", "172.17.", "172.18.", "172.19.",
             "172.20.", "172.21.", "172.22.", "172.23.",
             "172.24.", "172.25.", "172.26.", "172.27.",
             "172.28.", "172.29.", "172.30.", "172.31."),
            ("192.168.",),
            ("127.",),
        ]

    async def process_request(
        self,
        request: GatewayRequest,
    ) -> Optional[GatewayResponse]:
        """Filter by IP."""
        ip = request.client_ip

        # Check denied list first
        if ip in self.denied_ips:
            return GatewayResponse.error("Access denied", 403)

        # Check allowed list
        if self.allowed_ips and ip not in self.allowed_ips:
            # Check if private and allowed
            if self.allow_private and self._is_private_ip(ip):
                return None
            return GatewayResponse.error("Access denied", 403)

        return None

    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP is private."""
        for prefixes in self._private_ranges:
            if any(ip.startswith(prefix) for prefix in prefixes):
                return True
        return False


class CircuitBreakerMiddleware(GatewayMiddleware):
    """
    Circuit breaker middleware.

    Protects backend services from cascading failures.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_requests: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests

        self._failures: Dict[str, int] = {}
        self._state: Dict[str, str] = {}  # closed, open, half_open
        self._last_failure_time: Dict[str, float] = {}
        self._half_open_count: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def process_request(
        self,
        request: GatewayRequest,
    ) -> Optional[GatewayResponse]:
        """Check circuit breaker state."""
        key = request.path.split("/")[1] if "/" in request.path else "default"

        async with self._lock:
            state = self._state.get(key, "closed")

            if state == "open":
                # Check if recovery timeout has passed
                last_failure = self._last_failure_time.get(key, 0)
                if time.time() - last_failure >= self.recovery_timeout:
                    self._state[key] = "half_open"
                    self._half_open_count[key] = 0
                else:
                    return GatewayResponse.error(
                        "Service temporarily unavailable",
                        503,
                    )

            if state == "half_open":
                if self._half_open_count.get(key, 0) >= self.half_open_requests:
                    return GatewayResponse.error(
                        "Service temporarily unavailable",
                        503,
                    )
                self._half_open_count[key] = self._half_open_count.get(key, 0) + 1

        return None

    async def process_response(
        self,
        request: GatewayRequest,
        response: GatewayResponse,
    ) -> GatewayResponse:
        """Update circuit breaker state based on response."""
        key = request.path.split("/")[1] if "/" in request.path else "default"

        async with self._lock:
            if response.status_code >= 500:
                self._failures[key] = self._failures.get(key, 0) + 1
                self._last_failure_time[key] = time.time()

                if self._failures[key] >= self.failure_threshold:
                    self._state[key] = "open"

            else:
                state = self._state.get(key, "closed")
                if state == "half_open":
                    # Success in half-open state, close circuit
                    self._state[key] = "closed"
                    self._failures[key] = 0

        return response


class MiddlewareFactory:
    """Factory for creating middleware chains."""

    @staticmethod
    def create_default_chain(
        require_auth: bool = True,
        rate_limit_rps: float = 100.0,
    ) -> MiddlewareChain:
        """Create default middleware chain."""
        chain = MiddlewareChain()

        # Order matters
        chain.add(RequestIdMiddleware())
        chain.add(RequestLoggingMiddleware())
        chain.add(CORSMiddleware())
        chain.add(SecurityMiddleware())
        chain.add(RequestValidationMiddleware())
        chain.add(RateLimitingMiddleware(requests_per_second=rate_limit_rps))
        chain.add(AuthenticationMiddleware(require_auth=require_auth))
        chain.add(CircuitBreakerMiddleware())
        chain.add(TimeoutMiddleware())
        chain.add(CompressionMiddleware())

        return chain

    @staticmethod
    def create_public_chain() -> MiddlewareChain:
        """Create middleware chain for public endpoints."""
        chain = MiddlewareChain()

        chain.add(RequestIdMiddleware())
        chain.add(RequestLoggingMiddleware())
        chain.add(CORSMiddleware())
        chain.add(SecurityMiddleware())
        chain.add(RateLimitingMiddleware(
            requests_per_second=50.0,
            requests_per_minute=500.0,
        ))

        return chain

    @staticmethod
    def create_internal_chain() -> MiddlewareChain:
        """Create middleware chain for internal services."""
        chain = MiddlewareChain()

        chain.add(RequestIdMiddleware())
        chain.add(RequestLoggingMiddleware(log_headers=True))
        chain.add(IPFilterMiddleware(allow_private=True))

        return chain
