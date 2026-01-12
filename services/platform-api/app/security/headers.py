"""
Security Headers

HTTP security headers for web applications:
- Content Security Policy (CSP)
- CORS configuration
- XSS protection
- Frame options
- HSTS
"""

from typing import Optional, Dict, List, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class XFrameOptions(str, Enum):
    """X-Frame-Options values."""
    DENY = "DENY"
    SAMEORIGIN = "SAMEORIGIN"


class ReferrerPolicy(str, Enum):
    """Referrer-Policy values."""
    NO_REFERRER = "no-referrer"
    NO_REFERRER_DOWNGRADE = "no-referrer-when-downgrade"
    ORIGIN = "origin"
    ORIGIN_CROSS = "origin-when-cross-origin"
    SAME_ORIGIN = "same-origin"
    STRICT_ORIGIN = "strict-origin"
    STRICT_ORIGIN_CROSS = "strict-origin-when-cross-origin"
    UNSAFE_URL = "unsafe-url"


class PermissionsPolicy(str, Enum):
    """Permissions-Policy directives."""
    ACCELEROMETER = "accelerometer"
    AMBIENT_LIGHT = "ambient-light-sensor"
    AUTOPLAY = "autoplay"
    BATTERY = "battery"
    CAMERA = "camera"
    DISPLAY_CAPTURE = "display-capture"
    FULLSCREEN = "fullscreen"
    GEOLOCATION = "geolocation"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
    MICROPHONE = "microphone"
    MIDI = "midi"
    PAYMENT = "payment"
    PICTURE_IN_PICTURE = "picture-in-picture"
    USB = "usb"
    XR_SPATIAL = "xr-spatial-tracking"


@dataclass
class CSPDirective:
    """Content Security Policy directive."""
    name: str
    values: List[str] = field(default_factory=list)

    def to_string(self) -> str:
        """Convert to CSP string."""
        if not self.values:
            return ""
        return f"{self.name} {' '.join(self.values)}"


class CSPBuilder:
    """
    Content Security Policy builder.

    Fluent interface for building CSP headers.
    """

    def __init__(self):
        self._directives: Dict[str, CSPDirective] = {}
        self._report_uri: Optional[str] = None
        self._report_only: bool = False

    def _add_directive(self, name: str, *values: str) -> "CSPBuilder":
        """Add values to a directive."""
        if name not in self._directives:
            self._directives[name] = CSPDirective(name=name)
        self._directives[name].values.extend(values)
        return self

    def default_src(self, *sources: str) -> "CSPBuilder":
        """Set default-src directive."""
        return self._add_directive("default-src", *sources)

    def script_src(self, *sources: str) -> "CSPBuilder":
        """Set script-src directive."""
        return self._add_directive("script-src", *sources)

    def style_src(self, *sources: str) -> "CSPBuilder":
        """Set style-src directive."""
        return self._add_directive("style-src", *sources)

    def img_src(self, *sources: str) -> "CSPBuilder":
        """Set img-src directive."""
        return self._add_directive("img-src", *sources)

    def font_src(self, *sources: str) -> "CSPBuilder":
        """Set font-src directive."""
        return self._add_directive("font-src", *sources)

    def connect_src(self, *sources: str) -> "CSPBuilder":
        """Set connect-src directive."""
        return self._add_directive("connect-src", *sources)

    def media_src(self, *sources: str) -> "CSPBuilder":
        """Set media-src directive."""
        return self._add_directive("media-src", *sources)

    def object_src(self, *sources: str) -> "CSPBuilder":
        """Set object-src directive."""
        return self._add_directive("object-src", *sources)

    def frame_src(self, *sources: str) -> "CSPBuilder":
        """Set frame-src directive."""
        return self._add_directive("frame-src", *sources)

    def frame_ancestors(self, *sources: str) -> "CSPBuilder":
        """Set frame-ancestors directive."""
        return self._add_directive("frame-ancestors", *sources)

    def base_uri(self, *sources: str) -> "CSPBuilder":
        """Set base-uri directive."""
        return self._add_directive("base-uri", *sources)

    def form_action(self, *sources: str) -> "CSPBuilder":
        """Set form-action directive."""
        return self._add_directive("form-action", *sources)

    def worker_src(self, *sources: str) -> "CSPBuilder":
        """Set worker-src directive."""
        return self._add_directive("worker-src", *sources)

    def manifest_src(self, *sources: str) -> "CSPBuilder":
        """Set manifest-src directive."""
        return self._add_directive("manifest-src", *sources)

    def upgrade_insecure_requests(self) -> "CSPBuilder":
        """Add upgrade-insecure-requests directive."""
        return self._add_directive("upgrade-insecure-requests")

    def block_all_mixed_content(self) -> "CSPBuilder":
        """Add block-all-mixed-content directive."""
        return self._add_directive("block-all-mixed-content")

    def report_uri(self, uri: str) -> "CSPBuilder":
        """Set report-uri."""
        self._report_uri = uri
        return self

    def report_only(self, enabled: bool = True) -> "CSPBuilder":
        """Set report-only mode."""
        self._report_only = enabled
        return self

    def build(self) -> str:
        """Build the CSP header value."""
        parts = []
        for directive in self._directives.values():
            part = directive.to_string()
            if part:
                parts.append(part)

        if self._report_uri:
            parts.append(f"report-uri {self._report_uri}")

        return "; ".join(parts)

    def get_header_name(self) -> str:
        """Get the header name based on report-only setting."""
        if self._report_only:
            return "Content-Security-Policy-Report-Only"
        return "Content-Security-Policy"


@dataclass
class SecurityHeadersConfig:
    """Configuration for security headers."""
    # Content Security Policy
    csp_enabled: bool = True
    csp_builder: Optional[CSPBuilder] = None

    # X-Frame-Options
    x_frame_options: Optional[XFrameOptions] = XFrameOptions.DENY

    # X-Content-Type-Options
    x_content_type_options: bool = True  # nosniff

    # X-XSS-Protection
    x_xss_protection: bool = True

    # Strict-Transport-Security
    hsts_enabled: bool = True
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    hsts_preload: bool = False

    # Referrer-Policy
    referrer_policy: Optional[ReferrerPolicy] = ReferrerPolicy.STRICT_ORIGIN_CROSS

    # Permissions-Policy
    permissions_policy_enabled: bool = True
    permissions_policy_deny: List[PermissionsPolicy] = field(
        default_factory=lambda: [
            PermissionsPolicy.CAMERA,
            PermissionsPolicy.MICROPHONE,
            PermissionsPolicy.GEOLOCATION,
            PermissionsPolicy.PAYMENT,
        ]
    )

    # Cross-Origin policies
    cross_origin_embedder_policy: Optional[str] = None  # "require-corp"
    cross_origin_opener_policy: Optional[str] = None  # "same-origin"
    cross_origin_resource_policy: Optional[str] = None  # "same-origin"

    # Cache control for sensitive pages
    cache_control: Optional[str] = "no-store, no-cache, must-revalidate"

    # Additional custom headers
    custom_headers: Dict[str, str] = field(default_factory=dict)


class SecurityHeaders:
    """
    Security headers manager.

    Applies security headers to HTTP responses.
    """

    def __init__(self, config: Optional[SecurityHeadersConfig] = None):
        self.config = config or SecurityHeadersConfig()

        # Build default CSP if not provided
        if self.config.csp_enabled and not self.config.csp_builder:
            self.config.csp_builder = self._default_csp()

    def _default_csp(self) -> CSPBuilder:
        """Build default CSP."""
        return (
            CSPBuilder()
            .default_src("'self'")
            .script_src("'self'")
            .style_src("'self'", "'unsafe-inline'")
            .img_src("'self'", "data:", "https:")
            .font_src("'self'", "https:", "data:")
            .connect_src("'self'")
            .frame_ancestors("'none'")
            .base_uri("'self'")
            .form_action("'self'")
            .upgrade_insecure_requests()
        )

    def get_headers(self) -> Dict[str, str]:
        """Get all security headers."""
        headers = {}

        # Content Security Policy
        if self.config.csp_enabled and self.config.csp_builder:
            header_name = self.config.csp_builder.get_header_name()
            headers[header_name] = self.config.csp_builder.build()

        # X-Frame-Options
        if self.config.x_frame_options:
            headers["X-Frame-Options"] = self.config.x_frame_options.value

        # X-Content-Type-Options
        if self.config.x_content_type_options:
            headers["X-Content-Type-Options"] = "nosniff"

        # X-XSS-Protection
        if self.config.x_xss_protection:
            headers["X-XSS-Protection"] = "1; mode=block"

        # HSTS
        if self.config.hsts_enabled:
            hsts_value = f"max-age={self.config.hsts_max_age}"
            if self.config.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"
            if self.config.hsts_preload:
                hsts_value += "; preload"
            headers["Strict-Transport-Security"] = hsts_value

        # Referrer-Policy
        if self.config.referrer_policy:
            headers["Referrer-Policy"] = self.config.referrer_policy.value

        # Permissions-Policy
        if self.config.permissions_policy_enabled and self.config.permissions_policy_deny:
            policy_parts = [f"{p.value}=()" for p in self.config.permissions_policy_deny]
            headers["Permissions-Policy"] = ", ".join(policy_parts)

        # Cross-Origin policies
        if self.config.cross_origin_embedder_policy:
            headers["Cross-Origin-Embedder-Policy"] = self.config.cross_origin_embedder_policy
        if self.config.cross_origin_opener_policy:
            headers["Cross-Origin-Opener-Policy"] = self.config.cross_origin_opener_policy
        if self.config.cross_origin_resource_policy:
            headers["Cross-Origin-Resource-Policy"] = self.config.cross_origin_resource_policy

        # Cache-Control
        if self.config.cache_control:
            headers["Cache-Control"] = self.config.cache_control

        # Custom headers
        headers.update(self.config.custom_headers)

        return headers

    def apply(self, response: Response) -> Response:
        """Apply security headers to response."""
        for name, value in self.get_headers().items():
            response.headers[name] = value
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for security headers.

    Usage:
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)
    """

    def __init__(
        self,
        app,
        config: Optional[SecurityHeadersConfig] = None,
        exclude_paths: Optional[Set[str]] = None,
    ):
        super().__init__(app)
        self.security_headers = SecurityHeaders(config)
        self.exclude_paths = exclude_paths or set()

    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        response = await call_next(request)

        # Skip excluded paths
        if request.url.path not in self.exclude_paths:
            self.security_headers.apply(response)

        return response


def apply_security_headers(
    config: Optional[SecurityHeadersConfig] = None,
) -> callable:
    """
    Decorator for applying security headers to route.

    Usage:
        @app.get("/secure")
        @apply_security_headers()
        async def secure_endpoint():
            ...
    """
    security_headers = SecurityHeaders(config)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            response = await func(*args, **kwargs)

            # If response is a Response object, apply headers
            if isinstance(response, Response):
                security_headers.apply(response)

            return response
        return wrapper
    return decorator


# Preset configurations
def strict_csp() -> CSPBuilder:
    """Build a strict CSP for high-security applications."""
    return (
        CSPBuilder()
        .default_src("'none'")
        .script_src("'self'")
        .style_src("'self'")
        .img_src("'self'")
        .font_src("'self'")
        .connect_src("'self'")
        .frame_ancestors("'none'")
        .base_uri("'self'")
        .form_action("'self'")
        .upgrade_insecure_requests()
        .block_all_mixed_content()
    )


def api_security_config() -> SecurityHeadersConfig:
    """Security headers config optimized for APIs."""
    return SecurityHeadersConfig(
        csp_enabled=False,  # APIs don't need CSP
        x_frame_options=XFrameOptions.DENY,
        x_content_type_options=True,
        x_xss_protection=False,  # Not needed for JSON APIs
        hsts_enabled=True,
        referrer_policy=ReferrerPolicy.NO_REFERRER,
        permissions_policy_enabled=False,
        cache_control="no-store, no-cache, must-revalidate, private",
    )


def web_app_security_config() -> SecurityHeadersConfig:
    """Security headers config for web applications."""
    return SecurityHeadersConfig(
        csp_enabled=True,
        csp_builder=strict_csp(),
        x_frame_options=XFrameOptions.SAMEORIGIN,
        x_content_type_options=True,
        x_xss_protection=True,
        hsts_enabled=True,
        hsts_max_age=31536000,
        hsts_include_subdomains=True,
        referrer_policy=ReferrerPolicy.STRICT_ORIGIN_CROSS,
        permissions_policy_enabled=True,
        cross_origin_opener_policy="same-origin",
    )
