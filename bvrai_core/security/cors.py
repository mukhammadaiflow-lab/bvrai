"""
CORS Configuration

This module provides secure CORS configuration for the API,
with environment-based origin whitelisting.
"""

import os
import re
from typing import List, Optional, Set

from pydantic import BaseModel, Field


# =============================================================================
# Configuration
# =============================================================================


class CORSConfig(BaseModel):
    """CORS configuration."""

    # Allowed origins
    allow_origins: List[str] = Field(default_factory=list)
    allow_origin_regex: Optional[str] = None

    # Allowed methods
    allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
    )

    # Allowed headers
    allow_headers: List[str] = Field(
        default=[
            "Accept",
            "Accept-Language",
            "Authorization",
            "Content-Type",
            "Content-Language",
            "X-API-Key",
            "X-Request-ID",
            "X-Correlation-ID",
        ]
    )

    # Exposed headers (accessible to JavaScript)
    expose_headers: List[str] = Field(
        default=[
            "X-Request-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ]
    )

    # Credentials
    allow_credentials: bool = True

    # Preflight cache
    max_age: int = 600  # 10 minutes

    @classmethod
    def from_env(cls) -> "CORSConfig":
        """
        Create CORS config from environment variables.

        Environment variables:
        - CORS_ORIGINS: Comma-separated list of allowed origins
        - CORS_ORIGIN_REGEX: Regex pattern for allowed origins
        - ENVIRONMENT: 'development', 'staging', 'production'
        """
        environment = os.getenv("ENVIRONMENT", "development")
        origins_str = os.getenv("CORS_ORIGINS", "")
        origin_regex = os.getenv("CORS_ORIGIN_REGEX")

        # Parse origins
        if origins_str:
            origins = [o.strip() for o in origins_str.split(",") if o.strip()]
        else:
            # Default origins based on environment
            if environment == "development":
                origins = [
                    "http://localhost:3000",
                    "http://localhost:3001",
                    "http://localhost:8080",
                    "http://127.0.0.1:3000",
                    "http://127.0.0.1:3001",
                ]
            elif environment == "staging":
                origins = [
                    "https://staging.buildervoice.ai",
                    "https://*.staging.buildervoice.ai",
                ]
            else:  # production
                origins = [
                    "https://buildervoice.ai",
                    "https://www.buildervoice.ai",
                    "https://app.buildervoice.ai",
                    "https://dashboard.buildervoice.ai",
                ]

        return cls(
            allow_origins=origins,
            allow_origin_regex=origin_regex,
        )

    def is_allowed_origin(self, origin: str) -> bool:
        """
        Check if an origin is allowed.

        Args:
            origin: The origin to check

        Returns:
            True if the origin is allowed
        """
        if not origin:
            return False

        # Check exact match
        if origin in self.allow_origins:
            return True

        # Check wildcard patterns
        for allowed in self.allow_origins:
            if "*" in allowed:
                # Convert wildcard to regex
                pattern = allowed.replace(".", r"\.").replace("*", r"[^.]+")
                if re.match(f"^{pattern}$", origin):
                    return True

        # Check regex pattern
        if self.allow_origin_regex:
            if re.match(self.allow_origin_regex, origin):
                return True

        return False


# =============================================================================
# Middleware Configuration Helper
# =============================================================================


def get_cors_middleware_config(config: Optional[CORSConfig] = None) -> dict:
    """
    Get configuration dict for FastAPI CORSMiddleware.

    Args:
        config: CORS configuration (uses env if not provided)

    Returns:
        Dict suitable for CORSMiddleware
    """
    if config is None:
        config = CORSConfig.from_env()

    # For CORSMiddleware, we need to handle wildcards differently
    origins = []
    for origin in config.allow_origins:
        if origin == "*":
            # Wildcard - allow all (NOT RECOMMENDED for production)
            return {
                "allow_origins": ["*"],
                "allow_credentials": False,  # Can't use credentials with *
                "allow_methods": config.allow_methods,
                "allow_headers": config.allow_headers,
                "expose_headers": config.expose_headers,
                "max_age": config.max_age,
            }
        origins.append(origin)

    return {
        "allow_origins": origins,
        "allow_origin_regex": config.allow_origin_regex,
        "allow_credentials": config.allow_credentials,
        "allow_methods": config.allow_methods,
        "allow_headers": config.allow_headers,
        "expose_headers": config.expose_headers,
        "max_age": config.max_age,
    }


# =============================================================================
# Security Best Practices
# =============================================================================


def validate_cors_config(config: CORSConfig) -> List[str]:
    """
    Validate CORS configuration for security issues.

    Args:
        config: The CORS configuration to validate

    Returns:
        List of warning messages
    """
    warnings = []

    # Check for wildcard origins
    if "*" in config.allow_origins:
        warnings.append(
            "SECURITY WARNING: Using wildcard origin (*) allows any site to access your API. "
            "This is NOT recommended for production."
        )

    # Check credentials with wildcard
    if "*" in config.allow_origins and config.allow_credentials:
        warnings.append(
            "SECURITY ERROR: Cannot use allow_credentials=True with wildcard origin. "
            "This configuration will fail."
        )

    # Check for overly permissive origins
    for origin in config.allow_origins:
        if "localhost" in origin or "127.0.0.1" in origin:
            environment = os.getenv("ENVIRONMENT", "development")
            if environment == "production":
                warnings.append(
                    f"WARNING: Localhost origin ({origin}) allowed in production. "
                    "This may be a security risk."
                )

    # Check for HTTP in production
    environment = os.getenv("ENVIRONMENT", "development")
    if environment == "production":
        for origin in config.allow_origins:
            if origin.startswith("http://") and origin != "http://localhost":
                warnings.append(
                    f"SECURITY WARNING: HTTP origin ({origin}) in production. "
                    "Use HTTPS instead."
                )

    return warnings


# =============================================================================
# Preset Configurations
# =============================================================================


def get_development_cors() -> CORSConfig:
    """Get permissive CORS config for development."""
    return CORSConfig(
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
        ],
        allow_credentials=True,
    )


def get_staging_cors() -> CORSConfig:
    """Get CORS config for staging environment."""
    return CORSConfig(
        allow_origins=[
            "https://staging.buildervoice.ai",
        ],
        allow_origin_regex=r"https://[a-z0-9-]+\.staging\.buildervoice\.ai",
        allow_credentials=True,
    )


def get_production_cors() -> CORSConfig:
    """Get strict CORS config for production."""
    return CORSConfig(
        allow_origins=[
            "https://buildervoice.ai",
            "https://www.buildervoice.ai",
            "https://app.buildervoice.ai",
            "https://dashboard.buildervoice.ai",
        ],
        allow_credentials=True,
    )


def get_cors_for_environment(environment: Optional[str] = None) -> CORSConfig:
    """
    Get CORS config for the specified environment.

    Args:
        environment: Environment name (uses ENVIRONMENT env var if not provided)

    Returns:
        Appropriate CORSConfig for the environment
    """
    env = environment or os.getenv("ENVIRONMENT", "development")

    if env == "production":
        return get_production_cors()
    elif env == "staging":
        return get_staging_cors()
    else:
        return get_development_cors()


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "CORSConfig",
    "get_cors_middleware_config",
    "validate_cors_config",
    "get_development_cors",
    "get_staging_cors",
    "get_production_cors",
    "get_cors_for_environment",
]
