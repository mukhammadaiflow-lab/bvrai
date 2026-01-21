"""
FastAPI Application Module

This module provides the main FastAPI application setup with
all routes, middleware, and configuration.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel

from .base import (
    APIException,
    APIError,
    ErrorCode,
    generate_request_id,
)
from .auth import (
    APIAuthenticator,
    AuthContext,
    JWTConfig,
    Permission,
)
from .middleware import (
    RateLimiter,
    RateLimitConfig,
    RateLimitMiddleware,
    RequestLogger,
    LogConfig,
    RequestLogMiddleware,
    RequestTracker,
    RequestTrackingMiddleware,
)
from ..security.cors import CORSConfig, get_cors_middleware_config, validate_cors_config
from .routes import (
    agents_router,
    calls_router,
    knowledge_router,
    phone_numbers_router,
    campaigns_router,
    webhooks_router,
    analytics_router,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Application Configuration
# =============================================================================


class AppConfig(BaseModel):
    """Application configuration."""

    # API settings
    title: str = "Builder Engine API"
    description: str = "AI Voice Agent Platform API"
    version: str = "1.0.0"
    api_prefix: str = "/api/v1"

    # Server settings
    debug: bool = False
    docs_enabled: bool = True

    # CORS - Use secure environment-based configuration
    cors_config: Optional[CORSConfig] = None

    # Auth
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    rate_limit_per_day: int = 10000

    # Logging
    log_requests: bool = True
    log_format: str = "json"

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        # Load secure CORS configuration based on environment
        cors_config = CORSConfig.from_env()

        # Validate CORS config and log warnings
        cors_warnings = validate_cors_config(cors_config)
        for warning in cors_warnings:
            logger.warning(warning)

        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            jwt_secret=os.getenv("JWT_SECRET"),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
            rate_limit_per_day=int(os.getenv("RATE_LIMIT_PER_DAY", "10000")),
            cors_config=cors_config,
        )


# =============================================================================
# Health Check Models
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    timestamp: datetime
    checks: Dict[str, Any] = {}


# =============================================================================
# Exception Handlers
# =============================================================================


async def api_exception_handler(request: Request, exc: APIException):
    """Handle API exceptions."""
    request_id = getattr(request.state, "request_id", generate_request_id())

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "data": None,
            "error": exc.to_error(request_id).dict(),
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    request_id = getattr(request.state, "request_id", generate_request_id())

    # Map HTTP status to error code
    error_codes = {
        400: ErrorCode.VALIDATION_ERROR,
        401: ErrorCode.AUTHENTICATION_REQUIRED,
        403: ErrorCode.INSUFFICIENT_PERMISSIONS,
        404: ErrorCode.RESOURCE_NOT_FOUND,
        429: ErrorCode.RATE_LIMIT_EXCEEDED,
        500: ErrorCode.INTERNAL_ERROR,
    }

    error_code = error_codes.get(exc.status_code, ErrorCode.INTERNAL_ERROR)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "data": None,
            "error": {
                "code": error_code.value,
                "message": exc.detail,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
            },
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    request_id = getattr(request.state, "request_id", generate_request_id())

    logger.exception(f"Unhandled exception: {exc}")

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "data": None,
            "error": {
                "code": ErrorCode.INTERNAL_ERROR.value,
                "message": "An unexpected error occurred",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
            },
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# =============================================================================
# Application Factory
# =============================================================================


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        config: Application configuration

    Returns:
        Configured FastAPI application
    """
    config = config or AppConfig.from_env()

    # Lifespan context manager
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        logger.info("Starting Builder Engine API...")
        yield
        # Shutdown
        logger.info("Shutting down Builder Engine API...")

    # Create app
    app = FastAPI(
        title=config.title,
        description=config.description,
        version=config.version,
        docs_url="/docs" if config.docs_enabled else None,
        redoc_url="/redoc" if config.docs_enabled else None,
        openapi_url="/openapi.json" if config.docs_enabled else None,
        lifespan=lifespan,
    )

    # Store config
    app.state.config = config

    # ==========================================================================
    # Add Middleware (order matters - first added = outermost)
    # ==========================================================================

    # CORS - Use secure environment-based configuration
    cors_config = config.cors_config or CORSConfig.from_env()
    cors_middleware_config = get_cors_middleware_config(cors_config)
    app.add_middleware(
        CORSMiddleware,
        **cors_middleware_config,
    )

    # Request tracking
    tracker = RequestTracker(enable_tracing=True)
    app.add_middleware(
        RequestTrackingMiddleware,
        tracker=tracker,
    )

    # Request logging
    if config.log_requests:
        log_config = LogConfig(format=config.log_format)
        request_logger = RequestLogger(log_config)
        app.add_middleware(
            RequestLogMiddleware,
            logger=request_logger,
        )

    # Rate limiting
    if config.rate_limit_enabled:
        rate_config = RateLimitConfig(
            default_requests_per_minute=config.rate_limit_per_minute,
            default_requests_per_day=config.rate_limit_per_day,
        )
        rate_limiter = RateLimiter(rate_config)
        app.add_middleware(
            RateLimitMiddleware,
            limiter=rate_limiter,
        )

    # ==========================================================================
    # Add Exception Handlers
    # ==========================================================================

    app.add_exception_handler(APIException, api_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # ==========================================================================
    # Add Routes
    # ==========================================================================

    # Health check (no prefix)
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version=config.version,
            timestamp=datetime.utcnow(),
            checks={
                "api": "ok",
            },
        )

    @app.get("/ready", tags=["Health"])
    async def readiness_check():
        """Readiness check for Kubernetes."""
        # In production, check database, cache, etc.
        return {"status": "ready"}

    # API routes
    app.include_router(agents_router, prefix=config.api_prefix)
    app.include_router(calls_router, prefix=config.api_prefix)
    app.include_router(knowledge_router, prefix=config.api_prefix)
    app.include_router(phone_numbers_router, prefix=config.api_prefix)
    app.include_router(campaigns_router, prefix=config.api_prefix)
    app.include_router(webhooks_router, prefix=config.api_prefix)
    app.include_router(analytics_router, prefix=config.api_prefix)

    # ==========================================================================
    # Custom OpenAPI Schema
    # ==========================================================================

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=config.title,
            version=config.version,
            description=config.description,
            routes=app.routes,
        )

        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for authentication",
            },
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT bearer token",
            },
        }

        # Apply security globally
        openapi_schema["security"] = [
            {"ApiKeyAuth": []},
            {"BearerAuth": []},
        ]

        # Add tags metadata
        openapi_schema["tags"] = [
            {
                "name": "Agents",
                "description": "Manage AI voice agents",
            },
            {
                "name": "Calls",
                "description": "Manage voice calls",
            },
            {
                "name": "Knowledge Bases",
                "description": "Manage knowledge bases and documents",
            },
            {
                "name": "Phone Numbers",
                "description": "Manage phone numbers",
            },
            {
                "name": "Campaigns",
                "description": "Manage outbound campaigns",
            },
            {
                "name": "Webhooks",
                "description": "Manage webhook endpoints",
            },
            {
                "name": "Analytics",
                "description": "Analytics and reporting",
            },
            {
                "name": "Health",
                "description": "Health and readiness checks",
            },
        ]

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    return app


# =============================================================================
# Default App Instance
# =============================================================================


# Create default app instance
app = create_app()


# =============================================================================
# Entry Point
# =============================================================================


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
):
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "platform.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    run_server(reload=True)
