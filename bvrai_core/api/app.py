"""
FastAPI Application Module

This module provides the main FastAPI application setup with
all routes, middleware, and configuration.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from fastapi import Depends, FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

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
from .routes import (
    agents_router,
    calls_router,
    knowledge_router,
    phone_numbers_router,
    campaigns_router,
    webhooks_router,
    analytics_router,
)

# Database imports
from ..database.base import (
    init_database,
    close_database,
    get_database,
    get_session,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Database Dependency
# =============================================================================


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.

    Usage in routes:
        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db_session)):
            repo = ItemRepository(db)
            return await repo.get_all()
    """
    async with get_session() as session:
        yield session


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

    # Database
    database_url: str = "sqlite+aiosqlite:///./bvrai.db"
    database_pool_size: int = 5
    database_max_overflow: int = 10

    # CORS - SECURITY: Never use "*" with credentials=True
    cors_origins: list = [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://app.bvrai.com",
        "https://dashboard.bvrai.com",
    ]
    cors_allow_credentials: bool = True

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

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            database_url=os.getenv(
                "DATABASE_URL",
                "sqlite+aiosqlite:///./bvrai.db"
            ),
            database_pool_size=int(os.getenv("DATABASE_POOL_SIZE", "5")),
            database_max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "10")),
            jwt_secret=os.getenv("JWT_SECRET"),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
            rate_limit_per_day=int(os.getenv("RATE_LIMIT_PER_DAY", "10000")),
            cors_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(","),
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

        # Initialize database
        logger.info(f"Initializing database connection...")
        db = init_database(
            database_url=config.database_url,
            pool_size=config.database_pool_size,
            max_overflow=config.database_max_overflow,
            echo=config.debug,
        )

        # Create tables if they don't exist (for development)
        if config.debug or "sqlite" in config.database_url:
            logger.info("Creating database tables...")
            await db.create_all()

        # Verify database connectivity
        if await db.health_check():
            logger.info("Database connection established successfully")
        else:
            logger.error("Database connection failed!")

        yield

        # Shutdown
        logger.info("Shutting down Builder Engine API...")
        await close_database()
        logger.info("Database connection closed")

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

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=config.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
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
        # Check database connectivity
        try:
            db = get_database()
            db_healthy = await db.health_check()
        except Exception:
            db_healthy = False

        return HealthResponse(
            status="healthy" if db_healthy else "degraded",
            version=config.version,
            timestamp=datetime.utcnow(),
            checks={
                "api": "ok",
                "database": "ok" if db_healthy else "error",
            },
        )

    @app.get("/ready", tags=["Health"])
    async def readiness_check():
        """Readiness check for Kubernetes."""
        try:
            db = get_database()
            db_healthy = await db.health_check()
            if not db_healthy:
                return JSONResponse(
                    status_code=503,
                    content={"status": "not_ready", "reason": "database_unavailable"}
                )
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": str(e)}
            )
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
        "bvrai_core.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    run_server(reload=True)
