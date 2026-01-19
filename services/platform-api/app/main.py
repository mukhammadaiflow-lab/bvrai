"""Platform API - Main application entry point.

This is the central API for the Builder Engine voice AI platform.
Provides endpoints for managing agents, calls, knowledge bases, and analytics.
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import get_settings
from app.database.session import init_db, close_db

# Import routers
from app.agents.routes import router as agents_router
from app.calls.routes import router as calls_router
from app.analytics.routes import router as analytics_router
from app.knowledge.routes import router as knowledge_router
from app.webhooks.routes import router as webhooks_router

# Import new API v1 routes
from app.api.v1 import router as api_v1_router

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(
        "Starting Platform API",
        host=settings.host,
        port=settings.port,
    )

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    yield

    # Cleanup
    logger.info("Shutting down Platform API")
    await close_db()


# Create FastAPI application
app = FastAPI(
    title="Builder Engine Platform API",
    description="""
    Central API for the Builder Engine voice AI platform.

    ## Features

    - **Agents**: Create and manage voice AI agents with custom prompts, voices, and tools
    - **Calls**: Monitor live calls, view transcripts, and access call recordings
    - **Knowledge Base**: Upload documents and search with RAG
    - **Analytics**: Real-time dashboards and usage metrics

    ## Authentication

    All endpoints require an API key. Include it in the `X-API-Key` header.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path,
        method=request.method,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# Health check
class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="platform-api",
        version="1.0.0",
    )


@app.get("/", tags=["health"])
async def root():
    """Root endpoint."""
    return {
        "service": "Builder Engine Platform API",
        "version": "1.0.0",
        "docs": "/docs",
    }


# Register routers
app.include_router(agents_router, prefix="/api/v1")
app.include_router(calls_router, prefix="/api/v1")
app.include_router(analytics_router, prefix="/api/v1")
app.include_router(knowledge_router, prefix="/api/v1")
app.include_router(webhooks_router, prefix="/api/v1")

# Register new comprehensive API v1 routes
app.include_router(api_v1_router, prefix="/api")


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
