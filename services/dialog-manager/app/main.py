"""
Dialog Manager Service - Main FastAPI Application.

Provides RAG-based dialog processing for voice AI agents.
"""
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.adapters import create_llm_adapter, create_vector_adapter
from app.config import get_settings
from app.models import DialogTurnRequest, DialogTurnResponse, ErrorResponse, HealthResponse
from app.services import DialogService, SessionService

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Global service instances
settings = get_settings()
session_service = SessionService()
llm_adapter = create_llm_adapter()
vector_adapter = create_vector_adapter()
dialog_service = DialogService(
    llm_adapter=llm_adapter,
    vector_adapter=vector_adapter,
    session_service=session_service,
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("starting_dialog_manager", port=settings.port, env=settings.environment)
    await session_service.start()

    yield

    # Shutdown
    logger.info("shutting_down_dialog_manager")
    await session_service.stop()


# Create FastAPI app
app = FastAPI(
    title="Dialog Manager Service",
    description="RAG-based dialog processing for Builder Engine voice AI agents",
    version="1.0.0",
    lifespan=lifespan,
)

# Add rate limiter
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Handle rate limit exceeded errors."""
    return JSONResponse(
        status_code=429,
        content={
            "error": "Too many requests",
            "code": "RATE_LIMIT_EXCEEDED",
        },
    )


# CORS middleware
cors_origins = (
    ["*"]
    if settings.cors_origins == "*"
    else [o.strip() for o in settings.cors_origins.split(",")]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="dialog-manager",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/ready", response_model=HealthResponse, tags=["Health"])
async def readiness_check() -> HealthResponse:
    """Readiness check endpoint."""
    # Check if LLM adapter is available
    llm_available = await llm_adapter.is_available()

    if not llm_available:
        raise HTTPException(status_code=503, detail="LLM adapter not available")

    return HealthResponse(
        status="healthy",
        service="dialog-manager",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post(
    "/dialog/turn",
    response_model=DialogTurnResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["Dialog"],
    summary="Process a dialog turn",
    description="""
Process a user transcript and generate an AI response.

The service:
1. Retrieves relevant context from the vector database
2. Builds a prompt with system message, history, and context
3. Generates a response using the LLM
4. Extracts any actions from the response
5. Returns speak_text for TTS and optional action_object for automations
""",
)
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window_seconds}seconds")
async def process_dialog_turn(
    request: Request,
    body: DialogTurnRequest,
) -> DialogTurnResponse:
    """
    Process a dialog turn and return AI response.

    Args:
        body: Dialog turn request with tenant_id, session_id, transcript

    Returns:
        DialogTurnResponse with speak_text and optional action_object
    """
    try:
        response = await dialog_service.process_turn(
            tenant_id=body.tenant_id,
            session_id=body.session_id,
            transcript=body.transcript,
            is_final=body.is_final,
        )
        return response

    except Exception as e:
        logger.exception(
            "dialog_turn_error",
            tenant_id=body.tenant_id,
            session_id=body.session_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="Failed to process dialog turn",
                code="PROCESSING_ERROR",
                details={"message": str(e)} if settings.debug else None,
            ).model_dump(),
        )


@app.get("/sessions/{session_id}", tags=["Sessions"])
async def get_session(session_id: str) -> dict[str, Any]:
    """
    Get session information.

    Args:
        session_id: Session identifier

    Returns:
        Session details including history
    """
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session.session_id,
        "tenant_id": session.tenant_id,
        "created_at": session.created_at.isoformat(),
        "last_activity": session.last_activity.isoformat(),
        "turn_count": len(session.history),
        "history": [
            {
                "role": turn.role,
                "content": turn.content[:100] + "..." if len(turn.content) > 100 else turn.content,
                "timestamp": turn.timestamp.isoformat(),
            }
            for turn in session.history
        ],
    }


@app.delete("/sessions/{session_id}", tags=["Sessions"])
async def delete_session(session_id: str) -> dict[str, str]:
    """
    Delete a session.

    Args:
        session_id: Session identifier

    Returns:
        Deletion confirmation
    """
    deleted = session_service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "deleted", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
    )
