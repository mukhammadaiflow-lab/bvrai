"""
Streaming Orchestrator Service - FastAPI Application.

Ultra-low latency voice AI streaming service targeting <300ms
end-to-end latency from speech end to first audio byte.
"""

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional, Any

import structlog
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Query,
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from .config import get_settings, Settings
from .models import (
    CreateSessionRequest,
    SessionResponse,
    HealthResponse,
    SessionState,
    AudioFormat,
    WSMessageType,
    StreamEvent,
    EventType,
)
from .pipeline import StreamingOrchestrator, LatencyTracker
from .pipeline.circuit_breaker import default_registry as circuit_registry


# =============================================================================
# Logging Configuration
# =============================================================================

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


# =============================================================================
# Application State
# =============================================================================

class AppState:
    """Application state container."""

    def __init__(self):
        self.sessions: Dict[str, StreamingOrchestrator] = {}
        self.latency_tracker = LatencyTracker()
        self.start_time = time.time()
        self.settings: Optional[Settings] = None

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def active_session_count(self) -> int:
        return len(self.sessions)


app_state = AppState()


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    settings = get_settings()
    app_state.settings = settings

    logger.info(
        "streaming_orchestrator_starting",
        port=settings.port,
        latency_profile=settings.latency_profile.value,
        default_asr=settings.default_asr_provider.value,
        default_llm=settings.default_llm_provider.value,
        default_tts=settings.default_tts_provider.value,
    )

    yield

    # Shutdown - stop all sessions
    logger.info("streaming_orchestrator_stopping")

    for session_id, orchestrator in list(app_state.sessions.items()):
        try:
            await orchestrator.stop()
        except Exception as e:
            logger.error(f"Error stopping session {session_id}: {e}")

    app_state.sessions.clear()


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Streaming Orchestrator Service",
    description="""
    Ultra-low latency voice AI streaming service.

    Features:
    - Sub-300ms end-to-end latency (speech end to first audio)
    - Parallel streaming ASR → LLM → TTS pipeline
    - Speculative LLM execution on stable partial transcripts
    - Adaptive audio buffering with jitter compensation
    - Circuit breaker patterns for provider resilience
    - Real-time latency monitoring and optimization

    Target latency breakdown:
    - ASR Processing: 50-100ms
    - LLM First Token: 100-150ms
    - TTS First Audio: 50-75ms
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Dependencies
# =============================================================================

def get_app_settings() -> Settings:
    """Dependency for getting settings."""
    if app_state.settings is None:
        return get_settings()
    return app_state.settings


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_app_settings)):
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="streaming-orchestrator",
        version="1.0.0",
        uptime_seconds=app_state.uptime_seconds,
        active_sessions=app_state.active_session_count,
        providers={
            "asr": settings.default_asr_provider.value,
            "llm": settings.default_llm_provider.value,
            "tts": settings.default_tts_provider.value,
        },
    )


@app.get("/ready")
async def readiness_check():
    """Readiness probe for Kubernetes."""
    if app_state.settings is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}


@app.get("/info")
async def service_info(settings: Settings = Depends(get_app_settings)):
    """Get detailed service information."""
    return {
        "service": "streaming-orchestrator",
        "version": "1.0.0",
        "uptime_seconds": round(app_state.uptime_seconds, 2),
        "active_sessions": app_state.active_session_count,
        "configuration": {
            "latency_profile": settings.latency_profile.value,
            "streaming_mode": settings.streaming_mode.value,
            "max_concurrent_sessions": settings.max_concurrent_sessions,
            "latency_targets": {
                "e2e_target_ms": settings.latency_targets.e2e_target_ms,
                "asr_target_ms": settings.latency_targets.asr_target_ms,
                "llm_first_token_target_ms": settings.latency_targets.llm_first_token_target_ms,
                "tts_first_audio_target_ms": settings.latency_targets.tts_first_audio_target_ms,
            },
            "speculative_execution": {
                "enabled": settings.speculative_execution.enabled,
                "min_confidence": settings.speculative_execution.min_confidence,
                "min_words": settings.speculative_execution.min_words,
            },
        },
        "providers": {
            "asr": {
                "default": settings.default_asr_provider.value,
                "fallback": settings.fallback_asr_provider.value if settings.fallback_asr_provider else None,
            },
            "llm": {
                "default": settings.default_llm_provider.value,
                "model": settings.default_llm_model,
                "fallback": settings.fallback_llm_provider.value if settings.fallback_llm_provider else None,
            },
            "tts": {
                "default": settings.default_tts_provider.value,
                "fallback": settings.fallback_tts_provider.value if settings.fallback_tts_provider else None,
            },
        },
    }


# =============================================================================
# Session Management
# =============================================================================

@app.post("/v1/sessions", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    settings: Settings = Depends(get_app_settings),
):
    """
    Create a new streaming session.

    This initializes the ultra-low latency pipeline with the specified
    configuration and returns connection details.
    """
    # Check limits
    if app_state.active_session_count >= settings.max_concurrent_sessions:
        raise HTTPException(
            status_code=429,
            detail=f"Maximum concurrent sessions ({settings.max_concurrent_sessions}) reached",
        )

    # Create orchestrator configuration
    from .pipeline.orchestrator import OrchestratorConfig

    config = OrchestratorConfig(
        tenant_id=request.tenant_id,
        agent_id=request.agent_id,
        input_format=request.input_format,
        output_format=request.output_format,
        asr_provider=request.asr_provider or settings.default_asr_provider.value,
        llm_provider=request.llm_provider or settings.default_llm_provider.value,
        llm_model=request.llm_model or settings.default_llm_model,
        tts_provider=request.tts_provider or settings.default_tts_provider.value,
        tts_voice=request.tts_voice or settings.default_tts_voice,
        enable_speculative_execution=request.enable_speculative_execution,
        enable_interruption=request.enable_interruption,
        system_prompt=request.system_prompt or "",
    )

    # Create orchestrator
    orchestrator = StreamingOrchestrator(config, settings)
    session_id = config.session_id

    # Store session
    app_state.sessions[session_id] = orchestrator

    logger.info(
        "session_created",
        session_id=session_id,
        tenant_id=request.tenant_id,
        agent_id=request.agent_id,
    )

    return SessionResponse(
        session_id=session_id,
        state=SessionState.CREATED,
        websocket_url=f"/v1/sessions/{session_id}/ws",
        created_at=orchestrator._current_turn.started_at if orchestrator._current_turn else time.time(),
        config=request,
    )


@app.get("/v1/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details and statistics."""
    orchestrator = app_state.sessions.get(session_id)
    if not orchestrator:
        raise HTTPException(status_code=404, detail="Session not found")

    return orchestrator.get_statistics()


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """Stop and delete a session."""
    orchestrator = app_state.sessions.get(session_id)
    if not orchestrator:
        raise HTTPException(status_code=404, detail="Session not found")

    await orchestrator.stop()
    del app_state.sessions[session_id]

    logger.info("session_deleted", session_id=session_id)

    return {"status": "deleted", "session_id": session_id}


@app.get("/v1/sessions")
async def list_sessions(
    state: Optional[str] = None,
    tenant_id: Optional[str] = None,
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List all active sessions."""
    sessions = []

    for session_id, orchestrator in app_state.sessions.items():
        # Filter by state
        if state and orchestrator.state.value != state:
            continue

        # Filter by tenant
        if tenant_id and orchestrator.config.tenant_id != tenant_id:
            continue

        sessions.append({
            "session_id": session_id,
            "state": orchestrator.state.value,
            "tenant_id": orchestrator.config.tenant_id,
            "agent_id": orchestrator.config.agent_id,
        })

    # Paginate
    total = len(sessions)
    sessions = sessions[offset:offset + limit]

    return {
        "sessions": sessions,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


# =============================================================================
# WebSocket Streaming
# =============================================================================

@app.websocket("/v1/sessions/{session_id}/ws")
async def session_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time audio streaming.

    Protocol:

    Client → Server:
    - {"type": "audio", "audio_base64": "...", "sequence": N}
    - {"type": "control", "action": "interrupt|pause|resume"}
    - {"type": "ping"}

    Server → Client:
    - {"type": "transcript", "text": "...", "is_final": bool}
    - {"type": "response", "text": "...", "is_complete": bool}
    - {"type": "audio_out", "audio_base64": "...", "sequence": N}
    - {"type": "event", "event_type": "...", "data": {...}}
    - {"type": "error", "message": "..."}
    - {"type": "pong"}
    """
    orchestrator = app_state.sessions.get(session_id)
    if not orchestrator:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()

    logger.info("websocket_connected", session_id=session_id)

    # Event handler for sending events to client
    async def event_handler(event: StreamEvent):
        try:
            await websocket.send_json({
                "type": "event",
                "event_type": event.event_type.value,
                "data": event.data,
                "timestamp": event.timestamp,
            })
        except Exception as e:
            logger.debug(f"Failed to send event: {e}")

    orchestrator.add_event_handler(event_handler)

    # Start orchestrator
    try:
        await orchestrator.start()
    except Exception as e:
        logger.error(f"Failed to start orchestrator: {e}")
        await websocket.close(code=1011, reason=str(e))
        return

    # Output streaming task
    async def output_streamer():
        try:
            sequence = 0
            async for chunk in orchestrator.get_output_audio():
                audio_b64 = base64.b64encode(chunk.data).decode("utf-8")
                await websocket.send_json({
                    "type": "audio_out",
                    "audio_base64": audio_b64,
                    "duration_ms": chunk.duration_ms,
                    "sequence": sequence,
                })
                sequence += 1
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Output streamer error: {e}")

    output_task = asyncio.create_task(output_streamer())

    try:
        while True:
            message = await websocket.receive_json()
            msg_type = message.get("type")

            if msg_type == "audio":
                # Process incoming audio
                audio_b64 = message.get("audio_base64", "")
                if audio_b64:
                    audio_data = base64.b64decode(audio_b64)
                    timestamp = message.get("timestamp", time.time() * 1000)
                    await orchestrator.process_audio(audio_data, timestamp)

            elif msg_type == "control":
                action = message.get("action")
                if action == "interrupt":
                    await orchestrator.handle_interruption()
                elif action == "pause":
                    # Implement pause logic
                    pass
                elif action == "resume":
                    # Implement resume logic
                    pass

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("websocket_disconnected", session_id=session_id)

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except Exception:
            pass

    finally:
        output_task.cancel()
        try:
            await output_task
        except asyncio.CancelledError:
            pass

        await orchestrator.stop()


# =============================================================================
# Metrics & Monitoring
# =============================================================================

@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    lines = []

    # Service metrics
    lines.append("# HELP streaming_orchestrator_sessions_active Active sessions")
    lines.append("# TYPE streaming_orchestrator_sessions_active gauge")
    lines.append(f"streaming_orchestrator_sessions_active {app_state.active_session_count}")
    lines.append("")

    lines.append("# HELP streaming_orchestrator_uptime_seconds Service uptime")
    lines.append("# TYPE streaming_orchestrator_uptime_seconds gauge")
    lines.append(f"streaming_orchestrator_uptime_seconds {app_state.uptime_seconds:.2f}")
    lines.append("")

    # Latency metrics
    lines.append(app_state.latency_tracker.get_prometheus_metrics())

    # Circuit breaker metrics
    lines.append(circuit_registry.get_prometheus_metrics())

    # Per-session metrics
    for session_id, orchestrator in app_state.sessions.items():
        stats = orchestrator.get_statistics()
        metrics = stats.get("metrics", {})

        lines.append(f'# Session: {session_id}')
        lines.append(
            f'streaming_session_e2e_latency_avg_ms{{session="{session_id}"}} '
            f'{metrics.get("avg_e2e_latency_ms", 0):.2f}'
        )
        lines.append(
            f'streaming_session_turns_total{{session="{session_id}"}} '
            f'{metrics.get("total_turns", 0)}'
        )
        lines.append("")

    return "\n".join(lines)


@app.get("/v1/latency")
async def get_latency_summary():
    """Get detailed latency statistics."""
    return await app_state.latency_tracker.get_summary()


@app.get("/v1/latency/recommendations")
async def get_latency_recommendations():
    """Get optimization recommendations based on current latency."""
    return {
        "recommendations": app_state.latency_tracker.get_optimization_recommendations(),
    }


@app.get("/v1/circuits")
async def get_circuit_breakers():
    """Get circuit breaker status for all providers."""
    return await circuit_registry.get_all_stats()


# =============================================================================
# Admin Endpoints
# =============================================================================

@app.post("/v1/admin/reset-metrics")
async def reset_metrics():
    """Reset all latency metrics."""
    await app_state.latency_tracker.reset()
    return {"status": "reset"}


@app.post("/v1/admin/reset-circuits")
async def reset_circuits():
    """Reset all circuit breakers."""
    await circuit_registry.reset_all()
    return {"status": "reset"}


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=settings.debug,
    )
