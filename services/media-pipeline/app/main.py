"""Media Pipeline Service - Main FastAPI application."""

import asyncio
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

from app.config import settings
from app.core.manager import SessionManager
from app.core.session import SessionConfig, SessionState
from app.streams.websocket import WebSocketStream, WebSocketConfig


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

# Session manager
session_manager: Optional[SessionManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global session_manager

    logger.info("media_pipeline_starting", port=settings.port)

    # Initialize session manager
    session_manager = SessionManager()
    await session_manager.start()

    yield

    # Shutdown
    logger.info("media_pipeline_stopping")
    if session_manager:
        await session_manager.stop()


# Create FastAPI app
app = FastAPI(
    title="Media Pipeline Service",
    description="Real-time audio processing for Builder Engine voice AI",
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
# Health & Info Endpoints
# =============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "media-pipeline",
        "version": "1.0.0",
    }


@app.get("/info")
async def service_info():
    """Service information."""
    metrics = session_manager.get_metrics() if session_manager else {}
    return {
        "service": "media-pipeline",
        "version": "1.0.0",
        "settings": {
            "sample_rate": settings.sample_rate,
            "sample_rate_webrtc": settings.sample_rate_webrtc,
            "default_codec": settings.default_codec,
            "max_concurrent_sessions": settings.max_concurrent_sessions,
        },
        "metrics": metrics,
    }


# =============================================================================
# Session Management Endpoints
# =============================================================================


class CreateSessionRequest(BaseModel):
    """Create session request."""
    call_id: str
    agent_id: str
    source: str = "telephony"  # telephony, webrtc
    direction: str = "inbound"
    codec: str = "pcmu"
    sample_rate: int = 8000
    caller_number: Optional[str] = None
    callee_number: Optional[str] = None
    metadata: Dict[str, Any] = {}


class SessionResponse(BaseModel):
    """Session response."""
    session_id: str
    call_id: str
    agent_id: str
    state: str
    websocket_url: str


@app.post("/v1/sessions", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new media session."""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Service not ready")

    config = SessionConfig(
        call_id=request.call_id,
        agent_id=request.agent_id,
        source=request.source,
        direction=request.direction,
        codec=request.codec,
        sample_rate=request.sample_rate,
        caller_number=request.caller_number,
        callee_number=request.callee_number,
        metadata=request.metadata,
    )

    session = await session_manager.create_session(config)

    return SessionResponse(
        session_id=session.id,
        call_id=session.call_id,
        agent_id=session.agent_id,
        state=session.state.value,
        websocket_url=f"/v1/sessions/{session.id}/ws",
    )


@app.get("/v1/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details."""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Service not ready")

    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session.get_stats()


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """End and delete a session."""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Service not ready")

    success = await session_manager.disconnect_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "disconnected", "session_id": session_id}


@app.get("/v1/sessions")
async def list_sessions(
    state: Optional[str] = None,
    agent_id: Optional[str] = None,
):
    """List all sessions."""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Service not ready")

    state_enum = SessionState(state) if state else None
    sessions = await session_manager.list_sessions(state=state_enum, agent_id=agent_id)

    return {"sessions": sessions}


# =============================================================================
# WebSocket Endpoints
# =============================================================================


@app.websocket("/v1/sessions/{session_id}/ws")
async def session_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for audio streaming."""
    if not session_manager:
        await websocket.close(code=1011, reason="Service not ready")
        return

    session = await session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=1008, reason="Session not found")
        return

    # Create WebSocket stream
    ws_config = WebSocketConfig(
        session_id=session_id,
        audio_format=session.config.codec,
        sample_rate=session.config.sample_rate,
    )
    ws_stream = WebSocketStream(websocket, ws_config)

    # Set up callbacks
    async def on_audio_received(audio_data: bytes, timestamp: int):
        await session.process_audio(audio_data, timestamp)

    ws_stream.set_callbacks(on_audio_received=on_audio_received)

    try:
        # Connect WebSocket
        await ws_stream.connect()

        # Connect session
        await session.connect()

        # Keep alive until disconnect
        while ws_stream.is_connected and session.state not in (
            SessionState.DISCONNECTED,
            SessionState.FAILED,
        ):
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        logger.info("websocket_disconnected", session_id=session_id)
    except Exception as e:
        logger.error("websocket_error", session_id=session_id, error=str(e))
    finally:
        await ws_stream.disconnect()
        await session_manager.disconnect_session(session_id)


@app.websocket("/v1/media")
async def media_websocket(websocket: WebSocket):
    """
    Generic media WebSocket endpoint (Twilio-compatible).

    Accepts Twilio Media Streams format.
    """
    await websocket.accept()

    session_id: Optional[str] = None

    try:
        while True:
            message = await websocket.receive_json()
            event = message.get("event")

            if event == "start":
                # Stream starting - create session
                start_data = message.get("start", {})
                call_sid = start_data.get("callSid", "unknown")
                stream_sid = start_data.get("streamSid", "unknown")

                if session_manager:
                    config = SessionConfig(
                        call_id=call_sid,
                        agent_id="default",
                        source="telephony",
                        codec="pcmu",
                    )
                    session = await session_manager.create_session(config)
                    session_id = session.id
                    await session.connect()

                logger.info(
                    "twilio_stream_started",
                    call_sid=call_sid,
                    stream_sid=stream_sid,
                    session_id=session_id,
                )

            elif event == "media":
                # Audio data
                if session_id and session_manager:
                    session = await session_manager.get_session(session_id)
                    if session:
                        import base64
                        media = message.get("media", {})
                        payload = media.get("payload", "")
                        if payload:
                            audio_data = base64.b64decode(payload)
                            timestamp = int(media.get("timestamp", 0))
                            await session.process_audio(audio_data, timestamp)

            elif event == "stop":
                # Stream ending
                logger.info("twilio_stream_stopped", session_id=session_id)
                break

    except WebSocketDisconnect:
        logger.info("twilio_websocket_disconnected", session_id=session_id)
    except Exception as e:
        logger.error("twilio_websocket_error", error=str(e))
    finally:
        if session_id and session_manager:
            await session_manager.disconnect_session(session_id)


# =============================================================================
# Audio Processing Endpoints
# =============================================================================


class SendAudioRequest(BaseModel):
    """Send audio request."""
    audio_base64: str
    interrupt: bool = False


@app.post("/v1/sessions/{session_id}/audio")
async def send_audio(session_id: str, request: SendAudioRequest):
    """Send audio to session (TTS output)."""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Service not ready")

    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    import base64
    audio_data = base64.b64decode(request.audio_base64)

    await session.send_audio(audio_data, interrupt=request.interrupt)

    return {"status": "sent", "bytes": len(audio_data)}


@app.post("/v1/sessions/{session_id}/interrupt")
async def interrupt_session(session_id: str):
    """Interrupt current audio playback."""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Service not ready")

    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Clear outbound queue
    session.pipeline.router.clear_outbound_queue()

    return {"status": "interrupted", "session_id": session_id}


# =============================================================================
# Metrics Endpoint
# =============================================================================


@app.get("/metrics")
async def get_metrics():
    """Get service metrics (Prometheus format)."""
    if not session_manager:
        return ""

    metrics = session_manager.get_metrics()

    # Format as Prometheus metrics
    lines = [
        "# HELP media_pipeline_active_sessions Number of active sessions",
        "# TYPE media_pipeline_active_sessions gauge",
        f"media_pipeline_active_sessions {metrics['active_sessions']}",
        "",
        "# HELP media_pipeline_total_sessions_created Total sessions created",
        "# TYPE media_pipeline_total_sessions_created counter",
        f"media_pipeline_total_sessions_created {metrics['total_created']}",
        "",
        "# HELP media_pipeline_total_sessions_completed Total sessions completed",
        "# TYPE media_pipeline_total_sessions_completed counter",
        f"media_pipeline_total_sessions_completed {metrics['total_completed']}",
        "",
        "# HELP media_pipeline_peak_concurrent Peak concurrent sessions",
        "# TYPE media_pipeline_peak_concurrent gauge",
        f"media_pipeline_peak_concurrent {metrics['peak_concurrent']}",
        "",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )
