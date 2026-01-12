"""ASR Service - Real-time speech-to-text API."""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import get_settings
from app.adapters import DeepgramAdapter, MockASRAdapter, ASRAdapter, TranscriptEvent

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
settings = get_settings()

# Active sessions
sessions: dict[str, ASRAdapter] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info(
        "Starting ASR Service",
        provider=settings.asr_provider,
        port=settings.port,
    )

    yield

    # Cleanup
    logger.info("Shutting down ASR Service")
    for session_id, adapter in list(sessions.items()):
        try:
            await adapter.disconnect()
        except Exception as e:
            logger.error("Error closing session", session_id=session_id, error=str(e))
    sessions.clear()


app = FastAPI(
    title="ASR Service",
    description="Real-time speech-to-text service for Builder Engine",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class HealthResponse(BaseModel):
    status: str
    service: str
    provider: str
    active_sessions: int


class SessionInfo(BaseModel):
    session_id: str
    provider: str
    is_connected: bool
    created_at: float


class TranscribeRequest(BaseModel):
    audio: str  # Base64 encoded audio
    encoding: str = "mulaw"
    sample_rate: int = 8000


class TranscribeResponse(BaseModel):
    text: str
    is_final: bool
    confidence: float
    duration: float


# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="asr-service",
        provider=settings.asr_provider,
        active_sessions=len(sessions),
    )


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    if len(sessions) >= settings.max_connections:
        raise HTTPException(
            status_code=503,
            detail="Max connections reached",
        )
    return {"status": "ready"}


def create_adapter() -> ASRAdapter:
    """Create ASR adapter based on configuration."""
    provider = settings.asr_provider

    if provider == "deepgram":
        if not settings.deepgram_api_key:
            raise ValueError("Deepgram API key is required")
        return DeepgramAdapter(settings.deepgram_api_key)

    elif provider == "mock":
        return MockASRAdapter()

    else:
        raise ValueError(f"Unknown ASR provider: {provider}")


@app.websocket("/ws/transcribe/{session_id}")
async def websocket_transcribe(
    websocket: WebSocket,
    session_id: str,
    language: str = Query(default="en-US"),
    model: str = Query(default=None),
):
    """
    WebSocket endpoint for real-time transcription.

    Audio is sent as binary frames, transcripts returned as JSON.

    Message format (from server):
    {
        "type": "transcript",
        "text": "hello world",
        "is_final": true,
        "confidence": 0.98,
        "words": [...],
        "speech_final": false
    }
    or
    {
        "type": "vad",
        "event": "speech_start" | "speech_end"
    }
    """
    await websocket.accept()

    log = logger.bind(session_id=session_id)
    log.info("WebSocket connection accepted")

    # Create adapter
    try:
        adapter = create_adapter()
        options = {}
        if model:
            options["model"] = model
        if language:
            options["language"] = language

        await adapter.connect(session_id, options)
        sessions[session_id] = adapter

    except Exception as e:
        log.error("Failed to create ASR adapter", error=str(e))
        await websocket.close(code=1011, reason=str(e))
        return

    # Task for sending transcripts back
    async def send_transcripts():
        try:
            async for event in adapter.stream_transcripts():
                message = event_to_message(event)
                if message:
                    await websocket.send_json(message)
        except Exception as e:
            log.error("Error sending transcripts", error=str(e))

    # Start transcript sender
    sender_task = asyncio.create_task(send_transcripts())

    try:
        # Receive audio
        while True:
            data = await websocket.receive_bytes()

            if not adapter.is_connected:
                break

            await adapter.send_audio(data)

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")

    except Exception as e:
        log.error("WebSocket error", error=str(e))

    finally:
        # Cleanup
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass

        if session_id in sessions:
            await adapter.disconnect()
            del sessions[session_id]

        log.info("Session cleaned up")


def event_to_message(event: TranscriptEvent) -> Optional[dict]:
    """Convert TranscriptEvent to WebSocket message."""
    if event.type == "transcript" and event.transcript:
        t = event.transcript
        return {
            "type": "transcript",
            "text": t.text,
            "is_final": t.is_final,
            "confidence": t.confidence,
            "speech_final": t.speech_final,
            "start_time": t.start_time,
            "end_time": t.end_time,
            "words": [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.confidence,
                }
                for w in t.words
            ],
        }

    elif event.type == "vad" and event.vad_event:
        return {
            "type": "vad",
            "event": event.vad_event.value,
        }

    elif event.type == "error":
        return {
            "type": "error",
            "error": event.error,
        }

    return None


@app.get("/sessions")
async def list_sessions():
    """List active sessions."""
    return {
        "sessions": [
            {
                "session_id": sid,
                "provider": adapter.name,
                "is_connected": adapter.is_connected,
            }
            for sid, adapter in sessions.items()
        ],
        "count": len(sessions),
    }


@app.delete("/sessions/{session_id}")
async def close_session(session_id: str):
    """Close a session."""
    adapter = sessions.pop(session_id, None)
    if not adapter:
        raise HTTPException(status_code=404, detail="Session not found")

    await adapter.disconnect()
    return {"status": "closed", "session_id": session_id}


# Metrics endpoint (for Prometheus)
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    metrics_text = f"""
# HELP asr_active_sessions Number of active ASR sessions
# TYPE asr_active_sessions gauge
asr_active_sessions {len(sessions)}

# HELP asr_provider Current ASR provider
# TYPE asr_provider gauge
asr_provider{{provider="{settings.asr_provider}"}} 1
"""
    return metrics_text


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
