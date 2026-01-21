"""Conversation Engine API."""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import get_settings
from app.engine import ConversationEngine, EngineConfig

# Configure logging
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

# Active conversation engines
engines: dict[str, ConversationEngine] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan."""
    logger.info("Starting Conversation Engine", port=settings.port)
    yield
    logger.info("Shutting down Conversation Engine")

    # Stop all engines
    for engine in engines.values():
        await engine.stop()
    engines.clear()


app = FastAPI(
    title="Conversation Engine",
    description="Turn management and state machine for voice conversations",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - use shared secure configuration
try:
    from bvrai_core.security.cors import get_cors_middleware_config
    app.add_middleware(CORSMiddleware, **get_cors_middleware_config())
except ImportError:
    import os
    env = os.getenv("ENVIRONMENT", "development")
    origins = (
        ["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"]
        if env == "development"
        else os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") or ["https://app.bvrai.com"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )


# Models
class HealthResponse(BaseModel):
    status: str
    service: str
    active_sessions: int


class StartSessionRequest(BaseModel):
    session_id: str
    agent_id: str
    greeting: Optional[str] = None
    voice_id: Optional[str] = None
    system_prompt: Optional[str] = None
    tools: list[dict] = []


class StartSessionResponse(BaseModel):
    session_id: str
    status: str


class TranscriptRequest(BaseModel):
    text: str
    is_final: bool = False
    speech_final: bool = False


class SessionStatusResponse(BaseModel):
    session_id: str
    state: str
    turn_count: int
    is_active: bool


# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check."""
    return HealthResponse(
        status="healthy",
        service="conversation-engine",
        active_sessions=len(engines),
    )


@app.post("/sessions", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest):
    """Start a new conversation session."""
    if request.session_id in engines:
        raise HTTPException(
            status_code=400,
            detail="Session already exists",
        )

    config = EngineConfig(
        session_id=request.session_id,
        agent_id=request.agent_id,
        greeting=request.greeting,
        voice_id=request.voice_id,
        system_prompt=request.system_prompt,
        tools=request.tools,
    )

    engine = ConversationEngine(config)
    await engine.start()

    engines[request.session_id] = engine

    logger.info("Session started", session_id=request.session_id)

    return StartSessionResponse(
        session_id=request.session_id,
        status="started",
    )


@app.get("/sessions/{session_id}", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """Get session status."""
    engine = engines.get(session_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionStatusResponse(
        session_id=session_id,
        state=engine.get_state(),
        turn_count=engine.context.turn_count,
        is_active=engine.state_machine.is_active(),
    )


@app.delete("/sessions/{session_id}")
async def end_session(session_id: str, reason: str = "api_request"):
    """End a session."""
    engine = engines.pop(session_id, None)
    if not engine:
        raise HTTPException(status_code=404, detail="Session not found")

    await engine.handle_call_end(reason)
    await engine.stop()

    return {"status": "ended", "session_id": session_id}


@app.post("/sessions/{session_id}/call-start")
async def handle_call_start(session_id: str):
    """Handle call start event."""
    engine = engines.get(session_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Session not found")

    await engine.handle_call_start()
    return {"status": "ok"}


@app.post("/sessions/{session_id}/speech-start")
async def handle_speech_start(session_id: str):
    """Handle speech start (VAD) event."""
    engine = engines.get(session_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Session not found")

    await engine.handle_speech_start()
    return {"status": "ok"}


@app.post("/sessions/{session_id}/speech-end")
async def handle_speech_end(session_id: str):
    """Handle speech end (VAD) event."""
    engine = engines.get(session_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Session not found")

    await engine.handle_speech_end()
    return {"status": "ok"}


@app.post("/sessions/{session_id}/transcript")
async def handle_transcript(session_id: str, request: TranscriptRequest):
    """Handle transcript from ASR."""
    engine = engines.get(session_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Session not found")

    await engine.handle_transcript(
        text=request.text,
        is_final=request.is_final,
        speech_final=request.speech_final,
    )

    return {
        "status": "ok",
        "state": engine.get_state(),
    }


@app.post("/sessions/{session_id}/tts-complete")
async def handle_tts_complete(session_id: str):
    """Handle TTS playback complete."""
    engine = engines.get(session_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Session not found")

    await engine.handle_tts_complete()
    return {"status": "ok"}


@app.get("/sessions/{session_id}/context")
async def get_session_context(session_id: str):
    """Get full session context."""
    engine = engines.get(session_id)
    if not engine:
        raise HTTPException(status_code=404, detail="Session not found")

    return engine.get_context()


@app.websocket("/ws/{session_id}")
async def websocket_session(websocket: WebSocket, session_id: str):
    """
    WebSocket for real-time session events.

    Receives:
    - {"type": "speech_start"}
    - {"type": "speech_end"}
    - {"type": "transcript", "text": "...", "is_final": true}
    - {"type": "tts_complete"}

    Sends:
    - {"type": "speak", "text": "..."}
    - {"type": "state_change", "from": "...", "to": "..."}
    - {"type": "clear_audio"}
    """
    await websocket.accept()

    log = logger.bind(session_id=session_id)
    log.info("WebSocket connected")

    engine = engines.get(session_id)
    if not engine:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return

    # Set up callbacks
    async def on_speak(text: str):
        await websocket.send_json({"type": "speak", "text": text})

    async def on_clear_audio():
        await websocket.send_json({"type": "clear_audio"})

    engine.on_speak(on_speak)
    engine.on_clear_audio(on_clear_audio)

    # Subscribe to state changes
    def on_state_change(transition):
        asyncio.create_task(
            websocket.send_json({
                "type": "state_change",
                "from": transition.from_state.value,
                "to": transition.to_state.value,
                "event": transition.event.value,
            })
        )

    engine.state_machine.on_state_change(on_state_change)

    try:
        while True:
            data = await websocket.receive_json()
            event_type = data.get("type")

            if event_type == "speech_start":
                await engine.handle_speech_start()

            elif event_type == "speech_end":
                await engine.handle_speech_end()

            elif event_type == "transcript":
                await engine.handle_transcript(
                    text=data.get("text", ""),
                    is_final=data.get("is_final", False),
                    speech_final=data.get("speech_final", False),
                )

            elif event_type == "tts_complete":
                await engine.handle_tts_complete()

            elif event_type == "call_start":
                await engine.handle_call_start()

            elif event_type == "call_end":
                await engine.handle_call_end(data.get("reason", "websocket"))
                break

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")

    except Exception as e:
        log.error("WebSocket error", error=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
