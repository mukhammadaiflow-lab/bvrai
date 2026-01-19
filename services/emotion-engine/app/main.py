"""
Emotion Engine Service.

Real-time emotion detection and response advisory service
for voice AI applications.

API Endpoints:
- POST /analyze: Analyze audio for emotion
- POST /analyze/stream: Stream analysis via WebSocket
- POST /sessions: Create analysis session
- GET /sessions/{id}: Get session state
- DELETE /sessions/{id}: End session
- GET /health: Health check
- GET /metrics: Performance metrics
"""

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np

from .config import get_settings, EmotionCategory
from .models import (
    AnalyzeRequest,
    AnalyzeResponse,
    CreateSessionRequest,
    SessionStateResponse,
    HealthResponse,
)
from .engine import EmotionEngine, get_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Service metadata
SERVICE_NAME = "emotion-engine"
SERVICE_VERSION = "1.0.0"
START_TIME = time.time()

# Global engine instance
_engine: Optional[EmotionEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _engine

    logger.info(f"Starting {SERVICE_NAME} v{SERVICE_VERSION}")

    # Initialize engine
    _engine = EmotionEngine()
    await _engine.initialize()

    logger.info("Emotion Engine initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down Emotion Engine")
    if _engine:
        await _engine.shutdown()


# Create FastAPI app
app = FastAPI(
    title="Emotion Engine Service",
    description="Real-time emotion detection from voice with response recommendations",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Metrics
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        uptime_seconds=time.time() - START_TIME,
        active_sessions=_engine.active_session_count if _engine else 0,
        analysis_mode=get_settings().analysis_mode,
    )


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get service metrics."""
    if not _engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "uptime_seconds": time.time() - START_TIME,
        "engine": _engine.get_metrics(),
    }


# =============================================================================
# Analysis Endpoints
# =============================================================================


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze audio for emotion.

    Args:
        request: Analysis request with base64-encoded audio

    Returns:
        Emotion prediction, prosodic features, state, and recommendations
    """
    if not _engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    start_time = time.time()

    try:
        # Decode audio
        audio_bytes = base64.b64decode(request.audio_base64)

        # Generate or use session ID
        session_id = request.session_id or f"temp-{int(time.time() * 1000)}"

        # Analyze
        result = await _engine.analyze(
            audio_data=audio_bytes,
            sample_rate=request.sample_rate,
            session_id=session_id,
            include_recommendation=request.include_recommendations,
        )

        processing_time = (time.time() - start_time) * 1000

        return AnalyzeResponse(
            prediction=result.prediction.to_dict(),
            prosodics=result.features.to_dict() if request.include_prosodics else None,
            state=result.state.to_dict() if request.session_id else None,
            recommendation=(
                result.recommendation.to_dict()
                if result.recommendation
                else None
            ),
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/quick")
async def quick_analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    """
    Quick emotion detection without state tracking.

    Faster but doesn't maintain conversation context.
    """
    if not _engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    start_time = time.time()

    try:
        audio_bytes = base64.b64decode(request.audio_base64)

        emotion, confidence = _engine.get_quick_emotion(
            audio_bytes, request.sample_rate
        )

        return {
            "emotion": emotion.value,
            "confidence": confidence,
            "processing_time_ms": (time.time() - start_time) * 1000,
        }

    except Exception as e:
        logger.error(f"Quick analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/dimensional")
async def dimensional_analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    """
    Get arousal-valence coordinates without tracking.

    Returns dimensional emotion representation.
    """
    if not _engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    start_time = time.time()

    try:
        audio_bytes = base64.b64decode(request.audio_base64)

        arousal, valence = _engine.get_dimensional_emotion(
            audio_bytes, request.sample_rate
        )

        return {
            "arousal": arousal,
            "valence": valence,
            "quadrant": _get_quadrant(arousal, valence),
            "processing_time_ms": (time.time() - start_time) * 1000,
        }

    except Exception as e:
        logger.error(f"Dimensional analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_quadrant(arousal: float, valence: float) -> str:
    """Get emotion quadrant from arousal-valence."""
    if arousal >= 0 and valence >= 0:
        return "excited_positive"  # Happy, excited
    elif arousal >= 0 and valence < 0:
        return "excited_negative"  # Angry, fearful
    elif arousal < 0 and valence >= 0:
        return "calm_positive"  # Content, relaxed
    else:
        return "calm_negative"  # Sad, bored


# =============================================================================
# Session Management
# =============================================================================


@app.post("/sessions")
async def create_session(request: CreateSessionRequest) -> Dict[str, Any]:
    """Create a new emotion tracking session."""
    if not _engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    session_id = request.session_id or f"session-{int(time.time() * 1000)}"

    await _engine.start_session(session_id)

    return {
        "session_id": session_id,
        "status": "created",
        "metadata": request.metadata,
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> Dict[str, Any]:
    """Get session emotional state and summary."""
    if not _engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    state = await _engine.get_session_state(session_id)

    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    return state


@app.delete("/sessions/{session_id}")
async def end_session(session_id: str) -> Dict[str, Any]:
    """End session and get final summary."""
    if not _engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    summary = await _engine.end_session(session_id)

    if not summary:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "status": "ended",
        "summary": summary,
    }


# =============================================================================
# WebSocket Streaming
# =============================================================================


@app.websocket("/ws/analyze/{session_id}")
async def websocket_analyze(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming emotion analysis.

    Protocol:
    - Client sends: {"audio": "<base64>", "sample_rate": 16000}
    - Server sends: {"emotion": {...}, "state": {...}, "recommendation": {...}}
    """
    await websocket.accept()

    if not _engine:
        await websocket.close(code=1011, reason="Engine not initialized")
        return

    logger.info(f"WebSocket connection established: {session_id}")

    # Start session
    await _engine.start_session(session_id)

    try:
        while True:
            # Receive audio data
            data = await websocket.receive_json()

            if "audio" not in data:
                await websocket.send_json({"error": "Missing audio data"})
                continue

            try:
                audio_bytes = base64.b64decode(data["audio"])
                sample_rate = data.get("sample_rate", 16000)

                # Analyze
                result = await _engine.analyze(
                    audio_data=audio_bytes,
                    sample_rate=sample_rate,
                    session_id=session_id,
                    include_recommendation=True,
                )

                # Send result
                await websocket.send_json({
                    "type": "emotion_result",
                    "prediction": result.prediction.to_dict(),
                    "state": result.state.to_dict(),
                    "recommendation": (
                        result.recommendation.to_dict()
                        if result.recommendation
                        else None
                    ),
                    "processing_time_ms": result.processing_time_ms,
                })

            except Exception as e:
                logger.error(f"WebSocket analysis error: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")

    finally:
        # End session and send summary
        summary = await _engine.end_session(session_id)
        logger.info(f"Session {session_id} ended")


@app.websocket("/ws/stream/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """
    WebSocket for continuous audio streaming.

    Protocol:
    - Client sends raw audio bytes (binary frames)
    - Server sends JSON emotion updates
    """
    await websocket.accept()

    if not _engine:
        await websocket.close(code=1011, reason="Engine not initialized")
        return

    logger.info(f"Stream connection established: {session_id}")

    # Configuration
    sample_rate = 16000
    chunk_samples = int(sample_rate * 0.1)  # 100ms chunks
    buffer = bytearray()

    await _engine.start_session(session_id)

    try:
        while True:
            # Receive raw audio bytes
            data = await websocket.receive_bytes()
            buffer.extend(data)

            # Process when we have enough data
            min_bytes = chunk_samples * 2  # int16

            while len(buffer) >= min_bytes:
                # Extract chunk
                chunk_bytes = bytes(buffer[:min_bytes])
                buffer = buffer[min_bytes:]

                # Analyze
                result = await _engine.analyze(
                    audio_data=chunk_bytes,
                    sample_rate=sample_rate,
                    session_id=session_id,
                    include_recommendation=True,
                )

                # Send compact result
                await websocket.send_json({
                    "e": result.prediction.primary_emotion.value,
                    "c": round(result.prediction.primary_confidence, 2),
                    "a": round(result.state.smoothed_arousal, 2),
                    "v": round(result.state.smoothed_valence, 2),
                    "s": result.recommendation.primary_strategy if result.recommendation else None,
                })

    except WebSocketDisconnect:
        logger.info(f"Stream disconnected: {session_id}")

    finally:
        await _engine.end_session(session_id)


# =============================================================================
# Utility Endpoints
# =============================================================================


@app.get("/emotions")
async def list_emotions() -> Dict[str, Any]:
    """List all supported emotion categories."""
    return {
        "categories": [e.value for e in EmotionCategory],
        "count": len(EmotionCategory),
    }


@app.post("/recommend")
async def get_recommendation(emotion: str, confidence: float = 0.5) -> Dict[str, Any]:
    """
    Get response recommendation for an emotion.

    Quick lookup without audio analysis.
    """
    try:
        emotion_cat = EmotionCategory(emotion.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown emotion: {emotion}. Valid: {[e.value for e in EmotionCategory]}",
        )

    from .engine import ResponseAdvisor

    advisor = ResponseAdvisor()
    rec = advisor.get_quick_recommendation(emotion_cat, confidence)

    return rec


# =============================================================================
# Error Handlers
# =============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8090,
        reload=True,
    )
