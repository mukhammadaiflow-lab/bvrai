"""TTS Service - Real-time text-to-speech API."""

import asyncio
import base64
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from app.config import get_settings
from app.adapters import ElevenLabsAdapter, MockTTSAdapter, TTSAdapter

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

# Global TTS adapter
tts_adapter: Optional[TTSAdapter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global tts_adapter

    logger.info(
        "Starting TTS Service",
        provider=settings.tts_provider,
        port=settings.port,
    )

    # Initialize TTS adapter
    tts_adapter = create_adapter()

    yield

    # Cleanup
    logger.info("Shutting down TTS Service")
    if tts_adapter:
        await tts_adapter.close()


app = FastAPI(
    title="TTS Service",
    description="Real-time text-to-speech service for Builder Engine",
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


def create_adapter() -> TTSAdapter:
    """Create TTS adapter based on configuration."""
    provider = settings.tts_provider

    if provider == "elevenlabs":
        if not settings.elevenlabs_api_key:
            raise ValueError("ElevenLabs API key is required")
        return ElevenLabsAdapter(settings.elevenlabs_api_key)

    elif provider == "mock":
        return MockTTSAdapter()

    else:
        raise ValueError(f"Unknown TTS provider: {provider}")


# Models
class HealthResponse(BaseModel):
    status: str
    service: str
    provider: str


class SynthesizeRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    model_id: Optional[str] = None
    stability: Optional[float] = None
    similarity_boost: Optional[float] = None
    output_format: str = "mulaw"  # mulaw for Twilio, mp3, pcm


class SynthesizeResponse(BaseModel):
    audio: str  # Base64 encoded audio
    sample_rate: int
    encoding: str
    duration_ms: int


class VoiceResponse(BaseModel):
    voice_id: str
    name: str
    description: Optional[str] = None


# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="tts-service",
        provider=settings.tts_provider,
    )


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize text to speech.

    Returns base64-encoded audio.
    """
    if not tts_adapter:
        raise HTTPException(status_code=503, detail="TTS adapter not initialized")

    if len(request.text) > settings.max_text_length:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long. Maximum {settings.max_text_length} characters.",
        )

    try:
        kwargs = {}
        if request.model_id:
            kwargs["model_id"] = request.model_id
        if request.stability is not None:
            kwargs["stability"] = request.stability
        if request.similarity_boost is not None:
            kwargs["similarity_boost"] = request.similarity_boost
        kwargs["output_format"] = "ulaw_8000" if request.output_format == "mulaw" else request.output_format

        audio = await tts_adapter.synthesize(
            text=request.text,
            voice_id=request.voice_id,
            **kwargs,
        )

        # Calculate duration
        sample_rate = 8000 if request.output_format == "mulaw" else 44100
        duration_ms = int(len(audio) / sample_rate * 1000)

        return SynthesizeResponse(
            audio=base64.b64encode(audio).decode(),
            sample_rate=sample_rate,
            encoding=request.output_format,
            duration_ms=duration_ms,
        )

    except Exception as e:
        logger.error("Synthesis failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize/audio")
async def synthesize_audio(request: SynthesizeRequest):
    """
    Synthesize text and return raw audio bytes.

    Returns audio file directly (for streaming to client).
    """
    if not tts_adapter:
        raise HTTPException(status_code=503, detail="TTS adapter not initialized")

    try:
        kwargs = {}
        if request.output_format == "mulaw":
            kwargs["output_format"] = "ulaw_8000"

        audio = await tts_adapter.synthesize(
            text=request.text,
            voice_id=request.voice_id,
            **kwargs,
        )

        # Determine content type
        if request.output_format == "mulaw":
            content_type = "audio/basic"
        elif request.output_format == "mp3":
            content_type = "audio/mpeg"
        else:
            content_type = "audio/pcm"

        return Response(
            content=audio,
            media_type=content_type,
            headers={
                "Content-Disposition": "inline",
                "X-Sample-Rate": "8000" if request.output_format == "mulaw" else "44100",
            },
        )

    except Exception as e:
        logger.error("Synthesis failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/synthesize")
async def websocket_synthesize(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS.

    Client sends JSON: {"text": "Hello world", "voice_id": "optional"}
    Server streams audio chunks as binary frames.
    """
    await websocket.accept()

    log = logger.bind(component="ws_tts")
    log.info("WebSocket connection accepted")

    try:
        while True:
            # Receive text request
            data = await websocket.receive_json()
            text = data.get("text", "")
            voice_id = data.get("voice_id")

            if not text:
                await websocket.send_json({"error": "No text provided"})
                continue

            log.info("Synthesizing", text_length=len(text))

            # Stream audio chunks
            chunk_count = 0
            async for chunk in tts_adapter.synthesize_stream(
                text=text,
                voice_id=voice_id,
                output_format="ulaw_8000",
            ):
                if chunk.data:
                    await websocket.send_bytes(chunk.data)
                    chunk_count += 1

                if chunk.is_final:
                    # Send completion message
                    await websocket.send_json({
                        "event": "complete",
                        "chunks": chunk_count,
                    })
                    break

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")

    except Exception as e:
        log.error("WebSocket error", error=str(e))
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass


@app.get("/voices", response_model=list[VoiceResponse])
async def list_voices():
    """Get available voices."""
    if not tts_adapter:
        raise HTTPException(status_code=503, detail="TTS adapter not initialized")

    try:
        voices = await tts_adapter.get_voices()
        return [
            VoiceResponse(
                voice_id=v.voice_id,
                name=v.name,
                description=v.description,
            )
            for v in voices
        ]
    except Exception as e:
        logger.error("Failed to get voices", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voices/presets")
async def get_voice_presets():
    """Get voice presets from configuration."""
    return settings.voices


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
