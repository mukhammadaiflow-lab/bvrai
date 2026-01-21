"""
Voice Lab Service - FastAPI Application.

Voice cloning and management service for Builder Engine.
"""

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import structlog
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    Depends,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .config import get_settings, Settings, VoiceProvider, VoiceQuality
from .models import (
    CreateVoiceRequest,
    UpdateVoiceRequest,
    VoiceResponse,
    VoiceListResponse,
    PreviewRequest,
    PreviewResponse,
    AnalyzeAudioResponse,
    CloneStatusResponse,
    Voice,
    VoiceStatus,
    AudioSample,
    SampleType,
    ServiceStats,
    ProviderStatus,
)
from .services import (
    VoiceCloner,
    AudioAnalyzer,
    VoiceStorage,
)
from .services.registry import get_registry, VoiceRegistry


# =============================================================================
# Logging Configuration
# =============================================================================

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()


# =============================================================================
# Application State
# =============================================================================

class AppState:
    """Application state container."""

    def __init__(self):
        self.settings: Optional[Settings] = None
        self.cloner: Optional[VoiceCloner] = None
        self.analyzer: Optional[AudioAnalyzer] = None
        self.storage: Optional[VoiceStorage] = None
        self.registry: Optional[VoiceRegistry] = None
        self.start_time = time.time()

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time


app_state = AppState()


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    settings = get_settings()
    app_state.settings = settings
    app_state.cloner = VoiceCloner(settings)
    app_state.analyzer = AudioAnalyzer(settings.audio)
    app_state.storage = VoiceStorage(settings.storage)
    app_state.registry = get_registry()

    logger.info(
        "voice_lab_starting",
        port=settings.port,
        default_provider=settings.default_provider.value,
    )

    yield

    logger.info("voice_lab_stopping")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Voice Lab Service",
    description="""
    Voice cloning and management service for Builder Engine.

    Features:
    - Instant voice cloning from audio samples (15-60 seconds)
    - Professional voice cloning with studio-quality training
    - Multi-provider support (ElevenLabs, PlayHT, Cartesia, Resemble)
    - Voice style customization
    - Voice library management
    - Compliance and consent management
    """,
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


# =============================================================================
# Dependencies
# =============================================================================

def get_app_settings() -> Settings:
    if app_state.settings is None:
        return get_settings()
    return app_state.settings


# =============================================================================
# Health Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "voice-lab",
        "version": "1.0.0",
        "uptime_seconds": round(app_state.uptime_seconds, 2),
    }


@app.get("/ready")
async def readiness_check():
    """Readiness probe."""
    if app_state.settings is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}


@app.get("/v1/stats", response_model=ServiceStats)
async def get_stats(settings: Settings = Depends(get_app_settings)):
    """Get service statistics."""
    # Check provider health
    providers = []
    for provider in VoiceProvider:
        api_key = settings.get_provider_key(provider)
        providers.append(ProviderStatus(
            provider=provider,
            is_available=bool(api_key),
        ))

    return ServiceStats(
        total_voices=len(app_state.registry._voices) if app_state.registry else 0,
        total_samples=len(app_state.registry._samples) if app_state.registry else 0,
        total_storage_bytes=0,
        active_jobs=len(app_state.cloner._active_jobs) if app_state.cloner else 0,
        completed_jobs_24h=0,
        failed_jobs_24h=0,
        providers=providers,
    )


# =============================================================================
# Voice Endpoints
# =============================================================================

@app.post("/v1/voices", response_model=VoiceResponse)
async def create_voice(
    request: CreateVoiceRequest,
    tenant_id: str = Query(..., description="Tenant ID"),
    user_id: str = Query(..., description="User ID"),
    settings: Settings = Depends(get_app_settings),
):
    """
    Create a new voice for cloning.

    This creates a voice record that samples can be uploaded to.
    After uploading samples, call POST /v1/voices/{voice_id}/clone to start cloning.
    """
    if not app_state.registry:
        raise HTTPException(status_code=503, detail="Registry not available")

    # Check tenant quota
    voices, total = await app_state.registry.list_voices(tenant_id)
    if total >= settings.max_voices_per_tenant:
        raise HTTPException(
            status_code=429,
            detail=f"Voice quota exceeded ({settings.max_voices_per_tenant} voices)",
        )

    voice = await app_state.registry.create_voice(
        tenant_id=tenant_id,
        created_by=user_id,
        name=request.name,
        provider=request.provider or settings.default_provider,
        quality=request.quality,
        description=request.description,
        tags=request.tags,
        style=request.style,
        consent_id=request.consent.consent_id if request.consent else None,
    )

    return VoiceResponse(voice=voice)


@app.get("/v1/voices", response_model=VoiceListResponse)
async def list_voices(
    tenant_id: str = Query(..., description="Tenant ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    search: Optional[str] = Query(None, description="Search by name/description"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
):
    """List voices for a tenant."""
    if not app_state.registry:
        raise HTTPException(status_code=503, detail="Registry not available")

    status_enum = VoiceStatus(status) if status else None
    provider_enum = VoiceProvider(provider) if provider else None
    tag_list = tags.split(",") if tags else None

    voices, total = await app_state.registry.list_voices(
        tenant_id=tenant_id,
        status=status_enum,
        provider=provider_enum,
        tags=tag_list,
        search=search,
        page=page,
        page_size=page_size,
    )

    return VoiceListResponse(
        voices=voices,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


@app.get("/v1/voices/{voice_id}", response_model=VoiceResponse)
async def get_voice(
    voice_id: str,
    tenant_id: str = Query(..., description="Tenant ID"),
):
    """Get voice details."""
    if not app_state.registry:
        raise HTTPException(status_code=503, detail="Registry not available")

    voice = await app_state.registry.get_voice_by_tenant(tenant_id, voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")

    samples = await app_state.registry.get_voice_samples(voice_id)

    return VoiceResponse(voice=voice, samples=samples)


@app.patch("/v1/voices/{voice_id}", response_model=VoiceResponse)
async def update_voice(
    voice_id: str,
    request: UpdateVoiceRequest,
    tenant_id: str = Query(..., description="Tenant ID"),
):
    """Update voice settings."""
    if not app_state.registry:
        raise HTTPException(status_code=503, detail="Registry not available")

    voice = await app_state.registry.get_voice_by_tenant(tenant_id, voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")

    updates = request.model_dump(exclude_unset=True)
    voice = await app_state.registry.update_voice(voice_id, **updates)

    return VoiceResponse(voice=voice)


@app.delete("/v1/voices/{voice_id}")
async def delete_voice(
    voice_id: str,
    tenant_id: str = Query(..., description="Tenant ID"),
    background_tasks: BackgroundTasks = None,
):
    """Delete a voice."""
    if not app_state.registry:
        raise HTTPException(status_code=503, detail="Registry not available")

    voice = await app_state.registry.get_voice_by_tenant(tenant_id, voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")

    # Delete from provider in background
    if voice.provider_voice_id and app_state.cloner:
        background_tasks.add_task(app_state.cloner.delete_voice, voice)

    await app_state.registry.delete_voice(voice_id)

    return {"status": "deleted", "voice_id": voice_id}


# =============================================================================
# Sample Endpoints
# =============================================================================

@app.post("/v1/voices/{voice_id}/samples")
async def upload_sample(
    voice_id: str,
    tenant_id: str = Query(..., description="Tenant ID"),
    file: UploadFile = File(..., description="Audio file"),
):
    """
    Upload an audio sample for voice cloning.

    Supported formats: WAV, MP3, M4A, FLAC, OGG, WebM
    Minimum duration: 15 seconds
    Recommended duration: 60+ seconds for best quality
    """
    if not app_state.registry or not app_state.analyzer or not app_state.storage:
        raise HTTPException(status_code=503, detail="Service not available")

    voice = await app_state.registry.get_voice_by_tenant(tenant_id, voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")

    if voice.status not in (VoiceStatus.PENDING, VoiceStatus.READY):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot add samples to voice in status: {voice.status}",
        )

    # Read file
    audio_data = await file.read()

    # Analyze audio
    analysis = await app_state.analyzer.analyze(
        audio_data=audio_data,
        filename=file.filename or "sample.wav",
        mime_type=file.content_type or "audio/wav",
    )

    if not analysis.is_valid:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Audio validation failed",
                "errors": analysis.validation_errors,
                "recommendations": analysis.recommendations,
            },
        )

    # Store sample
    storage_key, storage_url = await app_state.storage.store_sample(
        tenant_id=tenant_id,
        audio_data=audio_data,
        filename=file.filename or "sample.wav",
        content_type=file.content_type or "audio/wav",
    )

    # Create sample record
    sample = AudioSample(
        voice_id=voice_id,
        sample_type=SampleType.UPLOADED,
        metadata=analysis.metadata,
        storage_path=storage_key,
        storage_url=storage_url,
    )

    sample = await app_state.registry.add_sample(voice_id, sample)

    return {
        "sample_id": sample.sample_id,
        "analysis": {
            "is_valid": analysis.is_valid,
            "quality_score": round(analysis.quality_score, 2),
            "duration_s": analysis.metadata.duration_s,
            "sample_rate": analysis.metadata.sample_rate,
        },
        "warnings": analysis.warnings,
        "recommendations": analysis.recommendations,
    }


@app.post("/v1/analyze")
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file to analyze"),
):
    """
    Analyze an audio file for voice cloning suitability.

    Returns quality assessment and recommendations without storing.
    """
    if not app_state.analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not available")

    audio_data = await file.read()

    analysis = await app_state.analyzer.analyze(
        audio_data=audio_data,
        filename=file.filename or "sample.wav",
        mime_type=file.content_type or "audio/wav",
    )

    return AnalyzeAudioResponse(
        is_valid=analysis.is_valid,
        metadata=analysis.metadata,
        quality_score=analysis.quality_score,
        recommendations=analysis.recommendations,
        validation_errors=analysis.validation_errors,
    )


# =============================================================================
# Clone Endpoints
# =============================================================================

@app.post("/v1/voices/{voice_id}/clone", response_model=CloneStatusResponse)
async def start_clone(
    voice_id: str,
    tenant_id: str = Query(..., description="Tenant ID"),
    settings: Settings = Depends(get_app_settings),
):
    """
    Start voice cloning process.

    Requires at least one uploaded sample. The cloning process runs
    asynchronously - use GET /v1/voices/{voice_id}/clone to check status.
    """
    if not app_state.registry or not app_state.cloner or not app_state.storage:
        raise HTTPException(status_code=503, detail="Service not available")

    voice = await app_state.registry.get_voice_by_tenant(tenant_id, voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")

    if voice.status != VoiceStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Voice already in status: {voice.status}",
        )

    # Check consent if required
    if settings.require_consent and not voice.consent_id:
        raise HTTPException(
            status_code=400,
            detail="Voice consent required before cloning",
        )

    # Get samples
    samples = await app_state.registry.get_voice_samples(voice_id)
    if not samples:
        raise HTTPException(
            status_code=400,
            detail="No samples uploaded. Upload at least one audio sample.",
        )

    # Check minimum duration
    total_duration = sum(s.metadata.duration_s for s in samples if s.metadata)
    min_duration = settings.audio.min_instant_duration_s

    if total_duration < min_duration:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient audio: {total_duration:.1f}s < {min_duration}s minimum",
        )

    # Load audio data
    audio_data = []
    for sample in samples:
        try:
            data = await app_state.storage.get_sample(sample.storage_path)
            audio_data.append(data)
        except Exception as e:
            logger.error(f"Failed to load sample {sample.sample_id}: {e}")

    if not audio_data:
        raise HTTPException(status_code=500, detail="Failed to load samples")

    # Update voice status
    await app_state.registry.update_voice_status(
        voice_id,
        VoiceStatus.PROCESSING,
        status_message="Cloning in progress...",
    )

    # Start clone job
    job = await app_state.cloner.start_clone_job(voice, samples, audio_data)

    return CloneStatusResponse(job=job, voice=voice)


@app.get("/v1/voices/{voice_id}/clone", response_model=CloneStatusResponse)
async def get_clone_status(
    voice_id: str,
    tenant_id: str = Query(..., description="Tenant ID"),
):
    """Get voice cloning status."""
    if not app_state.registry or not app_state.cloner:
        raise HTTPException(status_code=503, detail="Service not available")

    voice = await app_state.registry.get_voice_by_tenant(tenant_id, voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")

    # Find active job for this voice
    job = None
    for j in app_state.cloner._active_jobs.values():
        if j.voice_id == voice_id:
            job = j
            break

    return CloneStatusResponse(job=job, voice=voice)


# =============================================================================
# Preview Endpoints
# =============================================================================

@app.post("/v1/voices/{voice_id}/preview")
async def preview_voice(
    voice_id: str,
    request: PreviewRequest,
    tenant_id: str = Query(..., description="Tenant ID"),
    settings: Settings = Depends(get_app_settings),
):
    """
    Generate a preview with the cloned voice.

    Only available for voices in READY status.
    """
    if not app_state.registry or not app_state.cloner:
        raise HTTPException(status_code=503, detail="Service not available")

    voice = await app_state.registry.get_voice_by_tenant(tenant_id, voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")

    if voice.status != VoiceStatus.READY:
        raise HTTPException(
            status_code=400,
            detail=f"Voice not ready for preview (status: {voice.status})",
        )

    # Limit preview length
    text = request.text[:int(settings.preview_max_length_s * 20)]  # ~20 chars/sec

    try:
        audio_data = await app_state.cloner.preview_voice(
            voice=voice,
            text=text,
            style=request.style,
        )

        # Record usage
        await app_state.registry.record_voice_usage(voice_id)

        return Response(
            content=audio_data,
            media_type=f"audio/{request.output_format}",
            headers={
                "Content-Disposition": f"attachment; filename=preview.{request.output_format}",
            },
        )

    except Exception as e:
        logger.error(f"Preview generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Library Endpoints
# =============================================================================

@app.get("/v1/library")
async def get_library(
    tenant_id: str = Query(..., description="Tenant ID"),
):
    """Get voice library for a tenant."""
    if not app_state.registry:
        raise HTTPException(status_code=503, detail="Registry not available")

    library = await app_state.registry.get_library(tenant_id)
    stats = await app_state.registry.get_tenant_stats(tenant_id)

    return {
        "library": library,
        "stats": stats,
    }


@app.get("/v1/voices/public")
async def get_public_voices(
    tags: Optional[str] = Query(None, description="Filter by tags"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """Get publicly shared voices."""
    if not app_state.registry:
        raise HTTPException(status_code=503, detail="Registry not available")

    tag_list = tags.split(",") if tags else None

    voices, total = await app_state.registry.get_public_voices(
        tags=tag_list,
        page=page,
        page_size=page_size,
    )

    return VoiceListResponse(
        voices=voices,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


# =============================================================================
# Main
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
