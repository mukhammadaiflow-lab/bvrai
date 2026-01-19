"""
Voice API Routes

Handles:
- Voice configuration
- TTS settings
- STT settings
- Voice cloning
- Audio processing
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_db_session,
    get_current_user,
    get_current_tenant,
    UserContext,
    TenantContext,
    require_permissions,
    require_feature,
)

router = APIRouter(prefix="/voice")


# ============================================================================
# Schemas
# ============================================================================

class VoiceProvider(str, Enum):
    """Voice providers."""
    ELEVEN_LABS = "eleven_labs"
    PLAY_HT = "play_ht"
    DEEPGRAM = "deepgram"
    AZURE = "azure"
    GOOGLE = "google"
    AWS = "aws"
    CARTESIA = "cartesia"


class VoiceGender(str, Enum):
    """Voice gender."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class VoiceStyle(str, Enum):
    """Voice style."""
    CONVERSATIONAL = "conversational"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    CHEERFUL = "cheerful"
    CALM = "calm"
    EMPATHETIC = "empathetic"


class VoiceResponse(BaseModel):
    """Voice response."""
    id: str
    name: str
    provider: VoiceProvider
    voice_id: str
    gender: VoiceGender
    language: str
    accent: Optional[str]
    style: Optional[VoiceStyle]
    preview_url: Optional[str]
    is_custom: bool
    is_cloned: bool
    created_at: datetime


class VoiceListResponse(BaseModel):
    """Voice list response."""
    voices: List[VoiceResponse]
    total: int


class VoiceSettings(BaseModel):
    """Voice settings."""
    stability: float = Field(0.5, ge=0, le=1)
    similarity_boost: float = Field(0.75, ge=0, le=1)
    style: float = Field(0.0, ge=0, le=1)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    pitch: float = Field(0.0, ge=-1, le=1)


class TTSRequest(BaseModel):
    """Text-to-speech request."""
    text: str = Field(..., min_length=1, max_length=5000)
    voice_id: str
    provider: VoiceProvider = VoiceProvider.ELEVEN_LABS
    settings: Optional[VoiceSettings] = None
    output_format: str = "mp3"


class TTSResponse(BaseModel):
    """TTS response."""
    audio_url: str
    duration_seconds: float
    characters_used: int


class STTRequest(BaseModel):
    """Speech-to-text request."""
    audio_url: str
    language: str = "en"
    provider: str = "deepgram"
    enable_diarization: bool = False


class STTResponse(BaseModel):
    """STT response."""
    text: str
    confidence: float
    language: str
    duration_seconds: float
    words: List[Dict[str, Any]]


class VoiceCloneRequest(BaseModel):
    """Voice cloning request."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    labels: Dict[str, str] = {}


class VoiceCloneResponse(BaseModel):
    """Voice clone response."""
    id: str
    name: str
    voice_id: str
    status: str
    preview_url: Optional[str]
    created_at: datetime


# ============================================================================
# Voice Library
# ============================================================================

@router.get("/voices", response_model=VoiceListResponse)
async def list_voices(
    provider: Optional[VoiceProvider] = None,
    gender: Optional[VoiceGender] = None,
    language: Optional[str] = None,
    include_custom: bool = True,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
):
    """List available voices."""
    from app.voice import VoiceService

    service = VoiceService()

    voices = await service.list_voices(
        provider=provider.value if provider else None,
        gender=gender.value if gender else None,
        language=language,
        tenant_id=tenant.tenant_id if include_custom else None,
    )

    return VoiceListResponse(
        voices=[
            VoiceResponse(
                id=v.id,
                name=v.name,
                provider=VoiceProvider(v.provider),
                voice_id=v.voice_id,
                gender=VoiceGender(v.gender),
                language=v.language,
                accent=v.accent,
                style=VoiceStyle(v.style) if v.style else None,
                preview_url=v.preview_url,
                is_custom=v.is_custom,
                is_cloned=v.is_cloned,
                created_at=v.created_at,
            )
            for v in voices
        ],
        total=len(voices),
    )


@router.get("/voices/{voice_id}", response_model=VoiceResponse)
async def get_voice(
    voice_id: str,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
):
    """Get voice details."""
    from app.voice import VoiceService

    service = VoiceService()
    voice = await service.get_voice(voice_id)

    if not voice:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Voice not found",
        )

    return VoiceResponse(
        id=voice.id,
        name=voice.name,
        provider=VoiceProvider(voice.provider),
        voice_id=voice.voice_id,
        gender=VoiceGender(voice.gender),
        language=voice.language,
        accent=voice.accent,
        style=VoiceStyle(voice.style) if voice.style else None,
        preview_url=voice.preview_url,
        is_custom=voice.is_custom,
        is_cloned=voice.is_cloned,
        created_at=voice.created_at,
    )


@router.get("/voices/{voice_id}/preview")
async def preview_voice(
    voice_id: str,
    text: str = Query("Hello, this is a preview of my voice.", max_length=500),
    user: UserContext = Depends(get_current_user),
):
    """Generate voice preview."""
    from app.voice import VoiceService

    service = VoiceService()

    try:
        audio_data = await service.generate_preview(voice_id, text)

        async def stream_audio():
            yield audio_data

        return StreamingResponse(
            stream_audio(),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"inline; filename=preview_{voice_id}.mp3",
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate preview: {str(e)}",
        )


# ============================================================================
# Text-to-Speech
# ============================================================================

@router.post("/tts", response_model=TTSResponse)
async def text_to_speech(
    data: TTSRequest,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
):
    """Convert text to speech."""
    from app.voice import VoiceService

    service = VoiceService()

    try:
        result = await service.text_to_speech(
            text=data.text,
            voice_id=data.voice_id,
            provider=data.provider.value,
            settings=data.settings.model_dump() if data.settings else None,
            output_format=data.output_format,
        )

        return TTSResponse(
            audio_url=result["audio_url"],
            duration_seconds=result["duration_seconds"],
            characters_used=len(data.text),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS failed: {str(e)}",
        )


@router.post("/tts/stream")
async def stream_tts(
    data: TTSRequest,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
):
    """Stream text-to-speech audio."""
    from app.voice import VoiceService

    service = VoiceService()

    async def generate():
        async for chunk in service.stream_tts(
            text=data.text,
            voice_id=data.voice_id,
            provider=data.provider.value,
            settings=data.settings.model_dump() if data.settings else None,
        ):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="audio/mpeg",
    )


# ============================================================================
# Speech-to-Text
# ============================================================================

@router.post("/stt", response_model=STTResponse)
async def speech_to_text(
    data: STTRequest,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
):
    """Convert speech to text."""
    from app.voice import VoiceService

    service = VoiceService()

    try:
        result = await service.speech_to_text(
            audio_url=data.audio_url,
            language=data.language,
            provider=data.provider,
            enable_diarization=data.enable_diarization,
        )

        return STTResponse(
            text=result["text"],
            confidence=result["confidence"],
            language=result["language"],
            duration_seconds=result["duration_seconds"],
            words=result.get("words", []),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"STT failed: {str(e)}",
        )


@router.post("/stt/upload", response_model=STTResponse)
async def transcribe_upload(
    file: UploadFile = File(...),
    language: str = "en",
    provider: str = "deepgram",
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
):
    """Transcribe uploaded audio file."""
    from app.voice import VoiceService

    # Validate file type
    allowed_types = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg", "audio/webm"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {allowed_types}",
        )

    # Check file size (max 50MB)
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large. Maximum size is 50MB.",
        )

    service = VoiceService()

    try:
        result = await service.transcribe_audio(
            audio_data=contents,
            language=language,
            provider=provider,
        )

        return STTResponse(
            text=result["text"],
            confidence=result["confidence"],
            language=result["language"],
            duration_seconds=result["duration_seconds"],
            words=result.get("words", []),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}",
        )


# ============================================================================
# Voice Cloning
# ============================================================================

@router.post("/clone", response_model=VoiceCloneResponse)
async def clone_voice(
    data: VoiceCloneRequest,
    files: List[UploadFile] = File(..., description="Audio samples (1-25 files)"),
    user: UserContext = Depends(require_permissions("voice:clone")),
    tenant: TenantContext = Depends(require_feature("voice_cloning")),
):
    """
    Clone a voice from audio samples.

    Upload 1-25 audio samples of the voice to clone.
    Each sample should be 1-10 minutes of clear speech.
    """
    from app.voice import VoiceService

    # Validate file count
    if len(files) < 1 or len(files) > 25:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide 1-25 audio samples",
        )

    # Read all files
    samples = []
    for file in files:
        content = await file.read()
        samples.append({
            "name": file.filename,
            "data": content,
        })

    service = VoiceService()

    try:
        result = await service.clone_voice(
            name=data.name,
            description=data.description,
            samples=samples,
            labels=data.labels,
            tenant_id=tenant.tenant_id,
        )

        return VoiceCloneResponse(
            id=result["id"],
            name=result["name"],
            voice_id=result["voice_id"],
            status=result["status"],
            preview_url=result.get("preview_url"),
            created_at=result["created_at"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice cloning failed: {str(e)}",
        )


@router.get("/clone/{clone_id}", response_model=VoiceCloneResponse)
async def get_voice_clone(
    clone_id: UUID,
    user: UserContext = Depends(get_current_user),
    tenant: TenantContext = Depends(get_current_tenant),
):
    """Get voice clone status."""
    from app.voice import VoiceService

    service = VoiceService()
    clone = await service.get_voice_clone(str(clone_id), tenant.tenant_id)

    if not clone:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Voice clone not found",
        )

    return VoiceCloneResponse(
        id=clone["id"],
        name=clone["name"],
        voice_id=clone["voice_id"],
        status=clone["status"],
        preview_url=clone.get("preview_url"),
        created_at=clone["created_at"],
    )


@router.delete("/clone/{clone_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_voice_clone(
    clone_id: UUID,
    user: UserContext = Depends(require_permissions("voice:clone")),
    tenant: TenantContext = Depends(get_current_tenant),
):
    """Delete a cloned voice."""
    from app.voice import VoiceService

    service = VoiceService()
    deleted = await service.delete_voice_clone(str(clone_id), tenant.tenant_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Voice clone not found",
        )


# ============================================================================
# Audio Processing
# ============================================================================

@router.post("/process/normalize")
async def normalize_audio(
    file: UploadFile = File(...),
    target_db: float = Query(-14.0, ge=-30, le=0),
    user: UserContext = Depends(get_current_user),
):
    """Normalize audio volume."""
    from app.voice import AudioProcessor

    contents = await file.read()
    processor = AudioProcessor()

    processed = await processor.normalize(contents, target_db)

    async def stream():
        yield processed

    return StreamingResponse(
        stream(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename=normalized_{file.filename}",
        },
    )


@router.post("/process/denoise")
async def denoise_audio(
    file: UploadFile = File(...),
    strength: float = Query(0.5, ge=0, le=1),
    user: UserContext = Depends(get_current_user),
):
    """Remove noise from audio."""
    from app.voice import AudioProcessor

    contents = await file.read()
    processor = AudioProcessor()

    processed = await processor.denoise(contents, strength)

    async def stream():
        yield processed

    return StreamingResponse(
        stream(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename=denoised_{file.filename}",
        },
    )


@router.post("/process/convert")
async def convert_audio(
    file: UploadFile = File(...),
    output_format: str = Query("mp3", regex="^(mp3|wav|ogg|flac)$"),
    sample_rate: int = Query(16000, ge=8000, le=48000),
    user: UserContext = Depends(get_current_user),
):
    """Convert audio format."""
    from app.voice import AudioProcessor

    contents = await file.read()
    processor = AudioProcessor()

    processed = await processor.convert(
        contents,
        output_format=output_format,
        sample_rate=sample_rate,
    )

    media_types = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
    }

    async def stream():
        yield processed

    return StreamingResponse(
        stream(),
        media_type=media_types[output_format],
        headers={
            "Content-Disposition": f"attachment; filename=converted.{output_format}",
        },
    )
