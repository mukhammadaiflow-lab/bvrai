"""
Voice Configuration API Routes

Provides REST API endpoints for managing voice configurations.
"""

import logging
import uuid
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..base import APIResponse, success_response, NotFoundError
from ..auth import AuthContext, Permission
from ..dependencies import get_db_session, get_auth_context
from ...database.models import VoiceConfigurationModel


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice-configs", tags=["Voice Configuration"])


# =============================================================================
# Request/Response Models
# =============================================================================

class VoiceConfigResponse(BaseModel):
    """Voice config response."""
    id: str
    name: str
    description: Optional[str] = None
    stt_provider: str
    stt_config: Optional[Dict[str, Any]] = None
    tts_provider: str
    tts_config: Optional[Dict[str, Any]] = None
    voice_id: Optional[str] = None
    language: str
    speed: float
    pitch: float
    is_default: bool
    created_at: str
    updated_at: str


class CreateVoiceConfigRequest(BaseModel):
    """Create voice config request."""
    name: str
    description: Optional[str] = None
    stt_provider: str = "deepgram"
    stt_config: Optional[Dict[str, Any]] = None
    tts_provider: str = "elevenlabs"
    tts_config: Optional[Dict[str, Any]] = None
    voice_id: Optional[str] = None
    language: str = "en"
    speed: float = 1.0
    pitch: float = 1.0


class UpdateVoiceConfigRequest(BaseModel):
    """Update voice config request."""
    name: Optional[str] = None
    description: Optional[str] = None
    stt_provider: Optional[str] = None
    stt_config: Optional[Dict[str, Any]] = None
    tts_provider: Optional[str] = None
    tts_config: Optional[Dict[str, Any]] = None
    voice_id: Optional[str] = None
    language: Optional[str] = None
    speed: Optional[float] = None
    pitch: Optional[float] = None


class VoiceResponse(BaseModel):
    """Available voice response."""
    id: str
    name: str
    provider: str
    gender: str
    language: str
    accent: Optional[str] = None
    preview_url: Optional[str] = None
    description: Optional[str] = None


# =============================================================================
# Available Voices (Mock Data for MVP)
# =============================================================================

AVAILABLE_VOICES = [
    {
        "id": "21m00Tcm4TlvDq8ikWAM",
        "name": "Rachel",
        "provider": "elevenlabs",
        "gender": "female",
        "language": "en",
        "accent": "American",
        "description": "Warm and professional female voice",
    },
    {
        "id": "EXAVITQu4vr4xnSDxMaL",
        "name": "Bella",
        "provider": "elevenlabs",
        "gender": "female",
        "language": "en",
        "accent": "American",
        "description": "Soft and friendly female voice",
    },
    {
        "id": "ErXwobaYiN019PkySvjV",
        "name": "Antoni",
        "provider": "elevenlabs",
        "gender": "male",
        "language": "en",
        "accent": "American",
        "description": "Confident and authoritative male voice",
    },
    {
        "id": "VR6AewLTigWG4xSOukaG",
        "name": "Arnold",
        "provider": "elevenlabs",
        "gender": "male",
        "language": "en",
        "accent": "American",
        "description": "Deep and commanding male voice",
    },
    {
        "id": "onyx",
        "name": "Onyx",
        "provider": "openai",
        "gender": "male",
        "language": "en",
        "description": "Deep, resonant male voice",
    },
    {
        "id": "nova",
        "name": "Nova",
        "provider": "openai",
        "gender": "female",
        "language": "en",
        "description": "Warm, engaging female voice",
    },
    {
        "id": "alloy",
        "name": "Alloy",
        "provider": "openai",
        "gender": "neutral",
        "language": "en",
        "description": "Neutral, balanced voice",
    },
]

STT_PROVIDERS = ["deepgram", "google", "azure", "openai", "assemblyai"]
TTS_PROVIDERS = ["elevenlabs", "openai", "google", "azure", "amazon"]


# =============================================================================
# Helper Functions
# =============================================================================

def config_to_response(config: VoiceConfigurationModel) -> dict:
    """Convert voice config model to response dict."""
    return {
        "id": config.id,
        "name": config.name,
        "description": config.description,
        "stt_provider": config.stt_provider,
        "stt_config": config.stt_config,
        "tts_provider": config.tts_provider,
        "tts_config": config.tts_config,
        "voice_id": config.voice_id,
        "language": config.language or "en",
        "speed": config.speed if hasattr(config, 'speed') and config.speed else 1.0,
        "pitch": config.pitch if hasattr(config, 'pitch') and config.pitch else 1.0,
        "is_default": config.is_default if hasattr(config, 'is_default') else False,
        "created_at": config.created_at.isoformat() if config.created_at else None,
        "updated_at": config.updated_at.isoformat() if config.updated_at else None,
    }


# =============================================================================
# Routes
# =============================================================================

@router.get(
    "",
    response_model=APIResponse[List[VoiceConfigResponse]],
    summary="List Voice Configs",
    description="List all voice configurations.",
)
async def list_voice_configs(
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """List voice configurations."""
    result = await db.execute(
        select(VoiceConfigurationModel).where(
            VoiceConfigurationModel.organization_id == auth.organization_id,
            VoiceConfigurationModel.is_deleted == False,
        )
    )
    configs = result.scalars().all()

    # Return default config if none exist
    if not configs:
        return success_response([{
            "id": "default",
            "name": "Default Voice",
            "description": "Default voice configuration",
            "stt_provider": "deepgram",
            "stt_config": {},
            "tts_provider": "elevenlabs",
            "tts_config": {},
            "voice_id": "21m00Tcm4TlvDq8ikWAM",
            "language": "en",
            "speed": 1.0,
            "pitch": 1.0,
            "is_default": True,
            "created_at": None,
            "updated_at": None,
        }])

    return success_response([config_to_response(c) for c in configs])


@router.get(
    "/{config_id}",
    response_model=APIResponse[VoiceConfigResponse],
    summary="Get Voice Config",
    description="Get a specific voice configuration.",
)
async def get_voice_config(
    config_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Get voice configuration."""
    if config_id == "default":
        return success_response({
            "id": "default",
            "name": "Default Voice",
            "description": "Default voice configuration",
            "stt_provider": "deepgram",
            "stt_config": {},
            "tts_provider": "elevenlabs",
            "tts_config": {},
            "voice_id": "21m00Tcm4TlvDq8ikWAM",
            "language": "en",
            "speed": 1.0,
            "pitch": 1.0,
            "is_default": True,
            "created_at": None,
            "updated_at": None,
        })

    result = await db.execute(
        select(VoiceConfigurationModel).where(
            VoiceConfigurationModel.id == config_id,
            VoiceConfigurationModel.organization_id == auth.organization_id,
        )
    )
    config = result.scalar_one_or_none()

    if not config:
        raise NotFoundError("Voice configuration not found")

    return success_response(config_to_response(config))


@router.post(
    "",
    response_model=APIResponse[VoiceConfigResponse],
    status_code=201,
    summary="Create Voice Config",
    description="Create a new voice configuration.",
)
async def create_voice_config(
    request: CreateVoiceConfigRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Create voice configuration."""
    config = VoiceConfigurationModel(
        organization_id=auth.organization_id,
        name=request.name,
        description=request.description,
        stt_provider=request.stt_provider,
        stt_config=request.stt_config,
        tts_provider=request.tts_provider,
        tts_config=request.tts_config,
        voice_id=request.voice_id,
        language=request.language,
    )
    db.add(config)
    await db.commit()
    await db.refresh(config)

    logger.info(f"Voice config created: {config.id}")
    return success_response(config_to_response(config))


@router.put(
    "/{config_id}",
    response_model=APIResponse[VoiceConfigResponse],
    summary="Update Voice Config",
    description="Update a voice configuration.",
)
async def update_voice_config(
    config_id: str,
    request: UpdateVoiceConfigRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Update voice configuration."""
    result = await db.execute(
        select(VoiceConfigurationModel).where(
            VoiceConfigurationModel.id == config_id,
            VoiceConfigurationModel.organization_id == auth.organization_id,
        )
    )
    config = result.scalar_one_or_none()

    if not config:
        raise NotFoundError("Voice configuration not found")

    # Update fields
    update_data = request.dict(exclude_unset=True)
    for key, value in update_data.items():
        if value is not None:
            setattr(config, key, value)

    await db.commit()
    await db.refresh(config)

    logger.info(f"Voice config updated: {config.id}")
    return success_response(config_to_response(config))


@router.delete(
    "/{config_id}",
    status_code=204,
    summary="Delete Voice Config",
    description="Delete a voice configuration.",
)
async def delete_voice_config(
    config_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Delete voice configuration."""
    result = await db.execute(
        select(VoiceConfigurationModel).where(
            VoiceConfigurationModel.id == config_id,
            VoiceConfigurationModel.organization_id == auth.organization_id,
        )
    )
    config = result.scalar_one_or_none()

    if not config:
        raise NotFoundError("Voice configuration not found")

    config.is_deleted = True
    await db.commit()

    logger.info(f"Voice config deleted: {config_id}")
    return None


@router.post(
    "/{config_id}/set-default",
    response_model=APIResponse[VoiceConfigResponse],
    summary="Set Default",
    description="Set a voice configuration as default.",
)
async def set_default_voice_config(
    config_id: str,
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Set default voice configuration."""
    # Clear existing defaults
    result = await db.execute(
        select(VoiceConfigurationModel).where(
            VoiceConfigurationModel.organization_id == auth.organization_id,
            VoiceConfigurationModel.is_default == True,
        )
    )
    for config in result.scalars().all():
        config.is_default = False

    # Set new default
    result = await db.execute(
        select(VoiceConfigurationModel).where(
            VoiceConfigurationModel.id == config_id,
            VoiceConfigurationModel.organization_id == auth.organization_id,
        )
    )
    config = result.scalar_one_or_none()

    if not config:
        raise NotFoundError("Voice configuration not found")

    config.is_default = True
    await db.commit()
    await db.refresh(config)

    logger.info(f"Default voice config set: {config_id}")
    return success_response(config_to_response(config))


@router.post(
    "/{config_id}/preview",
    summary="Preview Voice",
    description="Generate a preview audio using this voice configuration.",
)
async def preview_voice(
    config_id: str,
    text: str = "Hello! This is a preview of the voice.",
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db_session),
):
    """Generate voice preview."""
    # In production, call TTS provider
    # For MVP, return mock response
    return success_response({
        "message": "Preview generation not implemented in MVP",
        "text": text,
        "config_id": config_id,
    })


@router.get(
    "/voices",
    response_model=APIResponse[List[VoiceResponse]],
    summary="List Available Voices",
    description="List all available voices from providers.",
)
async def list_voices(
    provider: Optional[str] = None,
    auth: AuthContext = Depends(get_auth_context),
):
    """List available voices."""
    voices = AVAILABLE_VOICES

    if provider:
        voices = [v for v in voices if v["provider"] == provider]

    return success_response(voices)


@router.get(
    "/providers",
    response_model=APIResponse[dict],
    summary="List Providers",
    description="List available STT and TTS providers.",
)
async def list_providers(
    auth: AuthContext = Depends(get_auth_context),
):
    """List available providers."""
    return success_response({
        "stt": STT_PROVIDERS,
        "tts": TTS_PROVIDERS,
    })
