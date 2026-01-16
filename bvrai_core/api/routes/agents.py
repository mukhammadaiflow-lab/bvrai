"""
Agent API Routes

This module provides REST API endpoints for managing voice agents.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..base import (
    APIResponse,
    ListResponse,
    PaginationParams,
    PaginationMeta,
    NotFoundError,
    ValidationError,
    success_response,
    paginated_response,
)
from ..auth import (
    AuthContext,
    Permission,
    require_permission,
)
from ..dependencies import get_db_session
from ...database.repositories import AgentRepository


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["Agents"])


# =============================================================================
# Request/Response Models
# =============================================================================


class VoiceConfig(BaseModel):
    """Voice configuration for an agent."""

    provider: str = Field(
        default="elevenlabs",
        description="Voice provider",
    )
    voice_id: str = Field(
        default="default",
        description="Voice ID from provider",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed",
    )
    pitch: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Voice pitch",
    )
    stability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Voice stability",
    )
    similarity_boost: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Similarity boost for cloned voices",
    )


class LLMConfig(BaseModel):
    """LLM configuration for an agent."""

    provider: str = Field(
        default="openai",
        description="LLM provider",
    )
    model: str = Field(
        default="gpt-4",
        description="Model name",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation",
    )
    max_tokens: int = Field(
        default=150,
        ge=1,
        le=4096,
        description="Maximum response tokens",
    )


class BehaviorConfig(BaseModel):
    """Behavior configuration for an agent."""

    interrupt_sensitivity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How sensitive to interruptions",
    )
    silence_timeout_ms: int = Field(
        default=1500,
        ge=500,
        le=10000,
        description="Silence timeout before agent speaks",
    )
    max_turn_duration_ms: int = Field(
        default=30000,
        ge=5000,
        le=120000,
        description="Maximum turn duration",
    )
    end_call_phrases: List[str] = Field(
        default_factory=lambda: ["goodbye", "bye", "end call"],
        description="Phrases that end the call",
    )
    transfer_phrases: List[str] = Field(
        default_factory=lambda: ["transfer me", "speak to human", "talk to agent"],
        description="Phrases that trigger transfer",
    )


class TranscriptionConfig(BaseModel):
    """Transcription configuration."""

    provider: str = Field(
        default="deepgram",
        description="Transcription provider",
    )
    language: str = Field(
        default="en-US",
        description="Primary language",
    )
    punctuate: bool = Field(
        default=True,
        description="Add punctuation",
    )
    profanity_filter: bool = Field(
        default=False,
        description="Filter profanity",
    )


class AgentCreateRequest(BaseModel):
    """Request to create a new agent."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Agent name",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Agent description",
    )
    system_prompt: str = Field(
        ...,
        min_length=10,
        description="System prompt defining agent behavior",
    )
    first_message: Optional[str] = Field(
        default=None,
        description="First message when call starts",
    )
    industry: Optional[str] = Field(
        default=None,
        description="Industry vertical",
    )

    # Configuration
    voice: Optional[VoiceConfig] = Field(
        default=None,
        description="Voice configuration",
    )
    llm: Optional[LLMConfig] = Field(
        default=None,
        description="LLM configuration",
    )
    behavior: Optional[BehaviorConfig] = Field(
        default=None,
        description="Behavior configuration",
    )
    transcription: Optional[TranscriptionConfig] = Field(
        default=None,
        description="Transcription configuration",
    )

    # Knowledge base
    knowledge_base_ids: List[str] = Field(
        default_factory=list,
        description="IDs of knowledge bases to attach",
    )

    # Functions/Tools
    functions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Functions the agent can call",
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata",
    )


class AgentUpdateRequest(BaseModel):
    """Request to update an agent."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)
    system_prompt: Optional[str] = Field(default=None, min_length=10)
    first_message: Optional[str] = None
    industry: Optional[str] = None
    voice: Optional[VoiceConfig] = None
    llm: Optional[LLMConfig] = None
    behavior: Optional[BehaviorConfig] = None
    transcription: Optional[TranscriptionConfig] = None
    knowledge_base_ids: Optional[List[str]] = None
    functions: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class AgentResponse(BaseModel):
    """Agent response model."""

    id: str
    organization_id: str
    name: str
    description: Optional[str] = None
    system_prompt: str
    first_message: Optional[str] = None
    industry: Optional[str] = None

    # Configuration
    voice: VoiceConfig
    llm: LLMConfig
    behavior: BehaviorConfig
    transcription: TranscriptionConfig

    # Knowledge bases
    knowledge_base_ids: List[str] = []

    # Functions
    functions: List[Dict[str, Any]] = []

    # Status
    is_active: bool = True

    # Metadata
    metadata: Dict[str, Any] = {}

    # Timestamps
    created_at: datetime
    updated_at: datetime

    # Stats
    total_calls: int = 0
    total_minutes: float = 0.0

    class Config:
        from_attributes = True


class AgentSummary(BaseModel):
    """Summary of an agent for list views."""

    id: str
    name: str
    description: Optional[str] = None
    industry: Optional[str] = None
    is_active: bool = True
    total_calls: int = 0
    created_at: datetime
    updated_at: datetime


# =============================================================================
# Helper Functions
# =============================================================================


def agent_to_response(agent) -> dict:
    """Convert database agent model to response dict."""
    # Parse JSON configs or use defaults
    voice_config = agent.voice_config if isinstance(agent.voice_config, dict) else {}
    llm_config = agent.llm_config if isinstance(agent.llm_config, dict) else {}
    behavior_config = agent.behavior_config if isinstance(agent.behavior_config, dict) else {}
    transcription_config = agent.transcription_config if isinstance(agent.transcription_config, dict) else {}

    return {
        "id": agent.id,
        "organization_id": agent.organization_id,
        "name": agent.name,
        "description": agent.description,
        "system_prompt": agent.system_prompt or "",
        "first_message": agent.first_message,
        "industry": agent.industry,
        "voice": VoiceConfig(**voice_config).dict() if voice_config else VoiceConfig().dict(),
        "llm": LLMConfig(**llm_config).dict() if llm_config else LLMConfig().dict(),
        "behavior": BehaviorConfig(**behavior_config).dict() if behavior_config else BehaviorConfig().dict(),
        "transcription": TranscriptionConfig(**transcription_config).dict() if transcription_config else TranscriptionConfig().dict(),
        "knowledge_base_ids": agent.knowledge_base_ids or [],
        "functions": agent.functions or [],
        "is_active": agent.is_active,
        "metadata": agent.extra_data or {},
        "created_at": agent.created_at,
        "updated_at": agent.updated_at,
        "total_calls": agent.total_calls or 0,
        "total_minutes": agent.total_minutes or 0.0,
    }


def agent_to_summary(agent) -> dict:
    """Convert database agent model to summary dict."""
    return {
        "id": agent.id,
        "name": agent.name,
        "description": agent.description,
        "industry": agent.industry,
        "is_active": agent.is_active,
        "total_calls": agent.total_calls or 0,
        "created_at": agent.created_at,
        "updated_at": agent.updated_at,
    }


# =============================================================================
# Routes
# =============================================================================


@router.post(
    "",
    response_model=APIResponse[AgentResponse],
    status_code=201,
    summary="Create Agent",
    description="Create a new voice agent with custom configuration.",
)
async def create_agent(
    request: AgentCreateRequest,
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Create a new voice agent."""
    auth.require_permission(Permission.AGENTS_WRITE)

    repo = AgentRepository(db)

    # Create agent in database
    agent = await repo.create(
        organization_id=auth.organization_id,
        name=request.name,
        description=request.description,
        system_prompt=request.system_prompt,
        first_message=request.first_message,
        industry=request.industry,
        voice_config=(request.voice or VoiceConfig()).dict(),
        llm_config=(request.llm or LLMConfig()).dict(),
        behavior_config=(request.behavior or BehaviorConfig()).dict(),
        transcription_config=(request.transcription or TranscriptionConfig()).dict(),
        knowledge_base_ids=request.knowledge_base_ids,
        functions=request.functions,
        extra_data=request.metadata,
        is_active=True,
    )

    logger.info(f"Created agent {agent.id} for org {auth.organization_id}")

    return success_response(agent_to_response(agent))


@router.get(
    "",
    response_model=ListResponse[AgentSummary],
    summary="List Agents",
    description="List all agents for the organization.",
)
async def list_agents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    is_active: Optional[bool] = Query(None),
    industry: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """List all agents."""
    auth.require_permission(Permission.AGENTS_READ)

    repo = AgentRepository(db)

    # Get agents for organization
    skip = (page - 1) * page_size
    agents = await repo.get_by_organization(
        organization_id=auth.organization_id,
        skip=skip,
        limit=page_size,
        is_active=is_active,
    )

    # Get total count
    total = await repo.count_by_organization(auth.organization_id, is_active=is_active)

    return paginated_response(
        items=[agent_to_summary(a) for a in agents],
        page=page,
        page_size=page_size,
        total_items=total,
    )


@router.get(
    "/{agent_id}",
    response_model=APIResponse[AgentResponse],
    summary="Get Agent",
    description="Get details of a specific agent.",
)
async def get_agent(
    agent_id: str = Path(..., description="Agent ID"),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Get agent by ID."""
    auth.require_permission(Permission.AGENTS_READ)

    repo = AgentRepository(db)
    agent = await repo.get_by_id(agent_id)

    if not agent:
        raise NotFoundError("Agent", agent_id)

    # Verify organization ownership
    if agent.organization_id != auth.organization_id:
        raise NotFoundError("Agent", agent_id)

    return success_response(agent_to_response(agent))


@router.patch(
    "/{agent_id}",
    response_model=APIResponse[AgentResponse],
    summary="Update Agent",
    description="Update an existing agent's configuration.",
)
async def update_agent(
    agent_id: str = Path(..., description="Agent ID"),
    request: AgentUpdateRequest = Body(...),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Update an agent."""
    auth.require_permission(Permission.AGENTS_WRITE)

    repo = AgentRepository(db)
    agent = await repo.get_by_id(agent_id)

    if not agent:
        raise NotFoundError("Agent", agent_id)

    if agent.organization_id != auth.organization_id:
        raise NotFoundError("Agent", agent_id)

    # Build update dict with only provided fields
    update_data = {}
    if request.name is not None:
        update_data["name"] = request.name
    if request.description is not None:
        update_data["description"] = request.description
    if request.system_prompt is not None:
        update_data["system_prompt"] = request.system_prompt
    if request.first_message is not None:
        update_data["first_message"] = request.first_message
    if request.industry is not None:
        update_data["industry"] = request.industry
    if request.voice is not None:
        update_data["voice_config"] = request.voice.dict()
    if request.llm is not None:
        update_data["llm_config"] = request.llm.dict()
    if request.behavior is not None:
        update_data["behavior_config"] = request.behavior.dict()
    if request.transcription is not None:
        update_data["transcription_config"] = request.transcription.dict()
    if request.knowledge_base_ids is not None:
        update_data["knowledge_base_ids"] = request.knowledge_base_ids
    if request.functions is not None:
        update_data["functions"] = request.functions
    if request.metadata is not None:
        update_data["extra_data"] = request.metadata
    if request.is_active is not None:
        update_data["is_active"] = request.is_active

    if update_data:
        agent = await repo.update(agent_id, **update_data)

    logger.info(f"Updated agent {agent_id}")

    return success_response(agent_to_response(agent))


@router.delete(
    "/{agent_id}",
    status_code=204,
    summary="Delete Agent",
    description="Delete an agent. This cannot be undone.",
)
async def delete_agent(
    agent_id: str = Path(..., description="Agent ID"),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Delete an agent."""
    auth.require_permission(Permission.AGENTS_DELETE)

    repo = AgentRepository(db)
    agent = await repo.get_by_id(agent_id)

    if not agent:
        raise NotFoundError("Agent", agent_id)

    if agent.organization_id != auth.organization_id:
        raise NotFoundError("Agent", agent_id)

    # Soft delete the agent
    await repo.soft_delete(agent_id)

    logger.info(f"Deleted agent {agent_id}")

    return None


@router.post(
    "/{agent_id}/duplicate",
    response_model=APIResponse[AgentResponse],
    status_code=201,
    summary="Duplicate Agent",
    description="Create a copy of an existing agent.",
)
async def duplicate_agent(
    agent_id: str = Path(..., description="Agent ID to duplicate"),
    name: str = Query(..., description="Name for the new agent"),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Duplicate an agent."""
    auth.require_permission(Permission.AGENTS_WRITE)

    repo = AgentRepository(db)
    original = await repo.get_by_id(agent_id)

    if not original:
        raise NotFoundError("Agent", agent_id)

    if original.organization_id != auth.organization_id:
        raise NotFoundError("Agent", agent_id)

    # Create duplicate
    duplicate = await repo.create(
        organization_id=auth.organization_id,
        name=name,
        description=original.description,
        system_prompt=original.system_prompt,
        first_message=original.first_message,
        industry=original.industry,
        voice_config=original.voice_config,
        llm_config=original.llm_config,
        behavior_config=original.behavior_config,
        transcription_config=original.transcription_config,
        knowledge_base_ids=original.knowledge_base_ids,
        functions=original.functions,
        extra_data=original.extra_data,
        is_active=True,
    )

    logger.info(f"Duplicated agent {agent_id} to {duplicate.id}")

    return success_response(agent_to_response(duplicate))


@router.post(
    "/{agent_id}/publish",
    response_model=APIResponse[AgentResponse],
    summary="Publish Agent",
    description="Publish agent changes and create a new version.",
)
async def publish_agent(
    agent_id: str = Path(..., description="Agent ID"),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Publish agent changes."""
    auth.require_permission(Permission.AGENTS_WRITE)

    repo = AgentRepository(db)
    agent = await repo.get_by_id(agent_id)

    if not agent:
        raise NotFoundError("Agent", agent_id)

    if agent.organization_id != auth.organization_id:
        raise NotFoundError("Agent", agent_id)

    # Create a new version
    await repo.create_version(agent_id)

    # Update agent to active
    agent = await repo.update(agent_id, is_active=True)

    logger.info(f"Published agent {agent_id}")

    return success_response(agent_to_response(agent))


@router.get(
    "/{agent_id}/versions",
    response_model=ListResponse[dict],
    summary="Get Agent Versions",
    description="Get version history for an agent.",
)
async def get_agent_versions(
    agent_id: str = Path(..., description="Agent ID"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Get agent version history."""
    auth.require_permission(Permission.AGENTS_READ)

    repo = AgentRepository(db)
    agent = await repo.get_by_id(agent_id)

    if not agent:
        raise NotFoundError("Agent", agent_id)

    if agent.organization_id != auth.organization_id:
        raise NotFoundError("Agent", agent_id)

    skip = (page - 1) * page_size
    versions = await repo.get_versions(agent_id, skip=skip, limit=page_size)

    # Convert to dict
    version_dicts = []
    for v in versions:
        version_dicts.append({
            "id": v.id,
            "agent_id": v.agent_id,
            "version_number": v.version_number,
            "config_snapshot": v.config_snapshot,
            "created_at": v.created_at,
            "created_by": v.created_by,
        })

    return paginated_response(
        items=version_dicts,
        page=page,
        page_size=page_size,
        total_items=len(version_dicts),  # Would need a count query
    )


@router.post(
    "/{agent_id}/versions/{version_id}/rollback",
    response_model=APIResponse[AgentResponse],
    summary="Rollback Agent",
    description="Rollback agent to a previous version.",
)
async def rollback_agent(
    agent_id: str = Path(..., description="Agent ID"),
    version_id: str = Path(..., description="Version ID to rollback to"),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_session),
):
    """Rollback agent to a previous version."""
    auth.require_permission(Permission.AGENTS_WRITE)

    repo = AgentRepository(db)
    agent = await repo.get_by_id(agent_id)

    if not agent:
        raise NotFoundError("Agent", agent_id)

    if agent.organization_id != auth.organization_id:
        raise NotFoundError("Agent", agent_id)

    # Rollback to version
    agent = await repo.rollback_to_version(agent_id, version_id)

    if not agent:
        raise NotFoundError("Version", version_id)

    logger.info(f"Rolled back agent {agent_id} to version {version_id}")

    return success_response(agent_to_response(agent))
