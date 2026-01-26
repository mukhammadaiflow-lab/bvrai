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
from slugify import slugify

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
from ..dependencies import get_db_with_org_context
from ...database.repositories import AgentRepository
from ...database.models import Agent


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
        ...,
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

    greeting_message: Optional[str] = Field(
        default=None,
        description="Custom greeting message",
    )
    end_call_message: Optional[str] = Field(
        default=None,
        description="Message before ending call",
    )
    silence_timeout_seconds: int = Field(
        default=10,
        ge=5,
        le=60,
        description="Seconds of silence before prompting",
    )
    max_call_duration_seconds: int = Field(
        default=1800,
        ge=60,
        le=7200,
        description="Maximum call duration",
    )
    interruption_sensitivity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How easily agent can be interrupted",
    )
    enable_voicemail_detection: bool = Field(
        default=True,
        description="Detect and handle voicemail",
    )
    voicemail_action: str = Field(
        default="leave_message",
        description="Action when voicemail detected",
    )


class TranscriptionConfig(BaseModel):
    """Transcription configuration."""

    provider: str = Field(
        default="deepgram",
        description="STT provider",
    )
    language: str = Field(
        default="en-US",
        description="Language code",
    )
    enable_punctuation: bool = Field(
        default=True,
        description="Enable auto-punctuation",
    )
    enable_profanity_filter: bool = Field(
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

    # System prompt
    system_prompt: str = Field(
        ...,
        min_length=10,
        description="System prompt defining agent behavior",
    )

    # First message
    first_message: Optional[str] = Field(
        default=None,
        description="First message when answering calls",
    )

    # Industry
    industry: Optional[str] = Field(
        default=None,
        description="Industry type for specialized behavior",
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

    name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=100,
    )
    description: Optional[str] = Field(
        default=None,
        max_length=500,
    )
    system_prompt: Optional[str] = Field(
        default=None,
        min_length=10,
    )
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
        orm_mode = True


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


class AgentFromBusinessRequest(BaseModel):
    """Request to generate agent from business information."""

    business_name: str = Field(..., description="Business name")
    business_type: str = Field(..., description="Type of business")
    industry: str = Field(..., description="Industry")

    # Business details
    description: Optional[str] = Field(
        default=None,
        description="Business description",
    )
    services: List[str] = Field(
        default_factory=list,
        description="Services offered",
    )
    hours_of_operation: Optional[Dict[str, str]] = Field(
        default=None,
        description="Business hours",
    )
    address: Optional[str] = Field(
        default=None,
        description="Business address",
    )
    phone: Optional[str] = Field(
        default=None,
        description="Business phone",
    )
    website: Optional[str] = Field(
        default=None,
        description="Business website",
    )

    # Branding
    tone: str = Field(
        default="professional",
        description="Desired tone (professional, friendly, formal)",
    )
    key_phrases: List[str] = Field(
        default_factory=list,
        description="Key phrases to use",
    )

    # Agent name
    agent_name: Optional[str] = Field(
        default=None,
        description="Name for the agent persona",
    )


class AgentTestRequest(BaseModel):
    """Request to test an agent."""

    message: str = Field(..., description="Test message to send")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context",
    )


class AgentTestResponse(BaseModel):
    """Response from agent test."""

    input_message: str
    response: str
    response_time_ms: float
    tokens_used: int
    functions_called: List[Dict[str, Any]] = []


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
    db: AsyncSession = Depends(get_db_with_org_context),
):
    """Create a new voice agent."""
    auth.require_permission(Permission.AGENTS_WRITE)

    # Generate unique slug from name
    base_slug = slugify(request.name) if request.name else "agent"
    slug = f"{base_slug}-{uuid.uuid4().hex[:8]}"

    # Prepare LLM config
    llm_config = request.llm or LLMConfig()
    voice_config = request.voice or VoiceConfig(voice_id="default")
    behavior = request.behavior or BehaviorConfig()

    # Create agent in database
    repo = AgentRepository(db)
    agent = await repo.create(
        id=f"agt_{uuid.uuid4().hex}",
        organization_id=auth.organization_id,
        name=request.name,
        description=request.description,
        slug=slug,
        system_prompt=request.system_prompt,
        first_message=request.first_message or behavior.greeting_message,
        llm_provider=llm_config.provider,
        llm_model=llm_config.model,
        llm_temperature=llm_config.temperature,
        llm_max_tokens=llm_config.max_tokens,
        is_active=True,
        metadata_json={
            "industry": request.industry,
            "voice": voice_config.dict() if voice_config else None,
            "behavior": behavior.dict() if behavior else None,
            "knowledge_base_ids": request.knowledge_base_ids,
            "functions": request.functions,
            **(request.metadata or {}),
        },
    )

    await db.commit()

    # Build response
    response = AgentResponse(
        id=agent.id,
        organization_id=agent.organization_id,
        name=agent.name,
        description=agent.description,
        system_prompt=agent.system_prompt,
        first_message=agent.first_message,
        industry=request.industry,
        voice=voice_config,
        llm=llm_config,
        behavior=behavior,
        transcription=request.transcription or TranscriptionConfig(),
        knowledge_base_ids=request.knowledge_base_ids or [],
        functions=request.functions or [],
        metadata=request.metadata or {},
        created_at=agent.created_at,
        updated_at=agent.updated_at,
        total_calls=agent.total_calls,
        total_minutes=agent.total_minutes,
    )

    return success_response(response.dict())


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
    db: AsyncSession = Depends(get_db_with_org_context),
):
    """List all agents."""
    auth.require_permission(Permission.AGENTS_READ)

    # Query database
    repo = AgentRepository(db)
    skip = (page - 1) * page_size

    # Get agents with filters
    include_inactive = is_active is None or is_active is False
    agents = await repo.list_by_organization(
        organization_id=auth.organization_id,
        include_inactive=include_inactive,
        skip=skip,
        limit=page_size,
    )

    # Filter by industry if specified (in metadata)
    if industry:
        agents = [
            a for a in agents
            if a.metadata_json and a.metadata_json.get("industry") == industry
        ]

    # Filter by search term if specified
    if search:
        search_lower = search.lower()
        agents = [
            a for a in agents
            if search_lower in (a.name or "").lower()
            or search_lower in (a.description or "").lower()
        ]

    # Build summaries
    agent_summaries = [
        AgentSummary(
            id=a.id,
            name=a.name,
            description=a.description,
            industry=a.metadata_json.get("industry") if a.metadata_json else None,
            is_active=a.is_active,
            total_calls=a.total_calls,
            created_at=a.created_at,
            updated_at=a.updated_at,
        )
        for a in agents
    ]

    # Get total count for pagination
    total_count = len(agent_summaries)  # For now, use filtered count

    return paginated_response(
        items=[a.dict() for a in agent_summaries],
        page=page,
        page_size=page_size,
        total_items=total_count,
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
    db: AsyncSession = Depends(get_db_with_org_context),
):
    """Get agent by ID."""
    auth.require_permission(Permission.AGENTS_READ)

    # Query database
    repo = AgentRepository(db)
    agent = await repo.get_by_id(agent_id)

    if not agent or agent.organization_id != auth.organization_id:
        raise NotFoundError("Agent", agent_id)

    if agent.is_deleted:
        raise NotFoundError("Agent", agent_id)

    # Build response from database model
    metadata = agent.metadata_json or {}
    voice_config = VoiceConfig(**(metadata.get("voice") or {"voice_id": "default"}))
    behavior_config = BehaviorConfig(**(metadata.get("behavior") or {}))

    response = AgentResponse(
        id=agent.id,
        organization_id=agent.organization_id,
        name=agent.name,
        description=agent.description,
        system_prompt=agent.system_prompt,
        first_message=agent.first_message,
        industry=metadata.get("industry"),
        voice=voice_config,
        llm=LLMConfig(
            provider=agent.llm_provider,
            model=agent.llm_model,
            temperature=agent.llm_temperature,
            max_tokens=agent.llm_max_tokens,
        ),
        behavior=behavior_config,
        transcription=TranscriptionConfig(),
        knowledge_base_ids=metadata.get("knowledge_base_ids") or [],
        functions=metadata.get("functions") or [],
        metadata={k: v for k, v in metadata.items() if k not in ["voice", "behavior", "industry", "knowledge_base_ids", "functions"]},
        is_active=agent.is_active,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
        total_calls=agent.total_calls,
        total_minutes=agent.total_minutes,
    )

    return success_response(response.dict())


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
    db: AsyncSession = Depends(get_db_with_org_context),
):
    """Update an agent."""
    auth.require_permission(Permission.AGENTS_WRITE)

    # Get existing agent
    repo = AgentRepository(db)
    agent = await repo.get_by_id(agent_id)

    if not agent or agent.organization_id != auth.organization_id:
        raise NotFoundError("Agent", agent_id)

    if agent.is_deleted:
        raise NotFoundError("Agent", agent_id)

    # Build update dict
    update_data = {}
    if request.name is not None:
        update_data["name"] = request.name
    if request.description is not None:
        update_data["description"] = request.description
    if request.system_prompt is not None:
        update_data["system_prompt"] = request.system_prompt
    if request.first_message is not None:
        update_data["first_message"] = request.first_message
    if request.is_active is not None:
        update_data["is_active"] = request.is_active

    # Handle LLM config
    if request.llm:
        update_data["llm_provider"] = request.llm.provider
        update_data["llm_model"] = request.llm.model
        update_data["llm_temperature"] = request.llm.temperature
        update_data["llm_max_tokens"] = request.llm.max_tokens

    # Update metadata fields
    current_metadata = agent.metadata_json or {}
    if request.voice:
        current_metadata["voice"] = request.voice.dict()
    if request.behavior:
        current_metadata["behavior"] = request.behavior.dict()
    if request.industry is not None:
        current_metadata["industry"] = request.industry
    if request.knowledge_base_ids is not None:
        current_metadata["knowledge_base_ids"] = request.knowledge_base_ids
    if request.functions is not None:
        current_metadata["functions"] = request.functions
    if request.metadata is not None:
        current_metadata.update(request.metadata)

    update_data["metadata_json"] = current_metadata
    update_data["updated_at"] = datetime.utcnow()

    # Update agent
    updated_agent = await repo.update(agent_id, **update_data)
    await db.commit()

    # Return updated agent
    return await get_agent(agent_id, auth, db)


@router.delete(
    "/{agent_id}",
    status_code=204,
    summary="Delete Agent",
    description="Delete an agent. This cannot be undone.",
)
async def delete_agent(
    agent_id: str = Path(..., description="Agent ID"),
    auth: AuthContext = Depends(),
    db: AsyncSession = Depends(get_db_with_org_context),
):
    """Delete an agent."""
    auth.require_permission(Permission.AGENTS_DELETE)

    # Get existing agent
    repo = AgentRepository(db)
    agent = await repo.get_by_id(agent_id)

    if not agent or agent.organization_id != auth.organization_id:
        raise NotFoundError("Agent", agent_id)

    if agent.is_deleted:
        raise NotFoundError("Agent", agent_id)

    # Soft delete the agent
    success = await repo.soft_delete(agent_id)
    if not success:
        raise NotFoundError("Agent", agent_id)

    await db.commit()
    return None  # 204 No Content


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
):
    """Duplicate an agent."""
    auth.require_permission(Permission.AGENTS_WRITE)

    raise NotFoundError("Agent", agent_id)


@router.post(
    "/generate-from-business",
    response_model=APIResponse[AgentResponse],
    status_code=201,
    summary="Generate Agent from Business Info",
    description="Automatically generate an agent based on business information.",
)
async def generate_agent_from_business(
    request: AgentFromBusinessRequest,
    auth: AuthContext = Depends(),
):
    """
    Generate an AI voice agent from business information.

    This uses the Agent Factory to:
    1. Analyze the business type and industry
    2. Generate appropriate persona and tone
    3. Create system prompt with industry knowledge
    4. Configure voice and behavior settings
    """
    auth.require_permission(Permission.AGENTS_WRITE)

    # In production, this would use the AgentFactory from platform/agent_factory
    # from bvrai_core.agent_factory import AgentFactory, BusinessInfo
    # factory = AgentFactory()
    # result = await factory.create_agent(business_info)

    # Placeholder response
    return success_response({
        "id": "agt_" + "x" * 24,
        "name": request.agent_name or f"{request.business_name} Assistant",
        "message": "Agent generation would happen here using AgentFactory",
    })


@router.post(
    "/{agent_id}/test",
    response_model=APIResponse[AgentTestResponse],
    summary="Test Agent",
    description="Send a test message to an agent and get a response.",
)
async def test_agent(
    agent_id: str = Path(..., description="Agent ID"),
    request: AgentTestRequest = Body(...),
    auth: AuthContext = Depends(),
):
    """Test an agent with a message."""
    auth.require_permission(Permission.AGENTS_EXECUTE)

    # In production, this would:
    # 1. Load agent configuration
    # 2. Send message to LLM with system prompt
    # 3. Return response with timing

    raise NotFoundError("Agent", agent_id)


@router.post(
    "/{agent_id}/activate",
    response_model=APIResponse[AgentResponse],
    summary="Activate Agent",
    description="Activate an agent so it can receive calls.",
)
async def activate_agent(
    agent_id: str = Path(..., description="Agent ID"),
    auth: AuthContext = Depends(),
):
    """Activate an agent."""
    auth.require_permission(Permission.AGENTS_WRITE)

    raise NotFoundError("Agent", agent_id)


@router.post(
    "/{agent_id}/deactivate",
    response_model=APIResponse[AgentResponse],
    summary="Deactivate Agent",
    description="Deactivate an agent. It will no longer receive calls.",
)
async def deactivate_agent(
    agent_id: str = Path(..., description="Agent ID"),
    auth: AuthContext = Depends(),
):
    """Deactivate an agent."""
    auth.require_permission(Permission.AGENTS_WRITE)

    raise NotFoundError("Agent", agent_id)


@router.get(
    "/{agent_id}/stats",
    response_model=APIResponse[Dict[str, Any]],
    summary="Get Agent Stats",
    description="Get usage statistics for an agent.",
)
async def get_agent_stats(
    agent_id: str = Path(..., description="Agent ID"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    auth: AuthContext = Depends(),
):
    """Get agent statistics."""
    auth.require_permission(Permission.ANALYTICS_READ)

    raise NotFoundError("Agent", agent_id)


@router.get(
    "/{agent_id}/versions",
    response_model=ListResponse[Dict[str, Any]],
    summary="List Agent Versions",
    description="List all versions/revisions of an agent.",
)
async def list_agent_versions(
    agent_id: str = Path(..., description="Agent ID"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    auth: AuthContext = Depends(),
):
    """List agent version history."""
    auth.require_permission(Permission.AGENTS_READ)

    raise NotFoundError("Agent", agent_id)


@router.post(
    "/{agent_id}/versions/{version_id}/restore",
    response_model=APIResponse[AgentResponse],
    summary="Restore Agent Version",
    description="Restore an agent to a previous version.",
)
async def restore_agent_version(
    agent_id: str = Path(..., description="Agent ID"),
    version_id: str = Path(..., description="Version ID to restore"),
    auth: AuthContext = Depends(),
):
    """Restore an agent to a previous version."""
    auth.require_permission(Permission.AGENTS_WRITE)

    raise NotFoundError("Agent", agent_id)
