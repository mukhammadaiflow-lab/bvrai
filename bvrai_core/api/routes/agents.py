"""
Agent API Routes

This module provides REST API endpoints for managing voice agents.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field

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
):
    """Create a new voice agent."""
    auth.require_permission(Permission.AGENTS_WRITE)

    # In production, this would:
    # 1. Validate voice_id with provider
    # 2. Create agent in database
    # 3. Initialize agent runtime

    agent = AgentResponse(
        id="agt_" + "x" * 24,
        organization_id=auth.organization_id,
        name=request.name,
        description=request.description,
        system_prompt=request.system_prompt,
        first_message=request.first_message,
        industry=request.industry,
        voice=request.voice or VoiceConfig(voice_id="default"),
        llm=request.llm or LLMConfig(),
        behavior=request.behavior or BehaviorConfig(),
        transcription=request.transcription or TranscriptionConfig(),
        knowledge_base_ids=request.knowledge_base_ids,
        functions=request.functions,
        metadata=request.metadata,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    return success_response(agent.dict())


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
):
    """List all agents."""
    auth.require_permission(Permission.AGENTS_READ)

    # In production, this would query the database
    agents = []  # Query results

    return paginated_response(
        items=[a.dict() for a in agents],
        page=page,
        page_size=page_size,
        total_items=0,
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
):
    """Get agent by ID."""
    auth.require_permission(Permission.AGENTS_READ)

    # In production, this would query the database
    # For now, return not found
    raise NotFoundError("Agent", agent_id)


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
):
    """Update an agent."""
    auth.require_permission(Permission.AGENTS_WRITE)

    # In production, this would update the database
    raise NotFoundError("Agent", agent_id)


@router.delete(
    "/{agent_id}",
    status_code=204,
    summary="Delete Agent",
    description="Delete an agent. This cannot be undone.",
)
async def delete_agent(
    agent_id: str = Path(..., description="Agent ID"),
    auth: AuthContext = Depends(),
):
    """Delete an agent."""
    auth.require_permission(Permission.AGENTS_DELETE)

    # In production, this would:
    # 1. Check no active calls
    # 2. Soft delete or hard delete based on policy
    raise NotFoundError("Agent", agent_id)


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
